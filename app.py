from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).parent
STATE_PATH = ROOT / 'training_state.json'
TOKEN_PATTERN = re.compile(r'[^a-z0-9\s]')
NAME_PATTERN = re.compile(r"(?:my name is|i am|i'm)\s+([a-zA-Z]+)", re.IGNORECASE)
SAFE_MATH_PATTERN = re.compile(r'^[0-9\s\.+\-\*/\(\)]+$')


class ChatEngine:
    def __init__(self) -> None:
        self.user_name = ''
        self.topics = []
        self.response_style = 'balanced'
        self.conversation_samples = []
        self.learned_map = {}
        self.recent_intents = []
        self.metrics = {
            'turns': 0,
            'avg_input_length': 0.0,
            'positive_feedback': 0,
            'negative_feedback': 0,
            'math_requests': 0,
            'plan_requests': 0,
            'summary_requests': 0,
        }
        self.last_assistant_reply = ''
        self.last_user_input = ''
        self.load_state()

    def tokenize(self, text: str):
        return [t for t in TOKEN_PATTERN.sub(' ', text.lower()).split() if t]

    def remember_topic(self, text: str) -> None:
        keywords = [w for w in self.tokenize(text) if len(w) > 4]
        for kw in keywords:
            if kw not in self.topics:
                self.topics.append(kw)
        self.topics = self.topics[-16:]

    def update_metrics(self, user_text: str) -> None:
        self.metrics['turns'] += 1
        length = len(user_text.strip())
        turns = self.metrics['turns']
        avg = self.metrics['avg_input_length']
        self.metrics['avg_input_length'] = ((avg * (turns - 1)) + length) / turns

        if self.metrics['avg_input_length'] > 150:
            self.response_style = 'detailed'
        elif self.metrics['avg_input_length'] < 45:
            self.response_style = 'concise'
        else:
            self.response_style = 'balanced'

    def get_tone_prefix(self) -> str:
        return {'concise': 'Quick answer: ', 'detailed': 'Detailed answer: '}.get(self.response_style, '')

    def summarize_text(self, content: str) -> str:
        self.metrics['summary_requests'] += 1
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content) if s.strip()]
        if len(sentences) <= 1:
            return f'Summary: {content}'
        picks = [sentences[0], sentences[len(sentences) // 2], sentences[-1]]
        out = []
        for p in picks:
            if p not in out:
                out.append(p)
        return f"Summary: {' '.join(out)}"

    def build_plan(self, user_input: str) -> str:
        self.metrics['plan_requests'] += 1
        goal = re.sub(r'^plan\s*', '', user_input, flags=re.IGNORECASE).strip() or 'your goal'
        complexity = 'small' if len(goal.split()) < 5 else 'medium'
        return (
            f'Plan for {goal} ({complexity} scope):\n'
            '1) Define outcome and success metric.\n'
            '2) Build the smallest testable version.\n'
            '3) Run a user feedback loop quickly.\n'
            '4) Prioritize one high-impact improvement each cycle.\n'
            '5) Review metrics and iterate weekly.'
        )

    def extract_math_expression(self, user_input: str) -> str:
        lowered = user_input.lower().strip()
        prefixes = ('calculate ', 'calc ', 'solve ', 'math ')
        for prefix in prefixes:
            if lowered.startswith(prefix):
                return user_input[len(prefix):].strip()
        return ''

    def solve_math(self, expression: str) -> str:
        cleaned = expression.replace('^', '**').strip()
        if not cleaned or not SAFE_MATH_PATTERN.match(cleaned):
            return ''
        try:
            value = eval(cleaned, {'__builtins__': {}}, {})
        except Exception:
            return ''
        self.metrics['math_requests'] += 1
        return f'Result: {value}'

    def detect_intent(self, text: str) -> str:
        if text.startswith('/teach'):
            return 'teach'
        if text in ('/good', '/bad', '/resettrain', '/train'):
            return text[1:]
        if text.startswith('summarize '):
            return 'summarize'
        if text.startswith('plan'):
            return 'plan'
        if re.search(r'\b(calculate|calc|solve|math)\b', text):
            return 'math'
        if re.search(r'\b(price|pricing|cost|plan)\b', text):
            return 'pricing'
        if re.search(r'\b(contact|support|email|help desk)\b', text):
            return 'support'
        if re.search(r'\b(refund|cancel|cancellation)\b', text):
            return 'refund'
        if re.search(r'\b(roadmap|next|future|upcoming)\b', text):
            return 'roadmap'
        return 'generic'

    def remember_intent(self, intent: str) -> None:
        self.recent_intents.append(intent)
        self.recent_intents = self.recent_intents[-8:]

    def match_learned_reply(self, user_input: str) -> str:
        norm = ' '.join(self.tokenize(user_input))
        if not norm:
            return ''
        input_tokens = set(norm.split())
        best_score = 0.0
        best_reply = ''
        for key, reply in self.learned_map.items():
            key_tokens = set(key.split())
            score = len(key_tokens & input_tokens) / max(1, len(key_tokens))
            if score > best_score:
                best_score = score
                best_reply = reply
        return f'From training memory: {best_reply}' if best_score >= 0.6 else ''

    def train_from_conversation(self, user_text: str, ai_reply: str) -> None:
        self.conversation_samples.append({'user_text': user_text, 'ai_reply': ai_reply})
        self.conversation_samples = self.conversation_samples[-12:]

        tokens = self.tokenize(user_text)
        if 'short' in tokens or 'brief' in tokens:
            self.response_style = 'concise'
        if any(t in tokens for t in ('detailed', 'deep', 'explain')):
            self.response_style = 'detailed'

        key = ' '.join(self.tokenize(user_text)[:10])
        if len(key) > 8 and key not in self.learned_map:
            self.learned_map[key] = ai_reply

    def teach_from_input(self, user_input: str) -> str:
        payload = re.sub(r'^/teach\s*', '', user_input, flags=re.IGNORECASE).strip()
        if '=>' not in payload:
            return 'Usage: /teach question => answer'
        question, answer = [p.strip() for p in payload.split('=>', 1)]
        if not question or not answer:
            return 'Please provide both question and answer. Example: /teach refund policy => Our refund window is 30 days.'
        key = ' '.join(self.tokenize(question)[:12])
        if not key:
            return 'I could not parse that training question. Try longer text.'
        self.learned_map[key] = answer
        return f'Learned new pair: "{question}"'

    def apply_feedback(self, kind: str) -> str:
        if not self.last_user_input or not self.last_assistant_reply:
            return 'No recent reply to rate yet. Send a message first.'
        key = ' '.join(self.tokenize(self.last_user_input)[:12])
        if not key:
            return 'Unable to apply feedback for that interaction.'
        if kind == 'good':
            self.metrics['positive_feedback'] += 1
            self.learned_map[key] = self.last_assistant_reply
            return 'Thanks! I reinforced that reply pattern.'
        self.metrics['negative_feedback'] += 1
        self.learned_map.pop(key, None)
        self.response_style = 'balanced'
        return 'Got it. I removed that learned mapping and reset style balance.'

    def reset_training_memory(self) -> str:
        self.learned_map = {}
        self.metrics['positive_feedback'] = 0
        self.metrics['negative_feedback'] = 0
        return 'Training memory reset complete.'

    def profile_summary(self) -> str:
        quality = self.metrics['positive_feedback'] - self.metrics['negative_feedback']
        return (
            'Training profile:\n'
            f'- Turns: {self.metrics["turns"]}\n'
            f'- Learned pairs: {len(self.learned_map)}\n'
            f'- Quality score: {quality}\n'
            f'- Intent history: {", ".join(self.recent_intents[-5:]) or "none"}\n'
            f'- Tools usage (plan/summarize/math): {self.metrics["plan_requests"]}/{self.metrics["summary_requests"]}/{self.metrics["math_requests"]}'
        )

    def rule_based_info_reply(self, intent: str) -> str:
        rules = {
            'pricing': 'Pricing guidance: start with a free tier, then add usage-based paid plans for power users.',
            'support': 'Support guidance: share a contact email, expected response SLA, and FAQ for common questions.',
            'refund': 'Refund guidance: use a clear policy window (e.g., 14 or 30 days) and document exclusions.',
            'roadmap': 'Roadmap guidance: focus next on reliability, analytics, and user-requested integrations.'
        }
        return rules.get(intent, '')

    def generate_reply(self, user_input: str) -> str:
        text = user_input.lower().strip()
        intent = self.detect_intent(text)
        self.remember_intent(intent)

        if text == '/train':
            sc = len(self.conversation_samples)
            lc = len(self.learned_map)
            qs = self.metrics['positive_feedback'] - self.metrics['negative_feedback']
            return f"Training complete for this session. Learned from {sc} samples, {len(self.topics)} topic signals, {lc} saved training pairs, and a quality score of {qs}."
        if text == '/profile':
            return self.profile_summary()
        if text == '/good':
            return self.apply_feedback('good')
        if text == '/bad':
            return self.apply_feedback('bad')
        if text == '/resettrain':
            return self.reset_training_memory()
        if text.startswith('/teach'):
            return self.teach_from_input(user_input)

        math_expr = self.extract_math_expression(user_input)
        if math_expr:
            math_reply = self.solve_math(math_expr)
            if math_reply:
                return math_reply

        learned = self.match_learned_reply(user_input)
        if learned:
            return learned

        name_match = NAME_PATTERN.search(user_input)
        if name_match:
            self.user_name = name_match.group(1)
            return f"Nice to meet you, {self.user_name}! I'll remember your name during this session."

        if re.search(r'\b(hello|hi|hey)\b', text):
            suffix = f' {self.user_name}' if self.user_name else ''
            return f'{self.get_tone_prefix()}Hey{suffix}! How can I help you today?'
        if re.search(r'\b(who are you|what are you)\b', text):
            return 'I am an offline web AI assistant using a Python backend with adaptive training behavior.'
        if re.search(r'\b(help|what can you do)\b', text):
            return 'I can chat, summarize, plan, solve basic math (`calculate ...`), and learn from /teach, /good, /bad, and /profile.'

        if text.startswith('summarize '):
            content = user_input[10:].strip()
            return self.summarize_text(content) if content else 'Please add text after "summarize".'
        if text.startswith('plan'):
            return self.build_plan(user_input)

        info_reply = self.rule_based_info_reply(intent)
        if info_reply:
            return f'{self.get_tone_prefix()}{info_reply}'

        if re.search(r'\b(idea|brainstorm)\b', text):
            seed = self.topics[-1] if self.topics else 'your project'
            return f"{self.get_tone_prefix()}Brainstorm for {seed}:\n1) Build one focused MVP flow\n2) Add a simple onboarding checklist\n3) Collect feedback from first 10 users\n4) Ship one visible improvement per week"

        recent = ', '.join(self.topics[-3:])
        replies = [
            f"{self.get_tone_prefix()}Interesting. Tell me more{', ' + self.user_name if self.user_name else ''}.",
            f'{self.get_tone_prefix()}I can convert this into a step-by-step plan if you want.',
            f'{self.get_tone_prefix()}Want a short answer or a deeper breakdown?',
            f'{self.get_tone_prefix()}I noticed recurring topics: {recent}. Want to focus on one?' if recent else f'{self.get_tone_prefix()}Share a bit more context and I can respond more precisely.',
        ]
        return replies[self.metrics['turns'] % len(replies)]

    def process(self, user_input: str) -> str:
        self.update_metrics(user_input)
        self.remember_topic(user_input)
        reply = self.generate_reply(user_input)
        self.train_from_conversation(user_input, reply)
        self.last_user_input = user_input
        self.last_assistant_reply = reply
        self.save_state()
        return reply

    def state_payload(self):
        quality = self.metrics['positive_feedback'] - self.metrics['negative_feedback']
        return {
            'topics': self.topics[-3:],
            'learned_pairs': len(self.learned_map),
            'quality_score': quality,
            'response_style': self.response_style,
            'recent_intents': self.recent_intents[-5:],
            'turns': self.metrics['turns'],
        }

    def load_state(self) -> None:
        if not STATE_PATH.exists():
            return
        try:
            payload = json.loads(STATE_PATH.read_text())
        except json.JSONDecodeError:
            return
        self.learned_map = payload.get('learned_map', {})
        self.response_style = payload.get('response_style', self.response_style)
        metrics = payload.get('metrics', {})
        self.metrics['positive_feedback'] = int(metrics.get('positive_feedback', 0))
        self.metrics['negative_feedback'] = int(metrics.get('negative_feedback', 0))
        self.metrics['math_requests'] = int(metrics.get('math_requests', 0))
        self.metrics['plan_requests'] = int(metrics.get('plan_requests', 0))
        self.metrics['summary_requests'] = int(metrics.get('summary_requests', 0))

    def save_state(self) -> None:
        STATE_PATH.write_text(json.dumps({
            'learned_map': self.learned_map,
            'response_style': self.response_style,
            'metrics': {
                'positive_feedback': self.metrics['positive_feedback'],
                'negative_feedback': self.metrics['negative_feedback'],
                'math_requests': self.metrics['math_requests'],
                'plan_requests': self.metrics['plan_requests'],
                'summary_requests': self.metrics['summary_requests'],
            },
        }, indent=2))


ENGINE = ChatEngine()


class AppHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: Path, content_type: str = 'text/html; charset=utf-8') -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404)
            return
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path == '/api/state':
            self._send_json(ENGINE.state_payload())
            return
        if path in ('/', '/index.html'):
            self._send_file(ROOT / 'index.html')
            return
        self.send_error(404)

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        if path != '/api/chat':
            self.send_error(404)
            return

        content_length = int(self.headers.get('Content-Length', '0'))
        raw = self.rfile.read(content_length)
        try:
            data = json.loads(raw.decode('utf-8')) if raw else {}
        except json.JSONDecodeError:
            self._send_json({'error': 'Invalid JSON payload.'}, 400)
            return

        message = str(data.get('message', '')).strip()
        if not message:
            self._send_json({'error': 'Message is required.'}, 400)
            return

        reply = ENGINE.process(message)
        self._send_json({'reply': reply, 'state': ENGINE.state_payload()})


def run() -> None:
    server = ThreadingHTTPServer(('0.0.0.0', 8000), AppHandler)
    print('Serving BOLT-AI backend on http://0.0.0.0:8000')
    server.serve_forever()


if __name__ == '__main__':
    run()
