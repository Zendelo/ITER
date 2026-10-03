"""Run the real `tongyi_client.py` against a local fake Bedrock endpoint (no network, no model, no index).

The fake server speaks the OpenAI chat-completions protocol, rejects `presence_penalty` once (as a hosted
endpoint may), returns its reasoning in a separate field, and replies with a search, a document read and an
answer. Set ITER_E2E_OUT to keep the trajectory JSON for inspection.
"""

import json
import os
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path[:0] = [str(SRC), str(SRC / "search_agent")]

DOCS = {"d1": {"docid": "d1", "text": "Ada Lovelace wrote the first program.", "score": 1.0},
        "d2": {"docid": "d2", "text": "Charles Babbage designed the engine.", "score": 0.5}}


class StubSearcher:
    @classmethod
    def parse_args(cls, parser):
        parser.add_argument("--index-path", default="none")

    def __init__(self, args):
        pass

    def search(self, query, k=10, **kw):
        return list(DOCS.values())[:k]

    def get_document(self, docid):
        return DOCS.get(docid)

    search_type = "stub"


class FakeBedrock(BaseHTTPRequestHandler):
    requests: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if "presence_penalty" in body:               # like a hosted endpoint: rejected every time it is sent
            return self.reply(400, {"error": {"message": "Unsupported parameter: presence_penalty", "type": "invalid_request_error"}})
        FakeBedrock.requests.append(body)
        turn = sum(m["role"] == "assistant" for m in body["messages"])
        replies = [
            ("<tool_call>\n{\"name\": \"search\", \"arguments\": {\"query\": \"first program\"}}\n</tool_call>", "I should search."),
            ("<tool_call>\n{\"name\": \"get_document\", \"arguments\": {\"docid\": \"d1\"}}\n</tool_call>", "Read the top hit."),
            ("<answer>\nAda Lovelace [DocID:d1]\n</answer>", "Enough evidence."),
        ]
        content, reasoning = replies[min(turn, 2)]
        message = {"role": "assistant", "content": content, "reasoning_content": reasoning}
        self.reply(200, {"id": "x", "object": "chat.completion", "model": body["model"],
                         "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}})

    def reply(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def server():
    FakeBedrock.requests = []
    httpd = HTTPServer(("127.0.0.1", 0), FakeBedrock)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_bedrock_arm_end_to_end(server, tmp_path, monkeypatch):
    import transformers

    fake_tokenizer = types.SimpleNamespace(
        encode=lambda text, add_special_tokens=False: list(range(len(text.split()))),
        decode=lambda ids, skip_special_tokens=True: " ".join("w" for _ in ids),
        apply_chat_template=lambda msgs, **kw: [0] * sum(len(m["content"]) for m in msgs),
    )
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: fake_tokenizer)
    searcher_module = types.ModuleType("searcher")
    searcher_module.SearcherType = types.SimpleNamespace(get_choices=lambda: ["stub"], get_searcher_class=lambda name: StubSearcher)
    monkeypatch.setitem(sys.modules, "searcher", searcher_module)

    queries = tmp_path / "q.tsv"
    queries.write_text("q1\tWho wrote the first program?\nq2\tWho designed the engine?\n")
    out = Path(os.environ.get("ITER_E2E_OUT", tmp_path / "out"))
    monkeypatch.setenv("BEDROCK_MANTLE_OPENAI_BASE_URL", server)
    monkeypatch.setenv("BEDROCK_MANTLE_OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(sys, "argv", [
        "tongyi_client.py", "--llm-backend", "bedrock", "--model", "fake.model-v1", "--query", str(queries),
        "--output-dir", str(out), "--searcher-type", "stub", "--num-threads", "2", "--snippet-max-tokens", "64",
        "--k", "10", "--attributed", "--max-visit-tokens", "12000", "--max-context-tokens", "120000"])

    import tongyi_client
    tongyi_client.main()

    runs = sorted(out.glob("run_*.json"))
    assert len(runs) == 2
    record = json.loads(runs[0].read_text())
    assert record["status"] == "completed"
    assert record["tool_call_counts"] == {"search": 1, "get_document": 1}
    meta = record["metadata"]
    assert (meta["model"], meta["attributed"], meta["max_visit_tokens"], meta["max_context_tokens"], meta["k"],
            meta["snippet_max_tokens"], meta["max_llm_calls"]) == ("fake.model-v1", True, 12000, 120000, 10, 64, 50)
    assert meta["llm_backend"]["backend"] == "bedrock-mantle" and meta["llm_backend"]["dropped_params"] == ["presence_penalty"]
    assert meta["llm_backend"]["usage"]["calls"] == 3 and meta["llm_backend"]["usage"]["prompt_tokens"] == 300
    system, user = record["raw_messages"][0]["content"], record["raw_messages"][1]["content"]
    assert "# ATTRIBUTION REQUIREMENT" in system and "[DocID:12345]" in user
    assert "<think>\nI should search.\n</think>" in record["raw_messages"][2]["content"]    # reasoning kept in the usual form
    # ITER's own parser reads the final message as the answer, so only the earlier turns yield reasoning items.
    assert [r["type"] for r in record["result"]] == ["reasoning", "tool_call", "reasoning", "tool_call", "output_text"]
    assert record["retrieved_docids"] == ["d1", "d2"]
    assert all("presence_penalty" not in r for r in FakeBedrock.requests)
