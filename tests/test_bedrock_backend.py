"""The Bedrock arm differs from the vLLM arm in where the LLM is served, and nothing else."""

import sys
import types
from pathlib import Path

import httpx
import openai
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path[:0] = [str(SRC), str(SRC / "search_agent")]

from tongyi_utils import llm_backend, react_agent  # noqa: E402
from tongyi_utils.llm_backend import BedrockBackend, FatalBackendError, VllmBackend, merged_content  # noqa: E402

REQUEST = httpx.Request("POST", "https://bedrock.example/v1/chat/completions")


def api_error(cls, status, message):
    return cls(message, response=httpx.Response(status, request=REQUEST), body=None)


def completion(content="ok", reasoning=None, finish="stop", usage=(10, 5)):
    message = types.SimpleNamespace(
        model_dump=lambda exclude_none=True: {k: v for k, v in
                                              {"content": content, "reasoning_content": reasoning}.items() if v is not None})
    choice = types.SimpleNamespace(message=message, finish_reason=finish)
    return types.SimpleNamespace(choices=[choice], usage=types.SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]))


class FakeClient:
    """Scripted `client.chat.completions.create`: a list of responses or exceptions."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


CFG = {"temperature": 0.6, "top_p": 0.95, "presence_penalty": 1.1}
MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]


def generate(backend, **kw):
    return backend.generate(MSGS, model="m", max_tokens=4096, generate_cfg=CFG, enable_thinking=True, **kw)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm_backend.time, "sleep", lambda s: None)


def test_request_keeps_the_released_runs_sampling_settings():
    client = FakeClient([completion()])
    assert generate(BedrockBackend(client=client)) == ("ok", "stop")
    sent = client.calls[0]
    assert (sent["temperature"], sent["top_p"], sent["presence_penalty"], sent["seed"]) == (0.6, 0.95, 1.1, 2026)
    assert sent["stop"] == ["\n<tool_response>", "<tool_response>"] and sent["max_tokens"] == 4096
    assert "logprobs" not in sent and "extra_body" not in sent      # vLLM-only fields are never sent


def test_rejected_parameters_are_dropped_once_and_reported():
    client = FakeClient([api_error(openai.BadRequestError, 400, "Unsupported parameter: presence_penalty"),
                         api_error(openai.BadRequestError, 400, "unknown field seed"), completion()])
    backend = BedrockBackend(client=client)
    generate(backend)
    assert "presence_penalty" not in client.calls[-1] and "seed" not in client.calls[-1]
    assert backend.report()["dropped_params"] == ["presence_penalty", "seed"]
    generate_again = FakeClient([completion()])
    backend._client = generate_again
    generate(backend)
    assert "presence_penalty" not in generate_again.calls[0]       # stays dropped for the whole run


def test_max_tokens_field_switches_when_the_endpoint_asks_for_max_completion_tokens():
    client = FakeClient([api_error(openai.BadRequestError, 400, "use max_completion_tokens instead of max_tokens"), completion()])
    backend = BedrockBackend(client=client)
    generate(backend)
    assert client.calls[-1]["max_completion_tokens"] == 4096 and "max_tokens" not in client.calls[-1]


def test_prefill_is_an_instruction_by_default_and_a_real_prefill_when_native():
    client = FakeClient([completion("<answer>x</answer>")])
    generate(BedrockBackend(client=client), prefill="<answer>")
    assert MSGS[-1]["content"] == "q"                               # the caller's messages are not mutated
    assert client.calls[0]["messages"][-1]["content"].endswith("Start your reply with <answer> and finish it with </answer>.")
    native = FakeClient([completion("x</answer>")])
    content, _ = generate(BedrockBackend(client=native, prefill_mode="native"), prefill="<answer>")
    assert content == "<answer>x</answer>"
    assert native.calls[0]["messages"][-1] == {"role": "assistant", "content": "<answer>"}
    assert native.calls[0]["extra_body"]["continue_final_message"] is True


def test_separate_reasoning_is_restored_inside_think_tags():
    message = types.SimpleNamespace(model_dump=lambda exclude_none=True: {"content": "<tool_call>{}</tool_call>", "reasoning_content": "plan"})
    assert merged_content(message) == "<think>\nplan\n</think>\n<tool_call>{}</tool_call>"
    inline = types.SimpleNamespace(model_dump=lambda exclude_none=True: {"content": "<think>a</think>b", "reasoning": "dup"})
    assert merged_content(inline) == "<think>a</think>b"


def test_empty_and_throttled_responses_are_retried_then_succeed():
    client = FakeClient([completion(""), api_error(openai.RateLimitError, 429, "slow down"), completion("done")])
    assert generate(BedrockBackend(client=client))[0] == "done"
    assert len(client.calls) == 3


def test_expired_key_stops_the_run_instead_of_failing_each_question(monkeypatch):
    client = FakeClient([api_error(openai.AuthenticationError, 401, "Signature expired")] * 2)
    backend = BedrockBackend(client=client)
    monkeypatch.setattr(backend, "client", lambda refresh=False: client)
    with pytest.raises(FatalBackendError, match="regenerate"):
        generate(backend)


def test_usage_is_counted_per_query_view():
    backend = BedrockBackend(client=FakeClient([completion(usage=(100, 20)), completion(usage=(7, 3))]))
    first, second = backend.for_query(), backend.for_query()
    generate(first)
    generate(second)
    assert first.report()["usage"]["prompt_tokens"] == 100 and second.report()["usage"]["prompt_tokens"] == 7


def test_credentials_prefer_the_environment_and_add_the_v1_path(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("BEDROCK_MANTLE_OPENAI_BASE_URL=https://file.example\nBEDROCK_MANTLE_OPENAI_API_KEY=file-key\n")
    monkeypatch.delenv("BEDROCK_MANTLE_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("BEDROCK_MANTLE_OPENAI_API_KEY", raising=False)
    assert llm_backend.credentials(str(env)) == ("https://file.example/v1", "file-key")
    monkeypatch.setenv("BEDROCK_MANTLE_OPENAI_BASE_URL", "https://env.example/v1/")
    monkeypatch.setenv("BEDROCK_MANTLE_OPENAI_API_KEY", "env-key")
    assert llm_backend.credentials(str(env)) == ("https://env.example/v1", "env-key")
    monkeypatch.delenv("BEDROCK_MANTLE_OPENAI_API_KEY")
    with pytest.raises(FatalBackendError):
        llm_backend.credentials(None)


def test_vllm_backend_still_sends_the_original_request(monkeypatch):
    client = FakeClient([completion("hi")])
    monkeypatch.setattr(llm_backend, "OpenAI", lambda **kw: client)
    VllmBackend(6008).generate(MSGS, model="m", max_tokens=1024, generate_cfg=CFG, enable_thinking=True, prefill="<answer>")
    sent = client.calls[0]
    assert sent["logprobs"] is True and sent["seed"] == 2026 and sent["top_p"] == 0.95
    assert sent["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True},
                                  "continue_final_message": True, "add_generation_prompt": False}
    assert sent["messages"][-1] == {"role": "assistant", "content": "<answer>"}


class ScriptedBackend:
    """Replies in order, so the real ReAct loop runs without any model."""

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def generate(self, msgs, **kw):
        self.seen.append((list(msgs), kw))
        return self.replies.pop(0), "stop"

    def for_query(self):
        return self


class StubTool:
    def __init__(self):
        self.visited, self.calls = [], []

    def reset_trajectory(self, question): pass
    def set_current_thinking(self, text): pass
    def add_visit_reasoning(self, text): pass
    def add_found_docids(self, docids): pass
    def add_visited_docid(self, docid): self.visited.append(docid)
    def get_search_traces(self): return []

    def call(self, params, **kw):
        self.calls.append(params)
        return ("[]", ["d1"]) if "query" in params else ("Document d1:\ntext.", ["d1"])


def agent(monkeypatch, replies, **llm):
    monkeypatch.setattr(react_agent.AutoTokenizer, "from_pretrained",
                        lambda *a, **k: types.SimpleNamespace(apply_chat_template=lambda m, **kw: [0] * sum(len(x["content"]) for x in m)))
    search, doc, backend = StubTool(), StubTool(), ScriptedBackend(replies)
    cfg = {"model": "tok", "generate_cfg": {}, "backend": backend, **llm}
    return react_agent.MultiTurnReactAgent(llm=cfg, search_tool_handler=search, get_document_handler=doc), backend, search, doc


QUESTION = {"item": {"question": "Who?", "answer": ""}, "planning_port": None}


def test_react_loop_runs_tools_and_stops_on_an_answer(monkeypatch):
    call = lambda body: "<think>t</think>\n<tool_call>\n" + body + "\n</tool_call>"
    a, backend, search, doc = agent(monkeypatch, [call('{"name": "search", "arguments": {"query": "x"}}'),
                                                  call('{"name": "get_document", "arguments": {"docid": "d1"}}'),
                                                  "<answer>Ada [DocID:d1]</answer>"])
    result = a._run(QUESTION, "m")
    assert result["termination"] == "answer" and result["prediction"].startswith("Ada")
    assert result["tool_call_counts"] == {"search": 1, "get_document": 1} and doc.calls == [{"docid": "d1"}]
    assert backend.seen[0][0][0]["content"] == react_agent.SYSTEM_PROMPT_SEARCH_ONLY        # the released system prompt, unchanged
    assert backend.seen[0][0][1]["content"] == "Who?"
    assert [kw["max_tokens"] for _, kw in backend.seen] == [4096, 2048, 1024]                 # the per-turn reasoning caps


def test_attributed_condition_appends_the_requirement_everywhere(monkeypatch):
    a, backend, *_ = agent(monkeypatch, ["<answer>x</answer>"], attributed=True)
    a._run(QUESTION, "m")
    system, user = backend.seen[0][0][0]["content"], backend.seen[0][0][1]["content"]
    assert system == react_agent.SYSTEM_PROMPT_SEARCH_ONLY + react_agent.ATTRIBUTION_REQUIREMENT
    assert user == "Who?" + react_agent.ATTRIBUTION_REMINDER


def test_context_budget_forces_the_final_answer_with_a_prefill(monkeypatch):
    call = "<tool_call>\n{\"name\": \"search\", \"arguments\": {\"query\": \"x\"}}\n</tool_call>"
    a, backend, *_ = agent(monkeypatch, [call, "<answer>done</answer>"], max_context_tokens=10, attributed=True)
    result = a._run(QUESTION, "m")
    messages, kw = backend.seen[-1]
    assert kw["prefill"] == "<answer>" and result["termination"] == "generate an answer as token limit reached"
    assert messages[-1]["content"].startswith("Retrieval complete.") and messages[-1]["content"].endswith(react_agent.ATTRIBUTION_REMINDER)


def test_a_parameter_already_dropped_by_another_thread_is_retried_not_raised():
    backend = BedrockBackend(client=FakeClient([]))
    error = api_error(openai.BadRequestError, 400, "Unsupported parameter: presence_penalty")
    stale = {"presence_penalty": 1.1}            # built before the other thread dropped it
    backend.dropped.add("presence_penalty")
    assert backend.adapt(error, stale) is True
