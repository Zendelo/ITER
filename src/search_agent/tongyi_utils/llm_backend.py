"""Where the agent's LLM is served. The agent loop, prompts, tools and budgets do not change.

`VllmBackend` is the original path (a local vLLM server) and keeps its request byte for byte.
`BedrockBackend` sends the same request to an API-served model on Amazon Bedrock (Mantle's
OpenAI-compatible endpoint). The Tongyi ReAct protocol is plain text (`<think>`, `<tool_call>`,
`<answer>`), so any chat model can run it; the backend only translates what a hosted API
cannot do like vLLM:

* vLLM-only request fields (`chat_template_kwargs`, `continue_final_message`, `logprobs`) are not sent.
* A parameter the endpoint rejects (for example `presence_penalty` or `seed`) is dropped and
  recorded in `report()["dropped_params"]`, so a run states exactly where it deviated.
* A model that returns its reasoning in a separate field has it put back inside `<think>` tags,
  the form every downstream stage (trajectory parsing, ITER's `i6`/`i7` queries) reads.
* The assistant prefill used to force the final `<answer>` is an instruction by default
  (`prefill_mode="instruction"`); `native` sends a real assistant prefill if the endpoint allows it.
"""

import logging
import os
import random
import threading
import time
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

logger = logging.getLogger(__name__)

STOP = ["\n<tool_response>", "<tool_response>"]
BASE_URL_ENV = "BEDROCK_MANTLE_OPENAI_BASE_URL"
API_KEY_ENV = "BEDROCK_MANTLE_OPENAI_API_KEY"
# Request fields a hosted endpoint may refuse; each is dropped when named in a 400 response.
DROPPABLE = ("presence_penalty", "top_p", "seed", "stop", "temperature", "reasoning_effort")
FORCE_ANSWER = "\n\nStart your reply with <answer> and finish it with </answer>."


class FatalBackendError(RuntimeError):
    """The backend cannot continue (for example an expired key); the run must stop, not skip questions."""


class VllmBackend:
    """The original local vLLM call, unchanged."""

    def __init__(self, port: int | None = None):
        self.port = port

    def report(self) -> dict:
        return {"backend": "vllm"}

    def for_query(self):
        return self

    def generate(self, msgs, *, model, max_tokens, generate_cfg, enable_thinking, prefill=None, port=None, max_tries=10):
        client = OpenAI(api_key="EMPTY", base_url=f"http://127.0.0.1:{port or self.port}/v1", timeout=600.0)
        base_sleep_time = 1
        call_msgs = msgs
        extra_body = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
        if prefill:
            call_msgs = msgs + [{"role": "assistant", "content": prefill}]
            extra_body["continue_final_message"] = True
            extra_body["add_generation_prompt"] = False
        for attempt in range(max_tries):
            try:
                chat_response = client.chat.completions.create(
                    model=model,
                    messages=call_msgs,
                    stop=STOP,
                    temperature=generate_cfg.get("temperature", 0.6),
                    top_p=0.95,
                    logprobs=True,
                    max_tokens=max_tokens,
                    seed=2026,
                    presence_penalty=generate_cfg.get("presence_penalty", 1.1),
                    extra_body=extra_body,
                )
                content = chat_response.choices[0].message.content
                finish_reason = chat_response.choices[0].finish_reason
                if content and content.strip():
                    if prefill:
                        content = prefill + content
                    return content.strip(), finish_reason
                logger.warning("Attempt %s received an empty response. enable_thinking=%s message=%s",
                               attempt + 1, enable_thinking, chat_response.choices[0].message)
            except (APIError, APIConnectionError, APITimeoutError) as e:
                logger.warning("Attempt %s failed with an API/network error: %s", attempt + 1, e)
            except Exception as e:
                logger.warning("Attempt %s failed with an unexpected error: %s", attempt + 1, e)
            if attempt < max_tries - 1:
                sleep_time = min(base_sleep_time * (2 ** attempt) + random.uniform(0, 1), 30)
                logger.info("Retrying in %.2f seconds...", sleep_time)
                time.sleep(sleep_time)
            else:
                logger.error("All retry attempts have been exhausted. The call has failed.")
        return "vllm server error!!!", "error"


def normalise_base_url(url: str) -> str:
    """The Mantle endpoint is served under `/v1`; accept the URL with or without it."""
    url = url.strip().rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


def credentials(env_file: str | None, base_url_env: str = BASE_URL_ENV, api_key_env: str = API_KEY_ENV):
    """Base URL and key: process environment first, then `env_file`. Never taken from the command line."""
    from dotenv import dotenv_values

    file_values = dotenv_values(env_file) if env_file else {}
    base = os.environ.get(base_url_env) or file_values.get(base_url_env)
    key = os.environ.get(api_key_env) or file_values.get(api_key_env)
    if not base or not key:
        raise FatalBackendError(f"set {base_url_env} and {api_key_env} (environment or --env-file)")
    return normalise_base_url(base), key


class BedrockBackend:
    """The same request served by an API model on Bedrock (OpenAI-compatible Mantle endpoint)."""

    def __init__(self, *, env_file=None, base_url_env=BASE_URL_ENV, api_key_env=API_KEY_ENV,
                 seed: int | None = 2026, timeout: float = 600.0, prefill_mode: str = "instruction",
                 reasoning_effort: str | None = None, client: Any = None):
        if prefill_mode not in ("instruction", "native"):
            raise ValueError("prefill_mode must be 'instruction' or 'native'")
        self.env_file, self.base_url_env, self.api_key_env = env_file, base_url_env, api_key_env
        self.seed, self.timeout, self.prefill_mode, self.reasoning_effort = seed, timeout, prefill_mode, reasoning_effort
        self._client = client
        self._lock = threading.Lock()
        self.dropped: set[str] = set()
        self.max_tokens_field = "max_tokens"
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
        self.parent = None

    def for_query(self):
        """A view with its own usage counters (so each trajectory records its own cost) that shares
        the client, the dropped-parameter record and the endpoint adaptations."""
        view = object.__new__(BedrockBackend)
        view.__dict__.update(self.__dict__)
        view.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
        view._lock = threading.Lock()
        view.client = self.client          # one shared client, rebuilt for every view on key refresh
        view.parent = self
        return view

    def report(self) -> dict:
        return {"backend": "bedrock-mantle", "dropped_params": sorted(self.dropped),
                "max_tokens_field": self.max_tokens_field, "prefill_mode": self.prefill_mode,
                "usage": {**self.usage, "seconds": round(self.usage["seconds"], 3)}}

    def client(self, refresh: bool = False) -> OpenAI:
        with self._lock:
            if self._client is None or refresh:
                base, key = credentials(self.env_file, self.base_url_env, self.api_key_env)
                self._client = OpenAI(base_url=base, api_key=key, timeout=self.timeout, max_retries=0)
            return self._client

    def request(self, msgs, *, model, max_tokens, generate_cfg, prefill):
        call_msgs = list(msgs)
        extra_body: dict[str, Any] = {}
        if prefill and self.prefill_mode == "native":
            call_msgs.append({"role": "assistant", "content": prefill})
            extra_body = {"continue_final_message": True, "add_generation_prompt": False}
        elif prefill:
            last = dict(call_msgs[-1])
            last["content"] = f"{last['content']}{FORCE_ANSWER}"
            call_msgs[-1] = last
        params: dict[str, Any] = {
            "model": model, "messages": call_msgs, self.max_tokens_field: max_tokens,
            "stop": STOP, "temperature": generate_cfg.get("temperature", 0.6), "top_p": generate_cfg.get("top_p", 0.95),
            "presence_penalty": generate_cfg.get("presence_penalty", 1.1),
        }
        if self.seed is not None:
            params["seed"] = self.seed
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
        if extra_body:
            params["extra_body"] = extra_body
        return {k: v for k, v in params.items() if k not in self.dropped}

    def adapt(self, error: BadRequestError, params: dict) -> bool:
        """Drop one parameter the endpoint named in its 400 response; False if it named none."""
        text = str(error).lower()
        if "max_tokens" in text and "max_completion_tokens" in text and self.max_tokens_field == "max_tokens":
            self.max_tokens_field = "max_completion_tokens"
            if self.parent is not None:
                self.parent.max_tokens_field = self.max_tokens_field    # later queries start from it
            return True
        for name in DROPPABLE:
            if name in text and name in params:         # another thread may have dropped it already; retry
                self.dropped.add(name)
                logger.warning("Endpoint rejected %s; dropping it for this run (recorded in the metadata)", name)
                return True
        return False

    def generate(self, msgs, *, model, max_tokens, generate_cfg, enable_thinking, prefill=None, port=None, max_tries=10):
        refreshed = False
        attempt = 0
        while attempt < max_tries:
            params = self.request(msgs, model=model, max_tokens=max_tokens, generate_cfg=generate_cfg, prefill=prefill)
            started = time.monotonic()
            try:
                response = self.client().chat.completions.create(**params)
            except AuthenticationError as error:
                if refreshed:
                    raise FatalBackendError(
                        f"Bedrock rejected the API key ({error}); regenerate it and rerun - finished questions are skipped"
                    ) from error
                refreshed = True
                self.client(refresh=True)       # a refreshed short-lived key in the environment is picked up
                continue
            except BadRequestError as error:
                if self.adapt(error, params):
                    continue
                raise
            except (RateLimitError, APIConnectionError, APITimeoutError) as error:
                logger.warning("Attempt %s throttled or failed: %s", attempt + 1, error)
            except APIStatusError as error:
                if error.status_code < 500 and error.status_code != 408:
                    raise
                logger.warning("Attempt %s failed with HTTP %s", attempt + 1, error.status_code)
            else:
                with self._lock:
                    self.usage["calls"] += 1
                    self.usage["seconds"] += time.monotonic() - started
                    counts = getattr(response, "usage", None)
                    if counts is not None:
                        self.usage["prompt_tokens"] += counts.prompt_tokens or 0
                        self.usage["completion_tokens"] += counts.completion_tokens or 0
                choice = response.choices[0]
                content = merged_content(choice.message)
                if content.strip():
                    return (prefill + content if prefill and self.prefill_mode == "native" else content).strip(), choice.finish_reason
                logger.warning("Attempt %s received an empty response: %s", attempt + 1, choice.message)
            attempt += 1
            if attempt < max_tries:
                time.sleep(min(2 ** attempt + random.uniform(0, 1), 60))
        logger.error("All retry attempts have been exhausted. The call has failed.")
        return "llm api error!!!", "error"


def merged_content(message) -> str:
    """Reply text with a separately returned reasoning field restored as `<think>...</think>`."""
    fields = message.model_dump(exclude_none=True)
    content = fields.get("content") or ""
    reasoning = next((str(fields[f]).strip() for f in ("reasoning_content", "reasoning") if str(fields.get(f) or "").strip()), "")
    if reasoning and "<think>" not in content:
        return f"<think>\n{reasoning}\n</think>\n{content}"
    return content
