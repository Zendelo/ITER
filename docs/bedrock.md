# Running the agent on an API-served LLM (Amazon Bedrock)

The Tongyi ReAct agent (`src/search_agent/tongyi_client.py`) can use an LLM served by
Amazon Bedrock instead of a local vLLM server. **Only the LLM changes.** The system prompt,
the `search` / `get_document` tools, the per-turn reasoning caps (4096 / 2048 / 1024 tokens),
the 50-call budget, the context budget, the forced final `<answer>` turn, the retrievers and the
trajectory JSON are the same, so a Bedrock arm is comparable with the runs in
[RMIT-ADMS/agentic-search-trajectories](https://huggingface.co/datasets/RMIT-ADMS/agentic-search-trajectories)
and feeds stages 2, 3 and 5 unchanged. The default (`--llm-backend vllm`) is the original behaviour.

## Credentials

Bedrock's OpenAI-compatible endpoint (Mantle) is configured through the environment, never the
command line. Put these in the environment or in a dotenv file passed with `--env-file`
(see `.env.example`):

```text
BEDROCK_MANTLE_OPENAI_BASE_URL   e.g. https://bedrock-mantle.us-west-2.api.aws  (with or without /v1)
BEDROCK_MANTLE_OPENAI_API_KEY    a Bedrock API key; short-lived keys expire
BEDROCK_MODEL_ID                 optional default for --model
```

The client checks them before loading an index. If the key expires mid-run the run stops
without writing failure records, so the finished questions are skipped when you rerun after
refreshing the key.

## Reproducing the released conditions

| Condition | Setting |
| --- | --- |
| Agent harness | the Tongyi ReAct loop, unchanged (`tongyi_utils/react_agent.py`) |
| Tools | `search` (top-10, 64-token snippets: `--k 10 --snippet-max-tokens 64`) and `get_document` |
| Document read cap | `--max-visit-tokens 12000` (12k-read runs; the legacy runs used 512) |
| Context budget | `--max-context-tokens 120000` (12k-read runs; legacy 90000); counted with the Tongyi tokenizer (`--tokenizer`) for every LLM |
| Tool-call budget | `MAX_LLM_CALL_PER_RUN=50` |
| Sampling | `--temperature 0.6 --top_p 0.95 --presence_penalty 1.1 --seed 2026` |
| Retrievers | BM25 (Pyserini) or the untrained `Qwen/Qwen3-Embedding-0.6B`, no result de-duplication |
| Plain vs attributed | add `--attributed` for the attributed-answer prompt (system prompt, question and forced final turn, verbatim from the dataset card) |
| Query style | `plain` (i0), the retriever sees the sub-query only |

```bash
export MAX_LLM_CALL_PER_RUN=50
python src/search_agent/tongyi_client.py \
  --llm-backend bedrock --model <bedrock-model-id> --env-file .env \
  --searcher-type bm25 --index-path <lucene-index> \
  --query <topics.tsv> --output-dir runs/bedrock_<model>_bm25_plain \
  --snippet-max-tokens 64 --k 10 \
  --max-visit-tokens 12000 --max-context-tokens 120000 \
  --temperature 0.6 --top_p 0.95 --presence_penalty 1.1 --num-threads 4
```

Try one question first: `--query "your question"` runs a single trajectory. `run_eval.py --backbone
bedrock --model <id> [--attributed --max-visit-tokens 12000 --max-context-tokens 120000]` builds
the same command. Finished questions are skipped on restart.

## What a hosted API cannot do like vLLM

`tongyi_utils/llm_backend.py` contains every difference, and each one is recorded in the run's
`metadata.llm_backend`:

| vLLM request | Bedrock backend |
| --- | --- |
| `chat_template_kwargs.enable_thinking`, `continue_final_message`, `logprobs` | not sent (vLLM-only; `logprobs` is unused) |
| `presence_penalty`, `top_p`, `seed`, `stop`, `temperature` | sent; if the endpoint rejects one, it is dropped for the rest of the run and listed in `dropped_params` |
| `max_tokens` | `max_tokens`; switches to `max_completion_tokens` if the endpoint asks |
| model reasoning inside `<think>` | a separately returned `reasoning_content` / `reasoning` field is put back inside `<think>` tags, the form `i6` / `i7` queries and the trajectory parsers read |
| assistant prefill `<answer>` for the forced final answer | an instruction by default (`--prefill-mode instruction`); `native` sends a real prefill if the endpoint supports it |

Anything the model does differently (its own reasoning style, tool-call formatting, refusals) is
the experimental variable, not a deviation. Record `metadata.llm_backend` with any result.

## Cost

Each trajectory's `metadata.llm_backend.usage` has the calls, prompt and completion tokens and the
API seconds of that question, so marginal agent cost can be priced per topic.

## Tests

`tests/` runs the real agent loop and CLI against a scripted local endpoint (no network, model or
index): `pip install pytest openai httpx python-dotenv tqdm json5 qwen-agent soundfile transformers
tiktoken numpy` then `pytest tests`.

## Not yet verified against live Bedrock

The implementation was developed against the scripted endpoint. The Bedrock key available when it
was written had expired, so a live smoke test (one question, then a few) is still to do with a fresh
key and a model id that supports a long multi-turn chat.
