# Code is mostly based on the original Alibaba-NLP/DeepResearch inference script
# in https://github.com/Alibaba-NLP/DeepResearch
# Modified to use only our local search tool to adhere to BrowseComp-Plus evaluation

import json5
import logging
import os
import re
from typing import Dict, List, Optional, Union
from qwen_agent.llm.schema import Message
from qwen_agent.utils.utils import build_text_completion_prompt
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
import tiktoken
from transformers import AutoTokenizer 
from datetime import datetime
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
import time


SYSTEM_PROMPT_SEARCH_ONLY = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within XML tags:
{"type": "function", "function": {"name": "search", "description": "Perform local web searches then returns a string of the top search results. Accepts a single query.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query."}}, "required": ["query"], "example": {"name": "search", "arguments": {"query": ["xxxx"]}}, "uniqueItems": true}}}
{"type": "function", "function": {"name": "get_document", "description": "Retrieve the full content one documents given document ID.", "parameters": {"type": "object", "properties": {"docid": {"type": "string", "description": "The document ID(s) to retrieve."}}, "required": ["docid"], "example": {"name": "get_document", "arguments": {"docid": xxx}}, "uniqueItems": true}}}

You must obey the following strict parameter formatting rules. Violating them is not allowed.

# STRICT TOOL RULES:
0. The ONLY available tools are search and get_document. Any tool not defined here DO NOT EXIST and must not be referenced or used. Document retrieval is local. A docid alone is sufficient to retrieve content using get_document. Use search to find document IDs or general information. Use get_document to retrieve document content.

1. For the search tool, the ONLY allowed parameter structure is:
{"query": "some text"}

query must be a plain string.
No additional keys may be included.

2. For the get_document tool, the ONLY allowed parameter structure is:
{"docid": "123456"}

docid must be EXACTLY the numeric document ID extracted from search results.
Do NOT prepend text such as "DocID:", "ID=", "docid=", "document #", URLs, paths, or filenames.
Do NOT wrap the docid in other characters, such as brackets, quotes inside quotes, markup, or whitespace.
The value must be ONLY the number.
NEVER guess or fabricate docid.

3. DO NOT construct URLs or attempt to visit external pages. Never fabricate document content—always retrieve it with get_document.
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": , "arguments": }
</tool_call>

4. YOU CAN NOT SCROLL
Repeated calls with the same docid will return the same document content again, not a later section. Never use visit or scrolling behavior. If one document is insufficient, use search again or provide your best answer.
You may only call get_document after a search result explicitly supplies a numeric document ID.

If the number of llm calls exceeds the limit, if reached the maximum context length. You MUST stop making tool calls and based on all the information above, provide what you consider the most likely answer ONLY in the following format:<answer>your answer</answer>"
"""


from tongyi_utils.llm_backend import VllmBackend
from tongyi_utils.tool_search import SearchToolHandler

# The attributed-answer condition (RMIT-ADMS/agentic-search-trajectories dataset card): appended to the
# system prompt, to the user question and to the forced final-answer turn.
ATTRIBUTION_REQUIREMENT = """

# ATTRIBUTION REQUIREMENT
Your final answer must be attributed to the documents you retrieved. Inside the <answer></answer> tags:
1. State the direct answer first, on its own line.
2. Then give a short justification in which every factual claim is followed by the DocID(s) of the document(s) that support it, in square brackets, e.g. [DocID:12345] or [DocID:12345, DocID:67890].
3. Cite only DocIDs that appeared in your search results or that you retrieved with get_document in this conversation. Never invent a DocID. Prefer documents you have read in full with get_document.
4. If a claim is not supported by any retrieved document, say so explicitly instead of citing.
"""
ATTRIBUTION_REMINDER = (
    "\n\nIn your final <answer>, give the direct answer first, then support every  factual claim with the "
    "DocID(s) of the retrieved documents it comes from, in the  form [DocID:12345]. Cite only DocIDs you "
    "actually saw in this conversation."
)

OBS_START = '<tool_response>'
OBS_END = '\n</tool_response>'

MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 50))


import random
import datetime


logger = logging.getLogger(__name__)


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")


DEDUP_NOTICE = """

# Retriever behavior
The search tool de-duplicates across steps: a document already surfaced by an earlier search will NOT appear again in later search results (this keeps each search focused on new material). If you still need a document you have seen before, you can retrieve its full content at any time with get_document using its DocID. When a search hides earlier-seen relevant documents, it lists them under an "Already-seen" section so you can revisit them."""


class MultiTurnReactAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 search_tool_handler: Optional[SearchToolHandler] = None,
                 get_document_handler: Optional[SearchToolHandler] = None,
                 **kwargs):

        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]
        self.search_tool = search_tool_handler
        self.get_document_tool = get_document_handler
        self.max_generation = 4096
        self.enable_thinking = True
        # Where the model is served is the only thing that may differ between arms; everything below
        # (prompts, caps, budgets) is the same for every backend. Defaults reproduce the original agent.
        self.backend = llm.get("backend") or VllmBackend()
        self.max_context_tokens = llm.get("max_context_tokens", 90000)
        self.attributed = bool(llm.get("attributed", False))
        # Context is always counted with the Tongyi tokenizer, so the budget means the same for any LLM.
        self.tokenizer = AutoTokenizer.from_pretrained(llm.get("tokenizer") or self.llm_local_path)

    
    def call_server(self, msgs, planning_port, max_tries=10, prefill=None):
        """One LLM turn through the configured backend: `(content, finish_reason)`.

        `prefill`, if set, forces the reply to start with that string (the forced final `<answer>`).
        """
        return self.backend.generate(
            msgs, model=self.model, max_tokens=self.max_generation, generate_cfg=self.llm_generate_cfg,
            enable_thinking=self.enable_thinking, prefill=prefill, port=planning_port, max_tries=max_tries,
        )

    def count_tokens(self, messages, model="gpt-4o"):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True
        )
        t_count = len(input_ids)
        return t_count


    def _run(self, data: str, model: str, **kwargs) -> List[List[Message]]:
        self.model=model
        try:
            question = data['item']['question']
        except:
            raw_msg = data['item']['messages'][1]["content"]
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg

        tool_call_counts = {}  # Only successful tool calls
        tool_call_counts_all = {}  # All tool calls (successful and failed)
        retrieved_docids = []
        tool_traces = []

        start_time = time.time()
        planning_port = data['planning_port']
        answer = data['item']['answer']
        self.user_prompt = question
        if self.search_tool:
            self.search_tool.reset_trajectory(question)
        system_prompt = SYSTEM_PROMPT_SEARCH_ONLY
        if self.search_tool and getattr(self.search_tool, "dedup_search", False):
            system_prompt += DEDUP_NOTICE
        user_prompt = question
        if self.attributed:
            system_prompt += ATTRIBUTION_REQUIREMENT
            user_prompt += ATTRIBUTION_REMINDER
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0
        pending_visit = False  # a doc was just read; its post-visit reasoning is the next think
        while num_llm_calls_available > 0:
            round += 1

            if round == 1:
                self.max_generation = 4096
            elif round == 2:
                self.max_generation = 2048
            else:
                self.max_generation = 1024

            num_llm_calls_available -= 1
            content, finish_reason = self.call_server(messages, planning_port)

            self.enable_thinking = True

            if finish_reason == "length" and num_llm_calls_available > 0:
                messages.append({"role": "assistant", "content": content})

                truncated_msg = (
                    "ERROR: Your previous thought was too long and has been discarded. "
                    "Now, skip all reasoning and directly provide the <tool_call> or <answer>."
                )
                messages.append({"role": "user", "content": f"<tool_response>\n{truncated_msg}\n</tool_response>"})
                self.enable_thinking = False
                continue

            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]

            messages.append({"role": "assistant", "content": content.strip()})

            # capture post-visit reasoning for [Memory]: this turn's think reflects on
            # the doc read last turn (mirrors data_builder's _reasoning_after).
            if self.search_tool and pending_visit:
                thinks = re.findall(r'<think>(.*?)</think>', content, re.DOTALL)
                self.search_tool.add_visit_reasoning(" ".join(t.strip() for t in thinks))
                pending_visit = False

            if '<tool_call>' in content and '</tool_call>' in content:
                tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]
                try:
                    tool_call = json5.loads(tool_call)
                    tool_name = tool_call.get('name', '')
                    tool_args = tool_call.get('arguments', {})

                    tool_call_counts_all[tool_name] = tool_call_counts_all.get(tool_name, 0) + 1

                    # i6/i7: hand the current turn's <think> to the search tool
                    if tool_name == 'search' and self.search_tool:
                        thinks = re.findall(r'<think>(.*?)</think>', content, re.DOTALL)
                        self.search_tool.set_current_thinking(" ".join(t.strip() for t in thinks))

                    result, docids = self.custom_call_tool(tool_name, tool_args)
                    if tool_name in ("get_document", "visit"):
                        pending_visit = True


                    if docids is not None:
                        tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
                        retrieved_docids.extend(docids)
                except:
                    tool_call_counts_all['invalid_json'] = tool_call_counts_all.get('invalid_json', 0) + 1
                    result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                result = "<tool_response>\n" + result + "\n</tool_response>"
                messages.append({"role": "user", "content": result})
            elif '<answer>' in content and '</answer>' in content:
                termination = 'answer'
                break

            max_tokens = self.max_context_tokens
            token_count = self.count_tokens(messages)

            if num_llm_calls_available <= 0 or token_count > max_tokens:
                messages[-1]['content'] = 'Retrieval complete. You are forbidden to call any tools now. Based only on the information already collected above, provide your best final answer.'
                if self.attributed:
                    messages[-1]['content'] += ATTRIBUTION_REMINDER
                self.max_generation = 10000
                content, finish_reason = self.call_server(messages, planning_port, prefill="<answer>")
                messages.append({"role": "assistant", "content": content.strip()})

                if finish_reason == "length":
                    self.enable_thinking = False
                    content, finish_reason = self.call_server(messages, planning_port, prefill="<answer>")
                
                if '<answer>' in content and '</answer>' in content:
                    prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
                    termination = 'generate an answer as token limit reached'
                else:
                    prediction = messages[-1]['content']
                    termination = 'format error: generate an answer as token limit reached'

                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                    "tool_call_counts": tool_call_counts,
                    "tool_call_counts_all": tool_call_counts_all,
                    "retrieved_docids": list(set(retrieved_docids)),
                    "tool_traces": self.search_tool.get_search_traces() if self.search_tool else tool_traces,
                }
                return result

        if '<answer>' in messages[-1]['content']:
            prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'
        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "tool_call_counts": tool_call_counts,
            "tool_call_counts_all": tool_call_counts_all,
            "retrieved_docids": list(set(retrieved_docids)),
            "tool_traces": self.search_tool.get_search_traces() if self.search_tool else tool_traces,
        }
        return result

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs): 
        if tool_name == "search" and self.search_tool:
            return self.search_tool.call(tool_args, **kwargs)
        elif tool_name == "get_document" and self.get_document_tool:
            result, docids = self.get_document_tool.call(tool_args, **kwargs)
            if self.search_tool and docids:
                self.search_tool.add_found_docids(docids)
                for d in docids:
                    self.search_tool.add_visited_docid(d)
            return result, docids
        elif tool_name == "visit" and self.get_document_tool:
            result, docids = self.get_document_tool.call(tool_args, **kwargs)
            if self.search_tool and docids:
                self.search_tool.add_found_docids(docids)
                for d in docids:
                    self.search_tool.add_visited_docid(d)
            return result, docids
        else:
            return f"Error: Tool {tool_name} not found", None
