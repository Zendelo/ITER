"""Rewrite a model's near-miss tool call into the form the Tongyi ReAct loop parses.

The loop reads `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` and counts anything
else as invalid JSON. GLM was trained on its own tag format and drifts to it (name first,
stray `</arg_value>`, unclosed tag, several calls in one reply). Repair never invents or edits
argument values: it re-wraps what the model wrote, and returns the reply unchanged when it
cannot do so safely. Only text after the last `</think>` is touched.
"""

import json
import re

import json5

TOOL_ARGUMENT = {"search": "query", "get_document": "docid"}
STRAY = re.compile(r"\s*(</arg_value>|</arg_key>)\s*$")
NATIVE_PAIR = re.compile(r"<arg_key>\s*(\w+)\s*</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)
NAME_FIRST = re.compile(r"\s*(\w+)\s*(?:</arg_key>)?\s*\(?\s*(\{.*)$", re.S)
UNTERMINATED_VALUE = re.compile(r'"(\w+)"\s*:\s*"(.*?)[\s}]*$', re.S)


def parses(span: str) -> bool:
    """True when the agent's own parse of this span yields a dict."""
    try:
        return isinstance(json5.loads(span), dict)
    except Exception:
        return False


def as_call(name: str, arguments) -> dict | None:
    """The call if it names a known tool and exactly its one argument, else None."""
    key = TOOL_ARGUMENT.get(name)
    if key and isinstance(arguments, dict) and list(arguments) == [key] and arguments[key]:
        return {"name": name, "arguments": arguments}
    return None


def read_call(body: str) -> tuple[dict | None, str]:
    """Interpret one call body; returns the call and the repair kind ("" when the JSON was already fine)."""
    if pairs := NATIVE_PAIR.findall(body):      # before stray-tag stripping, which would eat the last closer
        name = re.match(r"\s*(\w+)", body).group(1)
        return as_call(name, {k: v.strip() for k, v in pairs}), "native_tags"
    body = STRAY.sub("", body.strip())
    if parses(body):
        found = json5.loads(body)
        return as_call(found.get("name"), found.get("arguments")), ""
    if (named := NAME_FIRST.match(body)) and (name := named.group(1)) in TOOL_ARGUMENT:
        rest = STRAY.sub("", named.group(2)).removesuffix(")")      # search({"query": ...}) call style
        if parses(rest):
            return as_call(name, json5.loads(rest)), "name_first"
    name = re.search(r'"name"\s*:\s*"(\w+)"', body) or re.match(r"\s*(\w+)\s*(?:</arg_key>)?\s*\{", body)
    value = UNTERMINATED_VALUE.search(body.split('"arguments"')[-1])
    if name and value:
        return as_call(name.group(1), {value.group(1): value.group(2)}), "unterminated_value"
    return None, ""


def repair_tool_call(content: str) -> tuple[str, list[str]]:
    """The reply with its first tool call rewritten canonically, and what was repaired.

    A reply whose first call the agent already parses is returned unchanged.
    """
    head, think_end, tail = content.rpartition("</think>")
    prose, opener, rest = tail.partition("<tool_call>")
    if not opener:
        return content, []
    end = re.search(r"</tool_call>|<tool_call>|$", rest)
    body, closed, more = rest[: end.start()], end.group() == "</tool_call>", rest[end.end():]
    if closed and parses(body):
        return content, []
    call, kind = read_call(body)
    if call is None:
        return content, []
    extra = end.group() == "<tool_call>" or "<tool_call>" in more
    repairs = [r for r in (kind, "" if closed else "unclosed", "extra_calls_dropped" if extra else "") if r]
    return head + think_end + prose + "<tool_call>" + json.dumps(call, ensure_ascii=False) + "</tool_call>", repairs
