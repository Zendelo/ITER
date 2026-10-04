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
NATIVE_CALL = re.compile(
    r"\s*(\w+)\s*<arg_key>\s*(\w+)\s*</arg_key>\s*<arg_value>((?:(?!</?arg_(?:key|value)>).)*)</arg_value>\s*",
    re.DOTALL,
)
NAME_FIRST = re.compile(r"\s*(\w+)\s*(?:</arg_key>)?\s*\(?\s*(\{.*)$", re.DOTALL)


def parses(span: str) -> bool:
    """True when the agent's own parse of this span yields a dict."""
    try:
        return isinstance(json5.loads(span), dict)
    except ValueError:
        return False


def as_call(name: str, arguments) -> dict | None:
    """The call if it names a known tool and exactly its one argument, else None."""
    key = TOOL_ARGUMENT.get(name)
    if (
        key
        and isinstance(arguments, dict)
        and list(arguments) == [key]
        and arguments[key]
    ):
        return {"name": name, "arguments": arguments}
    return None


def read_call(body: str) -> tuple[dict | None, str]:
    """Interpret one call body; returns the call and the repair kind ("" when the JSON was already fine)."""
    if native := NATIVE_CALL.fullmatch(body):
        # Exactly one complete argument; retain whitespace inside its value.
        return as_call(native[1], {native[2]: native[3]}), "native_tags"
    body = STRAY.sub("", body.strip())
    if parses(body):
        try:
            found = json5.loads(body, allow_duplicate_keys=False)
        except ValueError:
            return None, ""
        return as_call(found.get("name"), found.get("arguments")), ""
    if (named := NAME_FIRST.match(body)) and (name := named.group(1)) in TOOL_ARGUMENT:
        rest = STRAY.sub("", named.group(2)).removesuffix(
            ")"
        )  # search({"query": ...}) call style
        if parses(rest):
            try:
                arguments = json5.loads(rest, allow_duplicate_keys=False)
            except ValueError:
                return None, ""
            return as_call(name, arguments), "name_first"
    # An unterminated quoted value has no unambiguous boundary. Leave it
    # to the controller rather than guessing which braces/space are content.
    return None, ""


def repair_tool_call(content: str) -> tuple[str, list[str]]:
    """The reply with its first tool call rewritten canonically, and what was repaired.

    A reply whose first call the agent already parses is returned unchanged.
    """
    # ITER checks for a closer anywhere, then splits at the first opener and
    # the next opener/closer. Preserve even nested-openers it already accepts.
    if "<tool_call>" in content and "</tool_call>" in content:
        body = content.split("<tool_call>")[1].split("</tool_call>")[0]
        if parses(body):
            return content, []
    head, think_end, tail = content.rpartition("</think>")
    prose, opener, rest = tail.partition("<tool_call>")
    if not opener:
        return content, []
    end = re.search(r"</tool_call>|<tool_call>|$", rest)
    assert end is not None  # The terminal alternative always matches.
    body, closed, more = (
        rest[: end.start()],
        end.group() == "</tool_call>",
        rest[end.end() :],
    )
    if closed and parses(body):
        return content, []
    call, kind = read_call(body)
    if call is None:
        return content, []
    extra = end.group() == "<tool_call>" or "<tool_call>" in more
    repairs = [
        r
        for r in (
            kind,
            "" if closed else "unclosed",
            "extra_calls_dropped" if extra else "",
        )
        if r
    ]
    return head + think_end + prose + "<tool_call>" + json.dumps(
        call, ensure_ascii=False
    ) + "</tool_call>", repairs
