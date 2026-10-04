"""Repairs re-wrap a model's near-miss tool call; they never change argument values."""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path[:0] = [str(SRC), str(SRC / "search_agent")]

from tongyi_utils.tool_call_repair import repair_tool_call

CANON = '<tool_call>{"name": "search", "arguments": {"query": "stress"}}</tool_call>'


@pytest.mark.parametrize(
    "reply, repairs",
    [
        (
            '<tool_call>search\n{"query": "stress"}</arg_value></tool_call>',
            ["name_first"],
        ),
        ('<tool_call>search</arg_key>{"query": "stress"}</tool_call>', ["name_first"]),
        (
            "<tool_call>search<arg_key>query</arg_key><arg_value>stress</arg_value></tool_call>",
            ["native_tags"],
        ),
        (
            '<tool_call>search({"query": "stress"})</arg_value>',
            ["name_first", "unclosed"],
        ),
        (
            '<tool_call>{"name": "search", "arguments": {"query": "stress"}}',
            ["unclosed"],
        ),
        (
            '<tool_call>{"name": "search", "arguments": {"query": "stress"}}<tool_call>{"name": "search", "arguments": {"query": "x"}}',
            ["unclosed", "extra_calls_dropped"],
        ),
    ],
)
def test_repairs_to_the_canonical_call(reply, repairs):
    assert repair_tool_call(reply) == (CANON, repairs)


@pytest.mark.parametrize(
    "reply",
    [
        CANON,
        "<think>a <tool_call> b</think>no call here",
        '<tool_call>search\n{"q": "stress"}</tool_call>',  # wrong argument name
        '<tool_call>browse\n{"query": "stress"}</tool_call>',  # unknown tool
        "<answer>done</answer>",
    ],
)
def test_leaves_valid_and_unsafe_replies_alone(reply):
    assert repair_tool_call(reply) == (reply, [])


def test_keeps_reasoning_and_prose():
    out, _ = repair_tool_call(
        '<think><tool_call>x</think>Looking.\n<tool_call>search\n{"query": "stress"}</tool_call>'
    )
    assert out == "<think><tool_call>x</think>Looking.\n" + CANON


@pytest.mark.parametrize(
    "reply",
    [
        "<tool_call><arg_key>query</arg_key><arg_value>stress</arg_value></tool_call>",
        "<tool_call>search<arg_key>query</arg_key><arg_value>first</arg_value><arg_key>query</arg_key><arg_value>second</arg_value></tool_call>",
        '<tool_call>search {"query": "find literal brace } </tool_call>',
        '<tool_call>search {"query": "first", "query": "second"}</tool_call>',
        '<tool_call>{"name": "search", "arguments": {"query": "stress}}</arg_value></tool_call>',
        '<tool_call>search {"query": "stress}</arg_value></tool_call>',
        CANON.removesuffix("</tool_call>")
        + '<tool_call>{"name": "search", "arguments": {"query": "x"}}</tool_call>',
    ],
)
def test_preserves_ambiguous_values_and_controller_accepted_calls(reply):
    assert repair_tool_call(reply) == (reply, [])


def test_preserves_native_argument_whitespace():
    reply = "<tool_call>search<arg_key>query</arg_key><arg_value>  literal }  </arg_value></tool_call>"
    out, kinds = repair_tool_call(reply)
    assert '"query": "  literal }  "' in out
    assert kinds == ["native_tags"]
