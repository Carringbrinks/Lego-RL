import json

from verl_patch.agent_loop.vllm_chat_completion_proxy import (
    _anthropic_to_openai_messages,
    _normalize_openai_messages_for_template,
    _openai_message_to_anthropic,
)


def test_anthropic_request_converts_messages_tools_and_results():
    messages, tools = _anthropic_to_openai_messages(
        {
            "system": [{"type": "text", "text": "system"}],
            "messages": [
                {"role": "user", "content": "inspect"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "running"},
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "bash",
                            "input": {"cmd": "pwd"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": "ok",
                        },
                        {"type": "text", "text": "continue"},
                    ],
                },
            ],
            "tools": [
                {
                    "name": "bash",
                    "description": "run command",
                    "input_schema": {"type": "object"},
                }
            ],
        }
    )

    assert messages[0] == {"role": "system", "content": "system"}
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == {"cmd": "pwd"}
    assert messages[3] == {"role": "tool", "tool_call_id": "tool-1", "content": "ok"}
    assert messages[4] == {"role": "user", "content": "continue"}
    assert tools[0]["function"]["parameters"] == {"type": "object"}


def test_claude_code_trailing_system_is_merged_to_one_leading_message():
    messages, _ = _anthropic_to_openai_messages(
        {
            "system": [{"type": "text", "text": "top-level system"}],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "<system-reminder>context</system-reminder>",
                        }
                    ],
                },
                {
                    "role": "system",
                    "content": "Available agent types for the Agent tool",
                },
            ],
        }
    )

    normalized = _normalize_openai_messages_for_template(messages)

    assert [message["role"] for message in normalized] == ["system", "user"]
    assert normalized[0]["content"] == (
        "top-level system\n\nAvailable agent types for the Agent tool"
    )
    assert normalized[1]["content"] == (
        "<system-reminder>context</system-reminder>"
    )


def test_openai_tool_call_converts_to_anthropic_response():
    body = _openai_message_to_anthropic(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"cmd": "pwd"}),
                    },
                }
            ],
        },
        finish_reason="tool_calls",
        model="vllm_model",
        prompt_tokens=10,
        completion_tokens=4,
        session_id="session",
    )

    assert body["stop_reason"] == "tool_use"
    assert body["content"] == [
        {"type": "tool_use", "id": "call-1", "name": "bash", "input": {"cmd": "pwd"}}
    ]
    assert body["usage"] == {"input_tokens": 10, "output_tokens": 4}
