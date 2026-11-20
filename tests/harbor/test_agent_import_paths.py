import pytest

pytest.importorskip("harbor", reason="Agent import tests require the Harbor dependency")
from pathlib import Path

from harbor_patch.agents.installed_openhands_sdk import OpenHandsSDK


def test_installed_openhands_sdk_package_exports_agent_class():
    from harbor_patch.agents.installed_openhands_sdk import OpenHandsSDK
    from harbor_patch.agents.installed_openhands_sdk.openhands_sdk import (
        OpenHandsSDK as OpenHandsSDKModuleClass,
    )

    assert OpenHandsSDK is OpenHandsSDKModuleClass


def test_ohsdk_non_swe_instruction_is_not_wrapped_in_testbed_template():
    agent = object.__new__(OpenHandsSDK)
    raw = (
        "Write a SQL query answering the question.\\n"
        "Database: /app/db.sqlite\\n"
        "Write only the final SQL query to /app/answer.sql."
    )
    template = (
        Path("src/harbor_patch/agents/installed_openhands_sdk/user_template.j2")
    )

    assert agent.render_prompt_template(template, raw) == raw


def test_ohsdk_swe_instruction_still_uses_repository_template():
    agent = object.__new__(OpenHandsSDK)
    raw = (
        "# Task\\nFix the parser.\\n\\n---\\n\\n"
        "**Repo:** `example/project`\\n"
        "**Base commit:** `abcdef123456`"
    )
    template = (
        Path("src/harbor_patch/agents/installed_openhands_sdk/user_template.j2")
    )

    rendered = agent.render_prompt_template(template, raw)
    assert "/testbed/project" in rendered
    assert "Fix the parser." in rendered
