"""Tests for ContextGraphBridge with a mocked contextgraph_sdk."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _make_fake_sdk():
    """Build a minimal fake contextgraph_sdk module tree."""
    sdk = types.ModuleType("contextgraph_sdk")

    class FakeHttpTransport:
        def __init__(self, base_url: str, api_key: str = ""):
            self.base_url = base_url
            self.api_key = api_key

    class FakeContextGraph:
        def __init__(self, transport):
            self.transport = transport

        def recall(self, agent_id, query, limit=5):
            return []

        def store(self, agent_id, content, metadata=None, evidence=None, citations=None):
            return {"id": "mem-1", "content": content}

        def agent_trust(self, agent_id, requester_id):
            return {"score": 1.0}

        def register_agent(self, name, org_id, capabilities=None):
            return {"agent_id": "agent-new"}

        def discover(self, agent_id, q="", min_reputation=0.0):
            return {"agents": []}

    sdk.ContextGraph = FakeContextGraph
    sdk.HttpTransport = FakeHttpTransport
    return sdk


@pytest.fixture(autouse=True)
def patch_sdk():
    """Inject fake contextgraph_sdk before every test in this module."""
    fake_sdk = _make_fake_sdk()
    # Remove any real module and replace with fake
    original = sys.modules.get("contextgraph_sdk")
    sys.modules["contextgraph_sdk"] = fake_sdk
    # Also remove cached bridge module so it re-imports from fake sdk
    bridge_mod = sys.modules.pop("contextclaw.knowledge.bridge", None)
    yield fake_sdk
    # Restore
    if original is not None:
        sys.modules["contextgraph_sdk"] = original
    else:
        sys.modules.pop("contextgraph_sdk", None)
    if bridge_mod is not None:
        sys.modules["contextclaw.knowledge.bridge"] = bridge_mod
    else:
        sys.modules.pop("contextclaw.knowledge.bridge", None)


def _make_bridge(**kwargs):
    from contextclaw.knowledge.bridge import ContextGraphBridge
    defaults = dict(cg_url="http://localhost:8765")
    defaults.update(kwargs)
    return ContextGraphBridge(**defaults)


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------


def test_recall_returns_empty_when_agent_id_not_set():
    bridge = _make_bridge(agent_id="")
    result = bridge.recall("some query")
    assert result == []


def test_recall_returns_empty_when_auto_recall_false():
    bridge = _make_bridge(agent_id="agent-1", auto_recall=False)
    result = bridge.recall("some query")
    assert result == []


def test_recall_calls_client_when_configured(patch_sdk):
    client_mock = MagicMock()
    client_mock.recall.return_value = [{"content": "memory item"}]
    bridge = _make_bridge(agent_id="agent-1", auto_recall=True)
    bridge._client = client_mock
    result = bridge.recall("test query")
    client_mock.recall.assert_called_once_with("agent-1", "test query", limit=5)
    assert result == [{"content": "memory item"}]


def test_recall_handles_exception_gracefully(patch_sdk):
    bridge = _make_bridge(agent_id="agent-1", auto_recall=True)
    bridge._client = MagicMock()
    bridge._client.recall.side_effect = RuntimeError("connection refused")
    result = bridge.recall("query")
    assert result == []


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


def test_store_returns_none_when_auto_store_false():
    bridge = _make_bridge(agent_id="agent-1", auto_store=False)
    result = bridge.store("some output")
    assert result is None


def test_store_returns_none_when_agent_id_not_set():
    bridge = _make_bridge(agent_id="", auto_store=True)
    result = bridge.store("some output")
    assert result is None


def test_store_calls_client_when_configured(patch_sdk):
    client_mock = MagicMock()
    client_mock.store.return_value = {"id": "mem-42"}
    bridge = _make_bridge(agent_id="agent-1", auto_store=True)
    bridge._client = client_mock
    result = bridge.store("important output", metadata={"key": "val"})
    client_mock.store.assert_called_once_with(
        "agent-1", "important output", metadata={"key": "val"}, evidence=None, citations=None
    )
    assert result == {"id": "mem-42"}


def test_store_handles_exception_gracefully(patch_sdk):
    bridge = _make_bridge(agent_id="agent-1", auto_store=True)
    bridge._client = MagicMock()
    bridge._client.store.side_effect = ConnectionError("server down")
    result = bridge.store("content")
    assert result is None


# ---------------------------------------------------------------------------
# get_trust
# ---------------------------------------------------------------------------


def test_get_trust_returns_empty_when_no_agent_id():
    bridge = _make_bridge(agent_id="")
    assert bridge.get_trust() == {}


def test_get_trust_returns_dict_when_agent_set(patch_sdk):
    client_mock = MagicMock()
    client_mock.agent_trust.return_value = {"score": 0.9, "level": "trusted"}
    bridge = _make_bridge(agent_id="agent-1")
    bridge._client = client_mock
    trust = bridge.get_trust()
    assert trust == {"score": 0.9, "level": "trusted"}


def test_get_trust_handles_exception_gracefully(patch_sdk):
    bridge = _make_bridge(agent_id="agent-1")
    bridge._client = MagicMock()
    bridge._client.agent_trust.side_effect = Exception("timeout")
    assert bridge.get_trust() == {}


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


def test_register_sets_agent_id(patch_sdk):
    client_mock = MagicMock()
    client_mock.register_agent.return_value = {"agent_id": "new-agent-99"}
    bridge = _make_bridge(agent_id="")
    bridge._client = client_mock
    agent_id = bridge.register("MyAgent", "org-1", capabilities=["read", "write"])
    assert agent_id == "new-agent-99"
    assert bridge.agent_id == "new-agent-99"


# ---------------------------------------------------------------------------
# discover
# ---------------------------------------------------------------------------


def test_discover_returns_empty_when_no_agent_id():
    bridge = _make_bridge(agent_id="")
    assert bridge.discover() == []


def test_discover_returns_agent_list(patch_sdk):
    client_mock = MagicMock()
    client_mock.discover.return_value = {"agents": [{"id": "other-agent"}]}
    bridge = _make_bridge(agent_id="agent-1")
    bridge._client = client_mock
    agents = bridge.discover(query="research", min_reputation=0.5)
    assert agents == [{"id": "other-agent"}]


def test_discover_handles_exception_gracefully(patch_sdk):
    bridge = _make_bridge(agent_id="agent-1")
    bridge._client = MagicMock()
    bridge._client.discover.side_effect = Exception("network error")
    assert bridge.discover() == []


# ---------------------------------------------------------------------------
# summarize_and_store
# ---------------------------------------------------------------------------


class FakeSummaryProvider:
    """Minimal LLM provider stub for summarization tests."""

    def __init__(self, response_content: str):
        self._content = response_content

    def complete(self, messages, tools, system=""):
        from contextclaw.providers.protocol import LLMResponse
        return LLMResponse(content=self._content)


def test_summarize_and_store_extracts_facts(patch_sdk):
    bridge = _make_bridge(agent_id="agent-1", auto_store=True)
    client_mock = MagicMock()
    client_mock.store.return_value = {"id": "mem-new"}
    bridge._client = client_mock

    llm_response = '[{"content": "User prefers Python", "metadata": {"type": "preference"}}]'
    provider = FakeSummaryProvider(llm_response)

    stored = bridge.summarize_and_store(
        conversation_context="User: I like Python\nAssistant: Got it!",
        provider=provider,
        agent_name="test-agent",
    )

    assert len(stored) == 1
    assert stored[0] == {"id": "mem-new"}
    client_mock.store.assert_called_once()
    call_kwargs = client_mock.store.call_args
    assert call_kwargs[1]["metadata"]["source"] == "session_summary"
    assert call_kwargs[1]["metadata"]["type"] == "preference"


def test_summarize_and_store_handles_empty_response(patch_sdk):
    bridge = _make_bridge(agent_id="agent-1", auto_store=True)
    provider = FakeSummaryProvider("[]")

    stored = bridge.summarize_and_store(
        conversation_context="User: Hi\nAssistant: Hello!",
        provider=provider,
    )
    assert stored == []


def test_summarize_and_store_without_agent_id(patch_sdk):
    bridge = _make_bridge(agent_id="", auto_store=True)
    provider = FakeSummaryProvider("[]")

    stored = bridge.summarize_and_store(
        conversation_context="User: Hi\nAssistant: Hello!",
        provider=provider,
    )
    assert stored == []


def test_summarize_and_store_handles_markdown_code_block(patch_sdk):
    bridge = _make_bridge(agent_id="agent-1", auto_store=True)
    client_mock = MagicMock()
    client_mock.store.return_value = {"id": "mem-md"}
    bridge._client = client_mock

    llm_response = '```json\n[{"content": "A fact", "metadata": {"type": "fact"}}]\n```'
    provider = FakeSummaryProvider(llm_response)

    stored = bridge.summarize_and_store(
        conversation_context="User: Tell me\nAssistant: Here",
        provider=provider,
    )
    assert len(stored) == 1


def test_summarize_and_store_handles_llm_exception(patch_sdk):
    bridge = _make_bridge(agent_id="agent-1", auto_store=True)

    class FailingProvider:
        def complete(self, messages, tools, system=""):
            raise RuntimeError("LLM unavailable")

    stored = bridge.summarize_and_store(
        conversation_context="User: Hi\nAssistant: Hello!",
        provider=FailingProvider(),
    )
    assert stored == []


# ---------------------------------------------------------------------------
# _parse_facts
# ---------------------------------------------------------------------------


def test_parse_facts_valid_json(patch_sdk):
    from contextclaw.knowledge.bridge import _parse_facts

    result = _parse_facts('[{"content": "fact1", "metadata": {"type": "fact"}}]')
    assert len(result) == 1
    assert result[0]["content"] == "fact1"


def test_parse_facts_empty_string(patch_sdk):
    from contextclaw.knowledge.bridge import _parse_facts

    assert _parse_facts("") == []
    assert _parse_facts("   ") == []


def test_parse_facts_invalid_json(patch_sdk):
    from contextclaw.knowledge.bridge import _parse_facts

    assert _parse_facts("not json at all") == []


def test_parse_facts_filters_invalid_items(patch_sdk):
    from contextclaw.knowledge.bridge import _parse_facts

    result = _parse_facts('[{"content": "valid"}, {"no_content": true}, "string"]')
    assert len(result) == 1
    assert result[0]["content"] == "valid"
