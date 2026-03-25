from __future__ import annotations

import logging
from typing import Any

from ..knowledge.bridge import ContextGraphBridge, _parse_facts

logger = logging.getLogger(__name__)


class StudioKnowledgeBridge:
    def __init__(
        self,
        *,
        cg_url: str,
        api_key: str,
        agent_id: str,
        memory_sink: Any,
        auto_store: bool = True,
        auto_recall: bool = True,
        recall_limit: int = 5,
    ) -> None:
        self.cg_url = cg_url
        self.api_key = api_key
        self.agent_id = agent_id
        self.auto_store = auto_store
        self.auto_recall = auto_recall
        self.recall_limit = recall_limit
        self._memory_sink = memory_sink
        self._remote = None
        if cg_url:
            try:
                self._remote = ContextGraphBridge(
                    cg_url=cg_url,
                    api_key=api_key,
                    agent_id=agent_id,
                    auto_store=auto_store,
                    auto_recall=auto_recall,
                    recall_limit=recall_limit,
                )
            except (ConnectionError, TimeoutError, OSError, ValueError) as exc:
                logger.warning("Studio knowledge bridge remote init failed: %s", exc)

    def recall(self, query: str) -> list[dict]:
        if self._remote is None or not self.auto_recall:
            return []
        return self._remote.recall(query)

    def recall_memories(
        self,
        query: str,
        *,
        limit: int | None = None,
        memory_kind: str | None = None,
        pinned_only: bool = False,
        min_importance: float | None = None,
        tags: list[str] | None = None,
        token_budget: int | None = None,
        summary_only: bool = False,
    ) -> list[dict]:
        if self._remote is None or not self.auto_recall:
            return []
        return self._remote.recall_memories(
            query,
            limit=limit,
            memory_kind=memory_kind,
            pinned_only=pinned_only,
            min_importance=min_importance,
            tags=tags,
            token_budget=token_budget,
            summary_only=summary_only,
        )

    def get_trust(self) -> dict:
        if self._remote is None:
            return {}
        return self._remote.get_trust()

    def discover(self, query: str = "", min_reputation: float = 0.0) -> list[dict]:
        if self._remote is None:
            return []
        return self._remote.discover(query=query, min_reputation=min_reputation)

    def register(
        self, name: str, org_id: str, capabilities: list[str] | None = None
    ) -> str:
        if self._remote is None:
            self._remote = ContextGraphBridge(
                cg_url=self.cg_url,
                api_key=self.api_key,
                agent_id=self.agent_id,
                auto_store=self.auto_store,
                auto_recall=self.auto_recall,
                recall_limit=self.recall_limit,
            )
        self.agent_id = self._remote.register(name, org_id, capabilities=capabilities)
        self.api_key = self._remote.api_key
        return self.agent_id

    def store(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        evidence: list[str] | None = None,
        citations: list[str] | None = None,
        **memory_fields: Any,
    ) -> dict | None:
        if not self.auto_store:
            return None
        payload = {
            "metadata": metadata or {},
            "evidence": evidence or [],
            "citations": citations or [],
        }
        payload["metadata"].update(
            {
                key: value
                for key, value in memory_fields.items()
                if value not in (None, "", [], {})
            }
        )
        return self._memory_sink(content, payload)

    def summarize_and_store(
        self,
        conversation_context: str,
        provider: Any,
        agent_name: str = "",
    ) -> list[dict]:
        if not conversation_context.strip():
            return []
        extraction_prompt = (
            "Review the conversation below (delimited by <conversation> tags) and extract "
            "0-5 key facts, decisions, or insights worth remembering for future conversations. "
            "Return ONLY a JSON array of objects with 'content' and 'metadata' keys.\n\n"
            f"<conversation>\n{conversation_context}\n</conversation>"
        )
        response = provider.complete(
            messages=[{"role": "user", "content": extraction_prompt}],
            tools=[],
            system="You extract structured knowledge from conversations. Return only valid JSON.",
        )
        facts = _parse_facts(response.content)
        stored: list[dict] = []
        for fact in facts:
            payload = {
                "metadata": {
                    "source": "session_summary",
                    "agent": agent_name,
                    "memory_kind": "summary",
                    "tags": ["session-summary"],
                    **(
                        fact.get("metadata")
                        if isinstance(fact.get("metadata"), dict)
                        else {}
                    ),
                }
            }
            result = self._memory_sink(str(fact.get("content", "")).strip(), payload)
            if result:
                stored.append(result)
        return stored
