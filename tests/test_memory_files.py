from __future__ import annotations

from pathlib import Path

from contextclaw.memory_files import (
    list_memory_file_revisions,
    list_tracked_memory_files,
    restore_memory_file_revision,
    write_memory_file,
)


def test_list_tracked_memory_files_materializes_baseline_revision(tmp_path: Path):
    workspace = tmp_path / "agent"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "MEMORY.md").write_text(
        "# Agent Memory\n\nKeep release plans explicit.\n",
        encoding="utf-8",
    )

    files = list_tracked_memory_files(workspace)

    assert len(files) == 1
    assert files[0]["filename"] == "MEMORY.md"
    assert files[0]["revision_count"] == 1
    assert files[0]["current_revision_id"].startswith("rev_")


def test_restore_memory_file_revision_creates_restore_lineage(tmp_path: Path):
    workspace = tmp_path / "agent"
    workspace.mkdir(parents=True, exist_ok=True)

    first = write_memory_file(
        workspace,
        scope="agent",
        content="# Agent Memory\n\nPrefer review gates.\n",
    )
    second = write_memory_file(
        workspace,
        scope="agent",
        content="# Agent Memory\n\nPrefer progressive delivery.\n",
    )
    assert first["current_revision_id"] != second["current_revision_id"]

    restored = restore_memory_file_revision(
        workspace,
        scope="agent",
        revision_id=first["current_revision_id"],
    )
    revisions = list_memory_file_revisions(workspace, scope="agent")

    assert restored["restored_from_revision_id"] == first["current_revision_id"]
    assert "Prefer review gates." in restored["content"]
    assert revisions["revisions"][0]["action"] == "restore"
    assert (
        revisions["revisions"][0]["source_revision_id"] == first["current_revision_id"]
    )
