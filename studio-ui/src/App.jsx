import { useEffect, useRef, useState } from "react";

const DEFAULT_MEMORY_EDITOR = {
  scope: "agent",
  filename: "",
  revisionId: "",
  content: "",
};
const API_BASE = (import.meta.env.VITE_STUDIO_API_BASE || "").replace(/\/$/, "");

function studioUrl(path) {
  if (!API_BASE) {
    return path;
  }
  return new URL(path, `${API_BASE}/`).toString();
}

function Card({ title, children, actions }) {
  return (
    <section className="card">
      <div className="card-header">
        <h2>{title}</h2>
        {actions ? <div className="card-actions">{actions}</div> : null}
      </div>
      {children}
    </section>
  );
}

function JsonBlock({ value }) {
  return <pre className="json-block">{JSON.stringify(value, null, 2)}</pre>;
}

async function fetchJson(path, options) {
  const response = await fetch(studioUrl(path), options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || JSON.stringify(payload));
  }
  return payload;
}

function previewText(value, limit = 120) {
  const compact = String(value || "").replace(/\s+/g, " ").trim();
  if (!compact) {
    return "No summary";
  }
  if (compact.length <= limit) {
    return compact;
  }
  return `${compact.slice(0, limit - 3)}...`;
}

function describeLiveEvent(event) {
  const payload = event?.payload || {};
  if (typeof payload.content === "string" && payload.content.trim()) {
    return previewText(payload.content, 140);
  }
  if (typeof payload.result === "string" && payload.result.trim()) {
    return previewText(payload.result, 140);
  }
  if (typeof payload.prompt === "string" && payload.prompt.trim()) {
    return previewText(payload.prompt, 140);
  }
  if (typeof payload.error === "string" && payload.error.trim()) {
    return previewText(payload.error, 140);
  }
  return previewText(payload, 140);
}

export default function App() {
  const [project, setProject] = useState(null);
  const [projectRoot, setProjectRoot] = useState("");
  const [projectEntryAgent, setProjectEntryAgent] = useState("orchestrator");
  const [projectProvider, setProjectProvider] = useState("claude");
  const [agents, setAgents] = useState([]);
  const [runs, setRuns] = useState([]);
  const [approvals, setApprovals] = useState([]);
  const [memoryQueue, setMemoryQueue] = useState([]);
  const [memoryFiles, setMemoryFiles] = useState([]);
  const [docsQueue, setDocsQueue] = useState([]);
  const [workflow, setWorkflow] = useState({});
  const [contextGraph, setContextGraph] = useState({});
  const [contextView, setContextView] = useState({});
  const [runResult, setRunResult] = useState({});
  const [catalogResult, setCatalogResult] = useState({});
  const [memoryHistory, setMemoryHistory] = useState([]);
  const [liveEvents, setLiveEvents] = useState([]);
  const [streamStatus, setStreamStatus] = useState("connecting");
  const [errorMessage, setErrorMessage] = useState("");
  const refreshTimerRef = useRef(null);

  const [prompt, setPrompt] = useState("");
  const [targetAgent, setTargetAgent] = useState("");
  const [contextAgent, setContextAgent] = useState("");
  const [catalogAgent, setCatalogAgent] = useState("");
  const [connectorId, setConnectorId] = useState("");
  const [skillId, setSkillId] = useState("");
  const [newAgentName, setNewAgentName] = useState("");
  const [newAgentProvider, setNewAgentProvider] = useState("claude");
  const [newAgentTemplate, setNewAgentTemplate] = useState("default");
  const [memoryFileAgent, setMemoryFileAgent] = useState("");
  const [memoryEditor, setMemoryEditor] = useState(DEFAULT_MEMORY_EDITOR);

  const selectedContextAgent =
    contextAgent.trim() || targetAgent.trim() || "";
  const selectedMemoryAgent =
    memoryFileAgent.trim() || selectedContextAgent || "";
  const streamStatusLabel = {
    connecting: "Connecting live feed",
    live: "Live feed connected",
    reconnecting: "Live feed reconnecting",
    unavailable: "Live feed unavailable",
  }[streamStatus];

  function isNoProjectError(error) {
    return /No project is currently open/i.test(error?.message || "");
  }

  async function refresh({ preserveContext = false } = {}) {
    try {
      const nextProject = await fetchJson("/projects/current");
      const [
        nextAgents,
        nextRuns,
        nextApprovals,
        nextMemoryQueue,
        nextDocsQueue,
        nextWorkflow,
        nextContextGraph,
      ] = await Promise.all([
        fetchJson("/agents"),
        fetchJson("/runs"),
        fetchJson("/approvals"),
        fetchJson("/memory"),
        fetchJson("/docs/proposals"),
        fetchJson("/workflow"),
        fetchJson("/contextgraph/status"),
      ]);

      const fileParams = new URLSearchParams();
      if (selectedMemoryAgent) {
        fileParams.set("agent_name", selectedMemoryAgent);
      }
      const nextMemoryFiles = await fetchJson(
        `/memory-files${fileParams.toString() ? `?${fileParams.toString()}` : ""}`,
      );

      setProject(nextProject);
      setAgents(nextAgents);
      setRuns(nextRuns);
      setApprovals(nextApprovals);
      setMemoryQueue(nextMemoryQueue);
      setDocsQueue(nextDocsQueue);
      setWorkflow(nextWorkflow);
      setContextGraph(nextContextGraph);
      setMemoryFiles(nextMemoryFiles);

      if (!preserveContext) {
        await refreshContext({ throwOnError: false });
      }
      setErrorMessage("");
    } catch (error) {
      if (isNoProjectError(error)) {
        setProject(null);
        setAgents([]);
        setRuns([]);
        setApprovals([]);
        setMemoryQueue([]);
        setDocsQueue([]);
        setWorkflow({});
        setContextGraph({});
        setContextView({});
        setMemoryFiles([]);
        setMemoryHistory([]);
        setErrorMessage("");
        return;
      }
      setErrorMessage(error.message);
    }
  }

  function scheduleRefresh({ preserveContext = true } = {}) {
    if (refreshTimerRef.current !== null) {
      return;
    }
    refreshTimerRef.current = window.setTimeout(() => {
      refreshTimerRef.current = null;
      void refresh({ preserveContext });
    }, 180);
  }

  useEffect(() => {
    void refresh();
  }, [selectedMemoryAgent, selectedContextAgent]);

  useEffect(() => {
    if (project?.root) {
      setProjectRoot(project.root);
    }
  }, [project]);

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.EventSource !== "function") {
      setStreamStatus("unavailable");
      return undefined;
    }

    const source = new window.EventSource(studioUrl("/events/stream"));
    const handleEvent = (message) => {
      try {
        const payload = JSON.parse(message.data);
        setLiveEvents((current) => [payload, ...current].slice(0, 14));
        scheduleRefresh({ preserveContext: true });
      } catch (error) {
        setErrorMessage(error.message);
      }
    };
    const handleHeartbeat = () => {
      setStreamStatus("live");
    };

    source.addEventListener("studio.event", handleEvent);
    source.addEventListener("studio.heartbeat", handleHeartbeat);
    source.onopen = () => {
      setStreamStatus("live");
    };
    source.onerror = () => {
      setStreamStatus("reconnecting");
    };

    return () => {
      if (refreshTimerRef.current !== null) {
        window.clearTimeout(refreshTimerRef.current);
        refreshTimerRef.current = null;
      }
      source.removeEventListener("studio.event", handleEvent);
      source.removeEventListener("studio.heartbeat", handleHeartbeat);
      source.close();
    };
  }, []);

  async function refreshContext({ throwOnError = true } = {}) {
    try {
      const params = new URLSearchParams();
      if (selectedContextAgent) {
        params.set("agent_name", selectedContextAgent);
      }
      const payload = await fetchJson(
        `/context${params.toString() ? `?${params.toString()}` : ""}`,
      );
      setContextView(payload);
      return payload;
    } catch (error) {
      setContextView({ error: error.message });
      if (throwOnError) {
        throw error;
      }
      return null;
    }
  }

  async function startRun() {
    if (!prompt.trim()) {
      return;
    }
    const result = await fetchJson("/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: prompt.trim(),
        source: "react-ui",
        agent_name: targetAgent.trim() || null,
      }),
    });
    setRunResult(result);
    setPrompt("");
    await refresh({ preserveContext: true });
  }

  async function openProject(initialize) {
    if (!projectRoot.trim()) {
      return;
    }
    const endpoint = initialize ? "/projects/init" : "/projects/open";
    await fetchJson(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        root: projectRoot.trim(),
        entry_agent: projectEntryAgent.trim() || "orchestrator",
        provider: projectProvider.trim() || "claude",
      }),
    });
    await refresh();
  }

  async function createAgent() {
    if (!newAgentName.trim()) {
      return;
    }
    await fetchJson("/agents", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: newAgentName.trim(),
        provider: newAgentProvider.trim() || "claude",
        template: newAgentTemplate.trim() || "default",
      }),
    });
    setNewAgentName("");
    await refresh();
  }

  async function resolveApproval(approvalId, approved) {
    await fetchJson(`/approvals/${approvalId}/${approved ? "approve" : "deny"}`, {
      method: "POST",
    });
    await refresh({ preserveContext: true });
  }

  async function pinMemory(proposalId, pinned) {
    await fetchJson(`/memory/${proposalId}/pin`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pinned }),
    });
    await refresh({ preserveContext: true });
  }

  async function syncMemory(proposalId) {
    await fetchJson(`/memory/${proposalId}/sync`, { method: "POST" });
    await refresh({ preserveContext: true });
  }

  async function rejectMemory(proposalId) {
    await fetchJson(`/memory/${proposalId}/reject`, { method: "POST" });
    await refresh({ preserveContext: true });
  }

  async function installConnector() {
    if (!catalogAgent.trim() || !connectorId.trim()) {
      return;
    }
    const result = await fetchJson("/connectors/install", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        agent_name: catalogAgent.trim(),
        connector_id: connectorId.trim(),
      }),
    });
    setCatalogResult(result);
    await refresh({ preserveContext: true });
  }

  async function installSkill() {
    if (!catalogAgent.trim() || !skillId.trim()) {
      return;
    }
    const result = await fetchJson("/skills/install", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        agent_name: catalogAgent.trim(),
        skill_id: skillId.trim(),
        no_deps: false,
      }),
    });
    setCatalogResult(result);
    await refresh({ preserveContext: true });
  }

  async function previewCompact() {
    const result = await fetchJson("/compact/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        agent_name: selectedContextAgent || null,
        reason: "react_preview",
      }),
    });
    setContextView(result);
    await refresh({ preserveContext: true });
  }

  async function applyCompact() {
    const result = await fetchJson("/compact/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        agent_name: selectedContextAgent || null,
        reason: "react_apply",
      }),
    });
    setContextView(result);
    await refresh({ preserveContext: true });
  }

  async function rejectCompact() {
    const result = await fetchJson("/compact/reject", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        agent_name: selectedContextAgent || null,
      }),
    });
    setContextView(result);
    await refresh({ preserveContext: true });
  }

  async function loadMemoryFile(scopeOverride = "", filenameOverride = "") {
    const scope = scopeOverride || memoryEditor.scope || "agent";
    const filename = filenameOverride || memoryEditor.filename;
    const params = new URLSearchParams({ scope });
    if (selectedMemoryAgent) {
      params.set("agent_name", selectedMemoryAgent);
    }
    if (filename) {
      params.set("filename", filename);
    }
    const result = await fetchJson(`/memory-files/content?${params.toString()}`);
    setMemoryEditor({
      scope: result.scope,
      filename: result.filename,
      revisionId: result.current_revision_id || "",
      content: result.content || "",
    });
    setMemoryHistory([]);
    setContextView(result);
  }

  async function loadMemoryRevision() {
    if (!memoryEditor.revisionId.trim()) {
      return;
    }
    const params = new URLSearchParams({
      scope: memoryEditor.scope || "agent",
      revision_id: memoryEditor.revisionId.trim(),
    });
    if (selectedMemoryAgent) {
      params.set("agent_name", selectedMemoryAgent);
    }
    if (memoryEditor.filename.trim()) {
      params.set("filename", memoryEditor.filename.trim());
    }
    const result = await fetchJson(`/memory-files/revision?${params.toString()}`);
    setMemoryEditor((current) => ({
      ...current,
      scope: result.scope,
      filename: result.filename,
      content: result.content || "",
    }));
    setContextView(result);
  }

  async function loadMemoryHistory(scopeOverride = "", filenameOverride = "") {
    const scope = scopeOverride || memoryEditor.scope || "agent";
    const filename = filenameOverride || memoryEditor.filename;
    const params = new URLSearchParams({ scope });
    if (selectedMemoryAgent) {
      params.set("agent_name", selectedMemoryAgent);
    }
    if (filename) {
      params.set("filename", filename);
    }
    const result = await fetchJson(`/memory-files/revisions?${params.toString()}`);
    const preferredRevision =
      result.revisions.find((item) => item.current) || result.revisions[0] || null;
    setMemoryEditor((current) => ({
      ...current,
      scope: result.scope,
      filename: result.filename,
      revisionId: preferredRevision ? preferredRevision.id : current.revisionId,
    }));
    setMemoryHistory(result.revisions || []);
    setContextView(result);
  }

  async function saveMemoryFile() {
    const result = await fetchJson("/memory-files/content", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scope: memoryEditor.scope || "agent",
        filename: memoryEditor.filename.trim(),
        content: memoryEditor.content,
        agent_name: selectedMemoryAgent || null,
        append: false,
      }),
    });
    setMemoryEditor({
      scope: result.scope,
      filename: result.filename,
      revisionId: result.current_revision_id || "",
      content: result.content || "",
    });
    setContextView(result);
    await refresh({ preserveContext: true });
  }

  async function restoreMemoryFile() {
    if (!memoryEditor.revisionId.trim()) {
      return;
    }
    const result = await fetchJson("/memory-files/restore", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scope: memoryEditor.scope || "agent",
        filename: memoryEditor.filename.trim(),
        revision_id: memoryEditor.revisionId.trim(),
        agent_name: selectedMemoryAgent || null,
      }),
    });
    setMemoryEditor({
      scope: result.scope,
      filename: result.filename,
      revisionId: result.current_revision_id || "",
      content: result.content || "",
    });
    setContextView(result);
    await loadMemoryHistory(result.scope, result.filename);
    await refresh({ preserveContext: true });
  }

  async function syncMemoryFile(scopeOverride = "", filenameOverride = "") {
    const result = await fetchJson("/memory-files/sync", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scope: scopeOverride || memoryEditor.scope || "agent",
        filename: filenameOverride || memoryEditor.filename.trim(),
        revision_id: memoryEditor.revisionId.trim(),
        agent_name: selectedMemoryAgent || null,
      }),
    });
    setContextView(result);
    await loadMemoryHistory(
      result.scope || scopeOverride || memoryEditor.scope,
      result.filename || filenameOverride || memoryEditor.filename,
    );
    await refresh({ preserveContext: true });
  }

  async function validateWorkflow() {
    const result = await fetchJson("/workflow/validate", { method: "POST" });
    setContextView(result);
  }

  async function applyDoc(proposalId) {
    await fetchJson(`/docs/proposals/${proposalId}/apply`, { method: "POST" });
    await refresh({ preserveContext: true });
  }

  async function rejectDoc(proposalId) {
    await fetchJson(`/docs/proposals/${proposalId}/reject`, { method: "POST" });
    await refresh({ preserveContext: true });
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Local-First Agent Control Plane</p>
          <h1>ContextClaw Studio</h1>
          <p className="hero-copy">
            Project memory, approvals, compaction, and ContextGraph sync in one place.
          </p>
        </div>
        <div className="hero-meta">
          <div className="pill">
            {project?.root || "No project loaded"}
          </div>
          <div className="pill subtle">
            Branch {project?.git?.branch || "n/a"}
          </div>
          <div className={`pill subtle live-status ${streamStatus}`}>
            {streamStatusLabel}
          </div>
        </div>
      </header>

      {errorMessage ? <div className="banner error">{errorMessage}</div> : null}

      <main className="card-grid">
        <Card title="Project">
          <input
            value={projectRoot}
            onChange={(event) => setProjectRoot(event.target.value)}
            placeholder="Project root path"
          />
          <input
            value={projectEntryAgent}
            onChange={(event) => setProjectEntryAgent(event.target.value)}
            placeholder="Entry agent"
          />
          <input
            value={projectProvider}
            onChange={(event) => setProjectProvider(event.target.value)}
            placeholder="Default provider"
          />
          <div className="inline-actions wrap">
            <button onClick={() => openProject(false)}>Open Existing Project</button>
            <button onClick={() => openProject(true)}>Init Project</button>
          </div>
        </Card>

        <Card title="Run Prompt">
          <input
            value={targetAgent}
            onChange={(event) => setTargetAgent(event.target.value)}
            placeholder="Optional target agent"
          />
          <textarea
            rows={6}
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="Describe the task for the orchestrator or routed agent"
          />
          <button onClick={startRun}>Start Run</button>
          <JsonBlock value={runResult} />
        </Card>

        <Card title="Agents">
          <input
            value={newAgentName}
            onChange={(event) => setNewAgentName(event.target.value)}
            placeholder="New agent name"
          />
          <input
            value={newAgentProvider}
            onChange={(event) => setNewAgentProvider(event.target.value)}
            placeholder="Provider"
          />
          <input
            value={newAgentTemplate}
            onChange={(event) => setNewAgentTemplate(event.target.value)}
            placeholder="Template"
          />
          <button onClick={createAgent}>Create Agent</button>
          <ul className="list">
            {agents.map((agent) => (
              <li key={agent.name}>
                <strong>{agent.name}</strong>
                <span>{agent.provider}</span>
              </li>
            ))}
          </ul>
        </Card>

        <Card title="Runs">
          <ul className="list">
            {runs.slice(0, 8).map((run) => (
              <li key={run.id}>
                <strong>{run.selected_agent}</strong>
                <span>{run.status}</span>
                <p>{previewText(run.prompt, 140)}</p>
              </li>
            ))}
          </ul>
        </Card>

        <Card title="Live Feed">
          <ul className="list compact">
            {liveEvents.length ? (
              liveEvents.map((event) => (
                <li key={`${event.id}-${event.type}`}>
                  <strong>{event.type}</strong>
                  <span>
                    {event.source} · run {event.run_id}
                  </span>
                  <p>{describeLiveEvent(event)}</p>
                </li>
              ))
            ) : (
              <li>
                <strong>No live events yet</strong>
                <p>Start a run or approve an action to watch Studio update live.</p>
              </li>
            )}
          </ul>
        </Card>

        <Card title="Catalog">
          <input
            value={catalogAgent}
            onChange={(event) => setCatalogAgent(event.target.value)}
            placeholder="Agent name"
          />
          <input
            value={connectorId}
            onChange={(event) => setConnectorId(event.target.value)}
            placeholder="Connector id"
          />
          <button onClick={installConnector}>Install Connector</button>
          <input
            value={skillId}
            onChange={(event) => setSkillId(event.target.value)}
            placeholder="Skill id"
          />
          <button onClick={installSkill}>Install Skill</button>
          <JsonBlock value={catalogResult} />
        </Card>

        <Card title="Approvals">
          <ul className="list">
            {approvals.map((approval) => (
              <li key={approval.id}>
                <strong>{approval.tool_name}</strong>
                <p>{previewText(approval.arguments, 100)}</p>
                <div className="inline-actions">
                  <button onClick={() => resolveApproval(approval.id, true)}>
                    Approve
                  </button>
                  <button
                    className="secondary"
                    onClick={() => resolveApproval(approval.id, false)}
                  >
                    Deny
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </Card>

        <Card title="Memory Queue">
          <ul className="list">
            {memoryQueue.map((proposal) => (
              <li key={proposal.id}>
                <strong>{proposal.status}</strong>
                <p>{proposal.content}</p>
                <div className="inline-actions">
                  <button onClick={() => pinMemory(proposal.id, !proposal.pinned)}>
                    {proposal.pinned ? "Unpin" : "Pin"}
                  </button>
                  <button onClick={() => syncMemory(proposal.id)}>Sync</button>
                  <button
                    className="secondary"
                    onClick={() => rejectMemory(proposal.id)}
                  >
                    Reject
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </Card>

        <Card title="Memory Files">
          <input
            value={memoryFileAgent}
            onChange={(event) => setMemoryFileAgent(event.target.value)}
            placeholder="Optional memory-file agent"
          />
          <input
            value={memoryEditor.scope}
            onChange={(event) =>
              setMemoryEditor((current) => ({
                ...current,
                scope: event.target.value,
              }))
            }
            placeholder="Scope"
          />
          <input
            value={memoryEditor.filename}
            onChange={(event) =>
              setMemoryEditor((current) => ({
                ...current,
                filename: event.target.value,
              }))
            }
            placeholder="Filename"
          />
          <input
            value={memoryEditor.revisionId}
            onChange={(event) =>
              setMemoryEditor((current) => ({
                ...current,
                revisionId: event.target.value,
              }))
            }
            placeholder="Revision id"
          />
          <textarea
            rows={8}
            value={memoryEditor.content}
            onChange={(event) =>
              setMemoryEditor((current) => ({
                ...current,
                content: event.target.value,
              }))
            }
            placeholder="Memory file content"
          />
          <div className="inline-actions wrap">
            <button onClick={() => loadMemoryFile()}>Load</button>
            <button onClick={loadMemoryRevision}>Load Revision</button>
            <button onClick={() => loadMemoryHistory()}>History</button>
            <button onClick={saveMemoryFile}>Save</button>
            <button onClick={restoreMemoryFile}>Restore Revision</button>
            <button onClick={() => syncMemoryFile()}>Sync To ContextGraph</button>
          </div>
          <ul className="list compact">
            {memoryFiles.map((item) => (
              <li key={`${item.scope}-${item.filename}`}>
                <strong>
                  {item.scope}: {item.filename}
                </strong>
                <span>
                  {item.token_estimate} tokens, {item.revision_count || 0} revisions
                </span>
                <p>
                  current {item.current_revision_id || "n/a"} | synced{" "}
                  {item.last_synced_revision_id || "n/a"}
                </p>
                <div className="inline-actions">
                  <button onClick={() => loadMemoryFile(item.scope, item.filename)}>
                    Load
                  </button>
                  <button onClick={() => loadMemoryHistory(item.scope, item.filename)}>
                    History
                  </button>
                  <button onClick={() => syncMemoryFile(item.scope, item.filename)}>
                    Sync
                  </button>
                </div>
              </li>
            ))}
          </ul>
          <pre className="history-block">
            {memoryHistory.length
              ? memoryHistory
                  .map((item) => {
                    const flags = [];
                    if (item.current) flags.push("current");
                    if (item.synced_memory_id) flags.push(`synced:${item.synced_memory_id}`);
                    if (item.source_revision_id) flags.push(`from:${item.source_revision_id}`);
                    return `${item.id} | ${item.action} | ${item.token_estimate || 0} tokens${
                      flags.length ? ` | ${flags.join(", ")}` : ""
                    }`;
                  })
                  .join("\n")
              : "No revision history loaded."}
          </pre>
        </Card>

        <Card title="Context">
          <input
            value={contextAgent}
            onChange={(event) => setContextAgent(event.target.value)}
            placeholder="Optional context agent"
          />
          <div className="inline-actions wrap">
            <button onClick={() => refreshContext()}>Inspect Context</button>
            <button onClick={previewCompact}>Preview Compact</button>
            <button onClick={applyCompact}>Apply Compact</button>
            <button onClick={rejectCompact}>Reject Pending</button>
          </div>
          <JsonBlock value={contextView} />
        </Card>

        <Card title="Docs Proposals">
          <ul className="list">
            {docsQueue.map((proposal) => (
              <li key={proposal.id}>
                <strong>{proposal.path}</strong>
                <p>{previewText(proposal.summary, 100)}</p>
                <div className="inline-actions">
                  <button onClick={() => applyDoc(proposal.id)}>Apply</button>
                  <button
                    className="secondary"
                    onClick={() => rejectDoc(proposal.id)}
                  >
                    Reject
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </Card>

        <Card
          title="Workflow"
          actions={<button onClick={validateWorkflow}>Validate Workflow</button>}
        >
          <JsonBlock value={workflow.config || {}} />
        </Card>

        <Card title="ContextGraph">
          <JsonBlock value={contextGraph} />
        </Card>
      </main>
    </div>
  );
}
