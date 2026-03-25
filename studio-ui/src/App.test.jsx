import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

const { invokeMock } = vi.hoisted(() => ({
  invokeMock: vi.fn(),
}));

vi.mock("@tauri-apps/api/core", () => ({
  invoke: invokeMock,
}));

let eventSources = [];

class FakeEventSource {
  constructor(url) {
    this.url = url;
    this.listeners = new Map();
    this.onopen = null;
    this.onerror = null;
    this.closed = false;
    eventSources.push(this);
  }

  addEventListener(name, handler) {
    const handlers = this.listeners.get(name) || [];
    handlers.push(handler);
    this.listeners.set(name, handlers);
  }

  removeEventListener(name, handler) {
    const handlers = this.listeners.get(name) || [];
    this.listeners.set(
      name,
      handlers.filter((entry) => entry !== handler),
    );
  }

  close() {
    this.closed = true;
  }

  open() {
    if (typeof this.onopen === "function") {
      this.onopen();
    }
  }

  emit(name, payload) {
    const handlers = this.listeners.get(name) || [];
    handlers.forEach((handler) =>
      handler({ data: JSON.stringify(payload) }),
    );
  }
}

function jsonResponse(payload, ok = true, status = 200) {
  return {
    ok,
    status,
    async json() {
      return payload;
    },
  };
}

function errorResponse(detail, status = 400) {
  return jsonResponse({ detail }, false, status);
}

function requestPath(input) {
  const value = String(input);
  if (/^https?:\/\//.test(value)) {
    const url = new URL(value);
    return `${url.pathname}${url.search}`;
  }
  return value;
}

function installFetchMock(overrides = {}) {
  const defaultPayloads = {
    "/projects/current": { root: "/tmp/project", git: { branch: "main" } },
    "/agents": [{ name: "orchestrator", provider: "claude" }],
    "/runs": [
      {
        id: "run_1",
        selected_agent: "orchestrator",
        status: "completed",
        prompt: "Summarize the deployment plan.",
      },
    ],
    "/approvals": [],
    "/memory": [],
    "/docs/proposals": [],
    "/workflow": { config: { entry_agent: "orchestrator" } },
    "/contextgraph/status": { linked_agents: 1, healthy: true },
    "/memory-files": [
      {
        scope: "agent",
        filename: "MEMORY.md",
        token_estimate: 42,
        revision_count: 2,
        current_revision_id: "rev_current",
        last_synced_revision_id: "rev_previous",
      },
    ],
    "/context": {
      agent: "orchestrator",
      budget: { total_tokens: 120, available_tokens: 4000, status: "healthy" },
    },
    "/memory-files/content?scope=agent&filename=MEMORY.md":
      {
        scope: "agent",
        filename: "MEMORY.md",
        current_revision_id: "rev_current",
        content: "# Agent Memory\n\nKeep release notes concise.\n",
      },
    "/memory-files/revisions?scope=agent&filename=MEMORY.md":
      {
        scope: "agent",
        filename: "MEMORY.md",
        revisions: [
          {
            id: "rev_current",
            action: "write",
            token_estimate: 42,
            current: true,
            synced_memory_id: "mem_1",
            source_revision_id: "",
          },
          {
            id: "rev_previous",
            action: "baseline",
            token_estimate: 39,
            current: false,
            synced_memory_id: "",
            source_revision_id: "",
          },
        ],
      },
  };

  global.fetch = vi.fn(async (input, options = {}) => {
    const path = requestPath(input);
    const method = (options.method || "GET").toUpperCase();
    const key = `${method} ${path}`;
    if (key in overrides) {
      return jsonResponse(overrides[key]);
    }
    if (path in overrides) {
      return jsonResponse(overrides[path]);
    }
    if (path in defaultPayloads) {
      return jsonResponse(defaultPayloads[path]);
    }
    throw new Error(`Unhandled fetch for ${method} ${path}`);
  });
}

describe("Studio App", () => {
  beforeEach(() => {
    eventSources = [];
    installFetchMock();
    invokeMock.mockReset();
    window.EventSource = FakeEventSource;
    global.EventSource = FakeEventSource;
  });

  afterEach(() => {
    cleanup();
    delete window.__TAURI_INTERNALS__;
    delete window.EventSource;
    delete global.EventSource;
    vi.restoreAllMocks();
  });

  it("renders the loaded project and primary dashboard sections", async () => {
    render(<App />);

    expect(await screen.findByText("ContextClaw Studio")).toBeInTheDocument();
    expect(await screen.findByText("/tmp/project")).toBeInTheDocument();
    expect(await screen.findByText("Branch main")).toBeInTheDocument();
    expect(await screen.findByText("Run Prompt")).toBeInTheDocument();
    expect(await screen.findByText("Memory Files")).toBeInTheDocument();
    expect(await screen.findByText("ContextGraph")).toBeInTheDocument();
  });

  it("loads and renders memory-file revision history", async () => {
    render(<App />);

    const entry = await screen.findByText("agent: MEMORY.md");
    const container = entry.closest("li");
    if (!container) {
      throw new Error("Expected memory-file list item container");
    }
    fireEvent.click(within(container).getByRole("button", { name: "History" }));

    await waitFor(() => {
      expect(screen.getByText(/rev_current \| write \| 42 tokens \| current, synced:mem_1/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/rev_previous \| baseline \| 39 tokens/i)).toBeInTheDocument();
  });

  it("subscribes to the live event stream and renders incoming events", async () => {
    render(<App />);

    await waitFor(() => {
      expect(eventSources).toHaveLength(1);
    });
    expect(eventSources[0].url).toBe("/events/stream");

    eventSources[0].open();
    eventSources[0].emit("studio.event", {
      id: 91,
      run_id: "run_live",
      source: "runner",
      type: "run.started",
      payload: { prompt: "Deploy the studio safely." },
    });

    await waitFor(() => {
      expect(screen.getByText("Live feed connected")).toBeInTheDocument();
      expect(screen.getByText("run.started")).toBeInTheDocument();
    });
    expect(screen.getByText(/Deploy the studio safely/i)).toBeInTheDocument();
  });

  it("resolves the runtime API base from the Tauri shell", async () => {
    window.__TAURI_INTERNALS__ = {};
    invokeMock.mockResolvedValue({
      baseUrl: "http://127.0.0.1:5123",
      port: 5123,
    });

    render(<App />);

    await waitFor(() => {
      expect(invokeMock).toHaveBeenCalledWith("studio_info");
    });
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        "http://127.0.0.1:5123/projects/current",
        undefined,
      );
    });
    await waitFor(() => {
      expect(eventSources).toHaveLength(1);
    });
    expect(eventSources[0].url).toBe("http://127.0.0.1:5123/events/stream");
  });

  it("renders the project bootstrap controls when no project is open", async () => {
    global.fetch = vi.fn(async (input) => {
      const path = requestPath(input);
      if (path === "/projects/current") {
        return errorResponse("No project is currently open");
      }
      throw new Error(`Unhandled fetch for ${path}`);
    });

    render(<App />);

    expect(await screen.findByText("No project loaded")).toBeInTheDocument();
    expect(await screen.findByText("Project")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Project root path")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open Existing Project" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Init Project" })).toBeInTheDocument();
  });
});
