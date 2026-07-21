const state = {
  lastResponse: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const elements = {
  form: $("#chatForm"),
  prompt: $("#prompt"),
  messages: $("#messages"),
  payload: $("#payloadEditor"),
  sessionId: $("#sessionId"),
  userId: $("#userId"),
  tenantId: $("#tenantId"),
  dryRun: $("#dryRun"),
  requestState: $("#requestState"),
  decisionLine: $("#decisionLine"),
  draftOutput: $("#draftOutput"),
  traceList: $("#traceList"),
  memoryOutput: $("#memoryOutput"),
  knowledgeTitle: $("#knowledgeTitle"),
  knowledgeTags: $("#knowledgeTags"),
  knowledgeContent: $("#knowledgeContent"),
  knowledgeQuery: $("#knowledgeQuery"),
  knowledgeOutput: $("#knowledgeOutput"),
  llmStatus: $("#llmStatus"),
  ragStatus: $("#ragStatus"),
  memoryStatus: $("#memoryStatus"),
};

elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await sendChat();
});

$("#memoryBtn").addEventListener("click", loadMemory);
$("#clearBtn").addEventListener("click", () => {
  elements.prompt.value = "";
  elements.payload.value = "{}";
});
$("#ingestBtn").addEventListener("click", ingestKnowledge);
$("#knowledgeStatsBtn").addEventListener("click", loadKnowledgeStats);
$("#knowledgeSearchBtn").addEventListener("click", searchKnowledge);

$$(".quick").forEach((button) => {
  button.addEventListener("click", () => {
    elements.prompt.value = button.dataset.template || "";
  });
});

$$(".tab").forEach((button) => {
  button.addEventListener("click", () => activateTab(button.dataset.tab));
});

async function sendChat() {
  const text = elements.prompt.value.trim();
  if (!text) return;

  let payload;
  try {
    payload = JSON.parse(elements.payload.value || "{}");
  } catch (error) {
    setState("Payload JSON 无效");
    activateTab("draft");
    elements.draftOutput.textContent = String(error.message || error);
    return;
  }

  appendMessage("user", text);
  setState("Running");

  const body = {
    sessionId: elements.sessionId.value.trim() || "web-demo",
    userId: elements.userId.value.trim() || "anonymous",
    tenantId: elements.tenantId.value.trim(),
    locale: "zh_CN",
    text,
    payload,
    dryRun: elements.dryRun.checked,
  };

  try {
    const response = await fetchJson("/api/chat", {
      method: "POST",
      body: JSON.stringify(body),
    });
    state.lastResponse = response;
    renderResponse(response);
    setState(response.ok ? "Ready" : "Needs Review");
  } catch (error) {
    setState("Failed");
    appendMessage("assistant", String(error.message || error), ["error"]);
  }
}

async function loadMemory() {
  const sessionId = encodeURIComponent(elements.sessionId.value.trim() || "web-demo");
  try {
    const response = await fetchJson(`/api/memory?sessionId=${sessionId}`);
    elements.memoryOutput.textContent = pretty(response.memory || {});
    activateTab("memory");
  } catch (error) {
    elements.memoryOutput.textContent = String(error.message || error);
  }
}

async function loadHealth() {
  try {
    const response = await fetchJson("/api/health");
    const llm = response.llm || {};
    const label = llm.mode === "online" ? `Online · ${llm.model || "model"}` : "Offline";
    elements.llmStatus.textContent = label;
    elements.memoryStatus.textContent = response.dataDir ? "SQLite" : "Ready";
  } catch (error) {
    elements.llmStatus.textContent = "Unknown";
  }
}

async function ingestKnowledge() {
  const body = {
    title: elements.knowledgeTitle.value.trim() || "Untitled",
    content: elements.knowledgeContent.value.trim(),
    source: "web-console",
    tags: parseTags(elements.knowledgeTags.value),
    replace: true,
  };
  try {
    const response = await fetchJson("/api/knowledge/ingest", {
      method: "POST",
      body: JSON.stringify(body),
    });
    renderKnowledge(response);
  } catch (error) {
    elements.knowledgeOutput.textContent = String(error.message || error);
  }
}

async function loadKnowledgeStats() {
  try {
    const response = await fetchJson("/api/knowledge/stats");
    renderKnowledge(response);
  } catch (error) {
    elements.knowledgeOutput.textContent = String(error.message || error);
  }
}

async function searchKnowledge() {
  const query = elements.knowledgeQuery.value.trim();
  if (!query) return;
  try {
    const response = await fetchJson("/api/knowledge/search", {
      method: "POST",
      body: JSON.stringify({ query, limit: 5 }),
    });
    renderKnowledge(response);
  } catch (error) {
    elements.knowledgeOutput.textContent = String(error.message || error);
  }
}

function renderResponse(response) {
  const decision = response.decision || {};
  const agents = decision.selected_agents || decision.selectedAgents || [];
  const issues = response.issues || [];
  const warningText = issues.length ? `, ${issues.length} 个提示` : "";
  elements.decisionLine.textContent = `${agents.join(" / ") || "no agent"}, confidence ${decision.confidence ?? "-"}${warningText}`;

  const lines = [];
  for (const result of response.results || []) {
    const answer = result.output && result.output.answer;
    lines.push(answer || `${result.agent}: ${result.ok ? "完成" : "失败"}`);
  }
  if (issues.length) {
    lines.push(...issues.map((issue) => `${issue.severity || "info"}: ${issue.message}`));
  }
  appendMessage("assistant", lines.join("\n\n") || "已完成", agents);

  elements.draftOutput.textContent = pretty(compactOutput(response));
  elements.memoryOutput.textContent = pretty(response.memory || {});
  renderTrace(response.results || []);
  activateTab("draft");
}

function compactOutput(response) {
  const output = {};
  for (const result of response.results || []) {
    output[result.agent] = result.output || {};
  }
  output.retrieved = response.retrieved || [];
  output.issues = response.issues || [];
  return output;
}

function renderTrace(results) {
  elements.traceList.innerHTML = "";
  for (const result of results) {
    for (const step of result.steps || []) {
      const item = document.createElement("div");
      item.className = "trace-item";
      item.innerHTML = `
        <div class="trace-head">
          <span>${escapeHtml(step.agent)} / ${escapeHtml(step.action)}</span>
          <span class="${step.status === "ok" ? "status-ok" : "status-error"}">${escapeHtml(step.status)}</span>
        </div>
        <p>${escapeHtml(step.thought || "")}</p>
        <p>${escapeHtml(step.observation || "")}</p>
      `;
      elements.traceList.appendChild(item);
    }
  }
}

function renderKnowledge(response) {
  elements.knowledgeOutput.textContent = pretty(response);
  const total = response.total_chunks ?? response.after_chunks ?? (response.stats && response.stats.total_chunks);
  if (typeof total === "number") {
    elements.ragStatus.textContent = `${total} chunks`;
  }
  activateTab("knowledge");
}

function appendMessage(role, text, chips = []) {
  const message = document.createElement("div");
  message.className = `message ${role}`;
  const avatar = role === "user" ? "U" : "A";
  const chipHtml = chips.length ? `<div class="agent-chips">${chips.map((chip) => `<span class="chip">${escapeHtml(chip)}</span>`).join("")}</div>` : "";
  message.innerHTML = `
    <div class="avatar">${avatar}</div>
    <div class="bubble">
      <div class="bubble-title">${role === "user" ? "User" : "Assistant"}</div>
      <p>${escapeHtml(text).replace(/\n/g, "<br />")}</p>
      ${chipHtml}
    </div>
  `;
  elements.messages.appendChild(message);
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function activateTab(name) {
  $$(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === name));
  $$(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `tab-${name}`));
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json();
  if (!response.ok && data.error) {
    throw new Error(data.error);
  }
  return data;
}

function setState(text) {
  elements.requestState.textContent = text;
}

function parseTags(value) {
  return String(value || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

loadHealth();
loadMemory();
loadKnowledgeStats();
