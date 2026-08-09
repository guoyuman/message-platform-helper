import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

const DEFAULT_SESSION = "web-demo";
const WELCOME_MESSAGES = [
  { role: "assistant", text: "可以直接描述模板翻译、邮箱通道配置、也可以询问消息平台使用说明和发送失败排查。" },
];
const GROUP_LABELS = ["今天", "昨天", "7天内", "一个月内", "更早"];

function App() {
  const [sessionId, setSessionId] = useState(DEFAULT_SESSION);
  const [userId, setUserId] = useState("consultant-001");
  const [tenantId, setTenantId] = useState("tenant-demo");
  const [dryRun, setDryRun] = useState(true);
  const [prompt, setPrompt] = useState("消息发送失败原因如何排查？");
  const [payload, setPayload] = useState("{}");
  const [messages, setMessages] = useState(WELCOME_MESSAGES);
  const [activeTab, setActiveTab] = useState("draft");
  const [memory, setMemory] = useState({});
  const [sessions, setSessions] = useState([]);
  const [localSessions, setLocalSessions] = useState([]);
  const [sessionTitles, setSessionTitles] = useState({});
  const [trace, setTrace] = useState([]);
  const [draft, setDraft] = useState({});
  const [stateText, setStateText] = useState("Ready");
  const [thinkingStage, setThinkingStage] = useState("");
  const [decisionLine, setDecisionLine] = useState("等待请求");
  const [health, setHealth] = useState({ llm: "Local", memory: "PostgreSQL", rag: "Ready" });
  const messagesEndRef = useRef(null);
  const didLoadTenantDataRef = useRef(false);

  useEffect(() => {
    loadHealth();
  }, []);

  useEffect(() => {
    if (didLoadTenantDataRef.current && !tenantId.trim()) return;
    didLoadTenantDataRef.current = true;
    loadSessions();
    loadMemory(sessionId);
  }, [tenantId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, thinkingStage, trace]);

  async function loadHealth() {
    try {
      const response = await fetchJson("/api/health");
      const llm = response.llm || {};
      setHealth({
        llm: llm.mode === "online" ? `Online ${llm.model || "model"}` : "Offline",
        memory: response.database ? "PostgreSQL" : "Ready",
        rag: "Ready",
      });
    } catch {
      setHealth((current) => ({ ...current, llm: "Unknown" }));
    }
  }

  async function loadSessions() {
    try {
      const response = await fetchJson(`/api/memory/sessions?tenantId=${encodeURIComponent(tenantId.trim())}`);
      const hydrated = await Promise.all((response.sessions || []).map(hydrateSession));
      setSessions(hydrated);
    } catch {
      setSessions([]);
    }
  }

  function memoryUrl(nextSessionId) {
    const params = new URLSearchParams();
    params.set("sessionId", nextSessionId || DEFAULT_SESSION);
    params.set("tenantId", tenantId.trim());
    return `/api/memory?${params.toString()}`;
  }

  async function hydrateSession(item) {
    try {
      const response = await fetchJson(memoryUrl(item.sessionId || DEFAULT_SESSION));
      const nextMemory = response.memory || {};
      return {
        ...item,
        memory: nextMemory,
        title: sessionTitle(item, nextMemory),
        updatedAt: item.updatedAt || nextMemory.updated_at || nextMemory.updatedAt,
      };
    } catch {
      return { ...item, title: sessionTitle(item, {}) };
    }
  }

  async function loadMemory(nextSessionId = sessionId, options = {}) {
    try {
      const response = await fetchJson(memoryUrl(nextSessionId || DEFAULT_SESSION));
      const nextMemory = response.memory || {};
      setMemory(nextMemory);
      if (options.syncMessages) {
        setMessages(memoryToMessages(nextMemory));
        setTrace([]);
        setDraft({});
        setDecisionLine(sessionTitle({ sessionId: nextSessionId }, nextMemory));
      }
      syncProfile(nextMemory);
      return nextMemory;
    } catch (error) {
      setMemory({ error: String(error.message || error) });
      if (options.syncMessages) setMessages([]);
      return {};
    }
  }

  async function sendChat(event) {
    event.preventDefault();
    if (thinkingStage) return;
    const text = prompt.trim();
    if (!text) return;
    let requestPayload;
    try {
      requestPayload = JSON.parse(payload || "{}");
    } catch (error) {
      setStateText("Payload JSON 无效");
      setDraft({ error: String(error.message || error) });
      setActiveTab("draft");
      return;
    }

    const body = {
      sessionId: sessionId.trim() || DEFAULT_SESSION,
      userId: userId.trim() || "anonymous",
      tenantId: tenantId.trim(),
      locale: "zh_CN",
      text,
      payload: requestPayload,
      dryRun,
    };
    const firstUserMessage = !messages.some((message) => message.role === "user");
    if (firstUserMessage) updateSessionTitle(body.sessionId, text);

    setMessages((items) => [...items, { role: "user", text }]);
    setTrace([{ label: "Started", text: "request queued", ok: true }]);
    setStateText("Running");
    setThinkingStage("正在接收并理解你的问题");

    try {
      await streamChat(body);
      await loadSessions();
      await loadMemory(body.sessionId);
    } catch (error) {
      setStateText("Failed");
      setMessages((items) => [...items, { role: "assistant", text: String(error.message || error), error: true }]);
    } finally {
      setThinkingStage("");
    }
  }

  async function streamChat(body) {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || response.statusText);
    }
    if (!response.body) {
      renderResponse(await fetchJson("/api/chat", { method: "POST", body: JSON.stringify(body) }));
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      parts.forEach(handleStreamChunk);
    }
    if (buffer.trim()) handleStreamChunk(buffer);
  }

  function handleStreamChunk(chunk) {
    const line = chunk.split("\n").find((item) => item.startsWith("data:"));
    if (!line) return;
    handleStreamEvent(JSON.parse(line.slice(5).trim()));
  }

  function handleStreamEvent(event) {
    if (event.type === "Started" || event.type === "Running") {
      setStateText(event.type);
      setThinkingStage(event.type === "Started" ? "正在建立请求" : "正在处理请求");
      setTrace((items) => [...items, { label: event.type, text: event.data?.message || "", ok: true }]);
      return;
    }
    if (event.type === "Reasoning") {
      const decision = event.data?.decision || {};
      const agents = decision.selected_agents || decision.selectedAgents || [];
      setThinkingStage("正在分析并选择处理方式");
      setDecisionLine(`${agents.join(" / ") || "no agent"}, confidence ${decision.confidence ?? "-"}`);
      setTrace((items) => [...items, { label: "Reasoning", text: decision.rationale || decision.intent || "decision ready", ok: true }]);
      return;
    }
    if (event.type === "ToolCall" && event.data?.step) {
      const step = event.data.step;
      setThinkingStage(`正在执行 ${step.agent || "处理任务"} → ${step.action || ""}`);
      setTrace((items) => [...items, { label: `${step.agent} / ${step.action}`, text: [step.thought, step.observation].filter(Boolean).join("\n"), ok: step.status === "ok" }]);
      return;
    }
    if (event.type === "Source") {
      const source = event.data?.source || {};
      setThinkingStage("正在查找相关资料");
      setTrace((items) => [...items, { label: "Source", text: source.title || source.source || source.id || "", ok: true }]);
      return;
    }
    if ((event.type === "Done" || event.type === "Error") && event.data?.response) {
      renderResponse(event.data.response);
    }
  }

  function renderResponse(response) {
    setThinkingStage("");
    const decision = response.decision || {};
    const agents = decision.selected_agents || decision.selectedAgents || [];
    const issues = response.issues || [];
    setDecisionLine(`${agents.join(" / ") || "no agent"}, confidence ${decision.confidence ?? "-"}${issues.length ? `, ${issues.length} 个提示` : ""}`);
    const lines = (response.results || []).map((result) => result.output?.answer || `${result.agent}: ${result.ok ? "完成" : "失败"}`);
    if (issues.length) lines.push(...issues.map((issue) => `${issue.severity || "info"}: ${issue.message}`));
    setMessages((items) => [...items, { role: "assistant", text: lines.join("\n\n") || "已完成", chips: agents }]);
    setDraft(compactOutput(response));
    setMemory(response.memory || {});
    setStateText(response.ok ? "Ready" : "Needs Review");
    setActiveTab("draft");
  }

  function newConversation() {
    const nextSessionId = createSessionId();
    const createdAt = Date.now() / 1000;
    setSessionId(nextSessionId);
    setMemory({ sessionId: nextSessionId, session_id: nextSessionId, recent_messages: [], updatedAt: createdAt, updated_at: createdAt });
    setMessages([]);
    setPrompt("");
    setTrace([]);
    setDraft({});
    setDecisionLine("新会话");
    setStateText("Ready");
    setActiveTab("draft");
    setLocalSessions((items) => [
      { sessionId: nextSessionId, title: "新会话", updatedAt: createdAt, recentMessageCount: 0, local: true },
      ...items,
    ]);
  }

  async function selectSession(nextSessionId) {
    setSessionId(nextSessionId);
    setStateText("Loading");
    await loadMemory(nextSessionId, { syncMessages: true });
    setStateText("Ready");
  }

  function updateSessionTitle(nextSessionId, text) {
    const title = titleFromText(text);
    const updatedAt = Date.now() / 1000;
    setSessionTitles((items) => ({ ...items, [nextSessionId]: title }));
    setLocalSessions((items) => {
      const nextItem = { sessionId: nextSessionId, title, updatedAt, recentMessageCount: 1, local: true };
      const exists = items.some((item) => item.sessionId === nextSessionId);
      return exists ? items.map((item) => (item.sessionId === nextSessionId ? { ...item, ...nextItem } : item)) : [nextItem, ...items];
    });
  }

  function syncProfile(nextMemory) {
    const profile = nextMemory?.profile || {};
    if (profile.user_id || profile.userId) setUserId(profile.user_id || profile.userId);
    if (profile.tenant_id || profile.tenantId) setTenantId(profile.tenant_id || profile.tenantId);
  }

  const groupedSessions = groupSessions(mergeSessions(sessions, localSessions, sessionTitles));
  const recentMessages = getRecentMessages(memory);
  const activeTitle = sessionTitles[sessionId] || sessionTitle({ sessionId }, memory);
  const lastUserIndex = messages.reduce((found, message, index) => (message.role === "user" ? index : found), -1);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">M</div>
          <div>
            <h1>消息平台助手</h1>
            <p>Agent Console</p>
          </div>
        </div>

        <button className="new-chat-button" type="button" onClick={newConversation}>+ 新建对话</button>

        <section className="history-panel">
          <div className="panel-title">会话历史</div>
          <div className="session-list">
            {GROUP_LABELS.map((label) => (
              groupedSessions[label]?.length ? (
                <div className="session-group" key={label}>
                  <div className="session-group-title">{label}</div>
                  {groupedSessions[label].map((item) => (
                    <button
                      className={`session-item ${item.sessionId === sessionId ? "active" : ""}`}
                      key={item.sessionId}
                      title={item.title}
                      type="button"
                      onClick={() => selectSession(item.sessionId)}
                    >
                      <strong>{item.title}</strong>
                      <span>{formatSessionTime(item.updatedAt)}</span>
                    </button>
                  ))}
                </div>
              ) : null
            ))}
            {mergeSessions(sessions, localSessions, sessionTitles).length ? null : <div className="empty-state">暂无历史会话</div>}
          </div>
        </section>

        <section className="panel compact">
          <div className="panel-title">用户信息</div>
          <label>
            <span>用户</span>
            <input value={userId} onChange={(event) => setUserId(event.target.value)} autoComplete="off" />
          </label>
        </section>

        <section className="panel compact">
          <div className="panel-title">租户信息</div>
          <label>
            <span>租户</span>
            <input value={tenantId} onChange={(event) => setTenantId(event.target.value)} autoComplete="off" />
          </label>
        </section>

        <section className="panel compact">
          <div className="panel-title">记忆信息</div>
          <Metric label="当前会话" value={activeTitle} />
          <Metric label="消息数量" value={`${recentMessages.length} 条`} />
          <div className="memory-summary">{memory.summary || "暂无摘要"}</div>
        </section>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <h2>{activeTitle}</h2>
            <p>{decisionLine}</p>
          </div>
          <div className="top-actions">
            <button className="icon-button" title="刷新会话" aria-label="刷新会话" type="button" onClick={() => { loadSessions(); loadMemory(); }}>↻</button>
            <button className="icon-button" title="清空输入" aria-label="清空输入" type="button" onClick={() => { setPrompt(""); setPayload("{}"); }}>×</button>
          </div>
        </header>

        <section className="chat-panel">
          <div className="messages" aria-live="polite">
            {messages.map((message, index) => {
              const isLastUser = message.role === "user" && index === lastUserIndex;
              return (
                <React.Fragment key={index}>
                  <Message message={message} />
                  {isLastUser && (trace.length || thinkingStage) ? (
                    <div className="message assistant">
                      <div className="avatar">A</div>
                      <div className="bubble">
                        <div className="trace-list">
                          {trace.map((item, traceIndex) => <TraceItem item={item} key={traceIndex} />)}
                        </div>
                        {thinkingStage ? <ThinkingLine stage={thinkingStage} /> : null}
                      </div>
                    </div>
                  ) : null}
                </React.Fragment>
              );
            })}
            <div ref={messagesEndRef} aria-hidden="true"></div>
          </div>
          <form className="composer" id="chatForm" onSubmit={sendChat}>
            <textarea value={prompt} rows="3" onChange={(event) => setPrompt(event.target.value)} />
            <div className="composer-footer">
              <button type="submit" className="primary" disabled={Boolean(thinkingStage)}>发送</button>
              <span id="requestState">{stateText}</span>
            </div>
          </form>
        </section>
      </main>

      <aside className="inspector">
        <section className="panel compact">
          <div className="panel-title">快速任务</div>
          <div className="quick-grid">
            {[
              ["给 ops@example.com 配置邮件通道并测试", "邮箱通道配置"],
              ["同步采购订单下所有模板到阿拉伯语", "模板同步翻译"],
              ["消息发送失败原因如何排查？", "失败排查"],
              ["消息平台如何使用？给我一份操作说明", "使用说明"],
            ].map(([value, label]) => <button className="quick" key={label} type="button" onClick={() => setPrompt(value)}>{label}</button>)}
          </div>
        </section>

        <section className="panel compact status-panel">
          <div className="panel-title">运行状态</div>
          <Metric label="LLM" value={health.llm} />
          <Metric label="RAG" value={health.rag} />
          <Metric label="Memory" value={health.memory} />
          <div className="toggle-row gap">
            <span>Dry Run</span>
            <label className="switch">
              <input type="checkbox" checked={dryRun} onChange={(event) => setDryRun(event.target.checked)} />
              <span></span>
            </label>
          </div>
        </section>

        {/*<div className="tabs" role="tablist">*/}
        {/*  <button className={`tab ${activeTab === "draft" ? "active" : ""}`} type="button" onClick={() => setActiveTab("draft")}>结果</button>*/}
        {/*  <button className={`tab ${activeTab === "trace" ? "active" : ""}`} type="button" onClick={() => setActiveTab("trace")}>轨迹</button>*/}
        {/*  <button className={`tab ${activeTab === "payload" ? "active" : ""}`} type="button" onClick={() => setActiveTab("payload")}>Payload</button>*/}
        {/*</div>*/}

        {/*<section className={`tab-panel ${activeTab === "draft" ? "active" : ""}`}>*/}
        {/*  <div className="panel-title">输出</div>*/}
        {/*  <pre className="code-output">{pretty(draft)}</pre>*/}
        {/*</section>*/}
        {/*<section className={`tab-panel ${activeTab === "trace" ? "active" : ""}`}>*/}
        {/*  <div className="panel-title">ReAct</div>*/}
        {/*  <div className="trace-list">*/}
        {/*    {trace.map((item, index) => <TraceItem item={item} key={index} />)}*/}
        {/*  </div>*/}
        {/*</section>*/}
        {/*<section className={`tab-panel ${activeTab === "payload" ? "active" : ""}`}>*/}
        {/*  <div className="panel-title">Payload</div>*/}
        {/*  <textarea className="code-input" spellCheck="false" value={payload} onChange={(event) => setPayload(event.target.value)} />*/}
        {/*</section>*/}
      </aside>
    </div>
  );
}

function Metric({ label, value }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

function Message({ message }) {
  return (
    <div className={`message ${message.role} ${message.error ? "error" : ""}`}>
      <div className="avatar">{message.role === "user" ? "U" : "A"}</div>
      <div className="bubble">
        <div className="bubble-title">{message.role === "user" ? "User" : "Assistant"}</div>
        <p>{message.text}</p>
        {message.chips?.length ? <div className="agent-chips">{message.chips.map((chip) => <span className="chip" key={chip}>{chip}</span>)}</div> : null}
      </div>
    </div>
  );
}

function ThinkingLine({ stage }) {
  return (
    <div className="thinking-line" role="status" aria-label={stage}>
      <span className="thinking-dots" aria-hidden="true">
        <i></i><i></i><i></i>
      </span>
      <span>{stage}</span>
    </div>
  );
}

function TraceItem({ item }) {
  const [expanded, setExpanded] = useState(false);
  const detail = [item.text].filter(Boolean).join("\n");
  const toggle = detail ? () => setExpanded((value) => !value) : undefined;
  return (
    <div
      className={`trace-item ${item.ok ? "ok" : "error"}${expanded ? " expanded" : ""}`}
      onClick={toggle}
      role={toggle ? "button" : undefined}
      tabIndex={toggle ? 0 : undefined}
      onKeyDown={toggle ? (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); toggle(); } } : undefined}
    >
      <div className="trace-head">
        <span className="trace-status" aria-hidden="true">{item.ok ? "✓" : "✗"}</span>
        <span className="trace-label" title={item.label}>{item.label}</span>
        {detail ? <span className="trace-chevron" aria-hidden="true">{expanded ? "▾" : "▸"}</span> : null}
      </div>
      {expanded && detail ? <pre className="trace-detail">{detail}</pre> : null}
    </div>
  );
}

function createSessionId() {
  return `web-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function getRecentMessages(memory) {
  return memory?.recent_messages || memory?.recentMessages || [];
}

function memoryToMessages(memory) {
  const items = getRecentMessages(memory).map((item) => ({
    role: item.role === "assistant" ? "assistant" : "user",
    text: String(item.text || ""),
  })).filter((item) => item.text);
  return items.length ? items : [];
}

function firstUserMessage(memory) {
  return getRecentMessages(memory).find((item) => item.role === "user" && item.text)?.text || "";
}

function titleFromText(text) {
  return String(text || "").replace(/\s+/g, " ").trim().slice(0, 80) || "新会话";
}

function sessionTitle(item, memory) {
  return titleFromText(firstUserMessage(memory) || item.title || item.summary || item.sessionId);
}

function mergeSessions(serverSessions, localSessions, sessionTitles) {
  const byId = new Map();
  for (const item of localSessions) {
    if (item.sessionId) byId.set(item.sessionId, { ...item, title: sessionTitles[item.sessionId] || item.title });
  }
  for (const item of serverSessions) {
    if (!item.sessionId) continue;
    const current = byId.get(item.sessionId) || {};
    byId.set(item.sessionId, {
      ...current,
      ...item,
      title: sessionTitles[item.sessionId] || item.title || current.title || sessionTitle(item, item.memory || {}),
    });
  }
  return Array.from(byId.values()).sort((a, b) => timestampMs(b.updatedAt) - timestampMs(a.updatedAt));
}

function groupSessions(items) {
  return items.reduce((groups, item) => {
    const label = sessionGroupLabel(item.updatedAt);
    groups[label] = [...(groups[label] || []), item];
    return groups;
  }, {});
}

function sessionGroupLabel(value) {
  const then = new Date(timestampMs(value));
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const day = new Date(then);
  day.setHours(0, 0, 0, 0);
  const diffDays = Math.floor((today.getTime() - day.getTime()) / 86400000);
  if (diffDays <= 0) return "今天";
  if (diffDays === 1) return "昨天";
  if (diffDays <= 7) return "7天内";
  if (diffDays <= 30) return "一个月内";
  return "更早";
}

function timestampMs(value) {
  if (typeof value === "number") return value < 1000000000000 ? value * 1000 : value;
  const parsed = Date.parse(value || "");
  return Number.isNaN(parsed) ? Date.now() : parsed;
}

function formatSessionTime(value) {
  const date = new Date(timestampMs(value));
  return date.toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function compactOutput(response) {
  const output = {};
  for (const result of response.results || []) output[result.agent] = result.output || {};
  output.retrieved = response.retrieved || [];
  output.issues = response.issues || [];
  return output;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

createRoot(document.getElementById("root")).render(<App />);
