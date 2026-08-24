/* ox-alpha harness — dashboard client (vanilla, no deps) */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = { models: [], suites: [], activeSuite: null, hasChat: false };

const PANEL_TITLES = { models: "Models", suites: "Suites", playground: "Playground" };

const PRESETS = [
  {
    title: "Long-context needle",
    hint: "retrieval stress test",
    text:
      "I will give you a log. Somewhere inside is the line: [config] CANARY_ROLLOUT_PERCENT for search-indexer is set to 17 percent. Ignore everything else and tell me only the canary rollout percent as a number.\n\n" +
      Array.from({ length: 40 }, (_, i) => `[log] seq=${i} svc=billing-worker region=us-east-1 flushed the write buffer in ${100 + i}ms`).join("\n") +
      "\n[config] CANARY_ROLLOUT_PERCENT for search-indexer is set to 17 percent.\n" +
      Array.from({ length: 40 }, (_, i) => `[log] seq=${100 + i} svc=auth-gateway region=eu-central-1 rotated the signing key in ${50 + i}ms`).join("\n"),
  },
  {
    title: "Tool-call JSON",
    hint: "agentic / structured output",
    text:
      "You have exactly these tools:\n- search_orders(customer_id, since)\n- refund_order(order_id, amount_cents)\n- send_email(to, subject)\n\nGoal: issue a refund of $12.50 on order ord_88213. Respond with ONE JSON object: {\"tool\": ..., \"arguments\": {...}}. Amounts are in cents.",
  },
  {
    title: "Fix the bug",
    hint: "coding reasoning",
    text:
      "This binary search for the rightmost match returns wrong indices for repeated values. Return the corrected function only, in a fenced python block.\n\ndef find_rightmost(values, target):\n    lo, hi = 0, len(values) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if values[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    if lo > 0 and values[lo - 1] == target:\n        return lo + 1\n    return -1",
  },
];

/* ---------- helpers ---------- */
function fmtInt(n) {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n % 1_000_000 ? 1 : 0)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(n % 1_000 ? 1 : 0)}K`;
  return String(n);
}
function fmtUsd(v) {
  if (v === null || v === undefined) return "$0.0000";
  return `$${Number(v).toFixed(4)}`;
}
function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch {
      /* ignore */
    }
    throw new Error(`HTTP ${res.status}: ${detail}`);
  }
  return res.json();
}

function showError(id, message) {
  const el = $(id);
  el.textContent = message;
  el.classList.remove("hidden");
}

function toast(message, kind = "") {
  const host = $("#toast-host");
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = message;
  host.appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    el.style.transform = "translateX(30px)";
    el.style.transition = "all 0.3s ease";
    setTimeout(() => el.remove(), 300);
  }, 3200);
}

/* ---------- theme ---------- */
function initTheme() {
  const saved = localStorage.getItem("oab-theme") || "dark";
  document.documentElement.setAttribute("data-theme", saved);
  $("#theme-ico").textContent = saved === "dark" ? "☾" : "☀";
}
$("#theme-toggle").addEventListener("click", () => {
  const cur = document.documentElement.getAttribute("data-theme");
  const next = cur === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("oab-theme", next);
  $("#theme-ico").textContent = next === "dark" ? "☾" : "☀";
});

/* ---------- health ---------- */
async function checkHealth() {
  const dot = $("#health-dot");
  const text = $("#health-text");
  try {
    const health = await api("/api/health");
    dot.className = `dot ${health.status === "ok" ? "ok" : "bad"}`;
    const parts = [`v${health.version}`];
    if (health.testing) parts.push("testing");
    parts.push(health.api_key_configured ? "key ✓" : "no key");
    if (health.free_models_only) parts.push("free tier");
    text.textContent = parts.join(" · ");
    $("#free-only-toggle").checked = !!health.free_models_only;
    if (!health.api_key_configured) {
      $("#pg-model").disabled = true;
      $("#chat-send").disabled = true;
    }
  } catch {
    dot.className = "dot bad";
    text.textContent = "API unreachable";
  }
}

/* ---------- models ---------- */
function renderModelStats() {
  const list = state.models;
  const free = list.filter((m) => m.is_free).length;
  const maxCtx = list.reduce((a, m) => Math.max(a, m.context_window), 0);
  const tools = list.filter((m) => m.supports_tools).length;
  const stats = [
    { val: list.length, lbl: "Models" },
    { val: free, lbl: "Free tier" },
    { val: fmtInt(maxCtx), lbl: "Max context" },
    { val: tools, lbl: "Tool-capable" },
  ];
  $("#model-stats").innerHTML = stats
    .map((s) => `<div class="stat"><div class="val">${s.val}</div><div class="lbl">${s.lbl}</div></div>`)
    .join("");
}

function renderModelRows(filter = "") {
  const tbody = $("#models-table tbody");
  const q = filter.trim().toLowerCase();
  const rows = state.models.filter(
    (m) => !q || m.id.toLowerCase().includes(q) || m.display_name.toLowerCase().includes(q),
  );
  tbody.innerHTML = "";
  if (rows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:var(--muted);padding:1.5rem">No models match “${esc(filter)}”.</td></tr>`;
    return;
  }
  for (const m of rows) {
    const freeBadge = m.is_free ? ' <span class="badge free">FREE</span>' : "";
    const caps = `<span class="cap">
        <span class="pill ${m.supports_vision ? "on" : ""}">vision</span>
        <span class="pill ${m.supports_tools ? "on" : ""}">tools</span>
      </span>`;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="model-name">${esc(m.display_name)}</span>${freeBadge}</td>
      <td><code>${esc(m.id)}</code></td>
      <td class="num">${fmtInt(m.context_window)}</td>
      <td class="num">${fmtInt(m.max_output_tokens)}</td>
      <td>${caps}</td>
      <td class="num">${m.prompt_per_million.toFixed(3)}</td>
      <td class="num">${m.completion_per_million.toFixed(3)}</td>`;
    tbody.appendChild(tr);
  }
}

async function loadModels() {
  const freeOnly = $("#free-only-toggle").checked;
  try {
    state.models = await api(`/api/models?free_only=${freeOnly}`);
  } catch (err) {
    showError("#models-error", err.message);
    return;
  }
  $("#models-error").classList.add("hidden");
  renderModelStats();
  renderModelRows($("#model-search").value);

  const select = $("#pg-model");
  const previous = select.value;
  select.innerHTML = "";
  for (const m of state.models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `${m.display_name}${m.is_free ? " · free" : ""}`;
    opt.selected = m.id === previous;
    select.appendChild(opt);
  }
}

$("#free-only-toggle").addEventListener("change", () => loadModels().catch(() => {}));
$("#model-search").addEventListener("input", (e) => renderModelRows(e.target.value));

/* ---------- suites ---------- */
async function loadSuites() {
  let suites;
  try {
    suites = await api("/api/suites");
  } catch (err) {
    showError("#suites-error", err.message);
    return;
  }
  state.suites = suites;
  const list = $("#suite-list");
  list.innerHTML = "";
  for (const suite of suites) {
    const btn = document.createElement("button");
    btn.className = "suite-item";
    btn.innerHTML = `<strong>${esc(suite.name)}</strong><small>${suite.tasks.length} task${suite.tasks.length === 1 ? "" : "s"}</small>`;
    btn.addEventListener("click", () => selectSuite(suite.name));
    list.appendChild(btn);
  }
  if (suites.length > 0) selectSuite(suites[0].name);
}

function selectSuite(name) {
  state.activeSuite = name;
  $$(".suite-item").forEach((el, i) => {
    el.classList.toggle("active", state.suites[i]?.name === name);
  });
  $("#task-detail").classList.add("hidden");
  $("#task-cards").classList.remove("hidden");
  renderTasks();
}

function stars(level) {
  return "★".repeat(level) + "☆".repeat(5 - level);
}

function renderTasks() {
  const suite = state.suites.find((s) => s.name === state.activeSuite);
  const container = $("#task-cards");
  container.innerHTML = "";
  if (!suite) return;
  for (const task of suite.tasks) {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <h4>${esc(task.title)}</h4>
      <div class="meta">
        <span class="tag">${esc(task.category)}</span>
        <span class="stars" title="difficulty">${stars(task.difficulty)}</span>
        <span class="tag grade">${esc(task.grader_type)}</span>
      </div>`;
    card.addEventListener("click", () => showTask(task));
    container.appendChild(card);
  }
}

function showTask(task) {
  const detail = $("#task-detail");
  detail.innerHTML = `
    <button class="detail-back" id="detail-back">← Back to ${esc(state.activeSuite)}</button>
    <h2>${esc(task.title)}</h2>
    <div class="meta">
      <span class="tag"><code>${esc(task.id)}</code></span>
      <span class="tag">${esc(task.category)}</span>
      <span class="stars">${stars(task.difficulty)}</span>
      <span class="tag grade">${esc(task.grader_type)}</span>
      ${task.target_file ? `<span class="tag">target: ${esc(task.target_file)}</span>` : ""}
      <span class="tag">${task.timeout_s}s</span>
    </div>
    <pre>${esc(task.prompt)}</pre>`;
  detail.classList.remove("hidden");
  $("#task-cards").classList.add("hidden");
  $("#detail-back").addEventListener("click", () => {
    detail.classList.add("hidden");
    $("#task-cards").classList.remove("hidden");
  });
}

/* ---------- playground ---------- */
function ensureChatMounted() {
  if (!state.hasChat) {
    $("#chat-empty")?.remove();
    state.hasChat = true;
  }
}

function addMsg(role, text, isHtml = false) {
  ensureChatMounted();
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  if (isHtml) div.innerHTML = text;
  else div.textContent = text;
  $("#chat-log").appendChild(div);
  $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
  return div;
}

async function sendChat(event) {
  event.preventDefault();
  const input = $("#chat-input");
  const prompt = input.value.trim();
  if (!prompt) return;
  input.value = "";
  autoGrow(input);
  addMsg("user", prompt);
  const sendBtn = $("#chat-send");
  sendBtn.disabled = true;
  const pending = addMsg("assistant", '<span class="typing"><i></i><i></i><i></i></span>', true);
  const startedAt = performance.now();
  try {
    const maxTokensRaw = $("#pg-max-tokens").value.trim();
    const body = {
      model: $("#pg-model").value,
      messages: [{ role: "user", content: prompt }],
      temperature: Number($("#pg-temp").value),
      ...(maxTokensRaw ? { max_tokens: Number(maxTokensRaw) } : {}),
    };
    const reply = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    pending.textContent = reply.content || "(empty response)";
    const elapsed = ((performance.now() - startedAt) / 1000).toFixed(2);
    const u = reply.usage;
    $("#chat-usage").innerHTML = [
      `<span class="chip"><b>${u.total_tokens}</b> tokens</span>`,
      `<span class="chip">${u.prompt_tokens}→${u.completion_tokens}</span>`,
      `<span class="chip">${fmtUsd(u.cost_usd)}</span>`,
      `<span class="chip"><b>${elapsed}</b>s</span>`,
      `<span class="chip">${esc(reply.finish_reason ?? "n/a")}</span>`,
    ].join("");
  } catch (err) {
    pending.className = "msg error";
    pending.textContent = err.message;
    $("#chat-usage").textContent = "";
    toast(err.message, "err");
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

function autoGrow(el) {
  el.style.height = "auto";
  el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
}

function renderPresets() {
  const host = $("#pg-presets");
  host.innerHTML = "";
  for (const p of PRESETS) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "preset";
    btn.innerHTML = `<b>${esc(p.title)}</b><small>${esc(p.hint || "")}</small>`;
    btn.addEventListener("click", () => {
      const input = $("#chat-input");
      input.value = p.text;
      autoGrow(input);
      input.focus();
      goToPanel("playground");
    });
    host.appendChild(btn);
  }
}

/* ---------- navigation ---------- */
function goToPanel(panel) {
  $$("#rail-nav .rail-btn").forEach((b) => b.classList.toggle("active", b.dataset.panel === panel));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === `panel-${panel}`));
  $("#crumb-active").textContent = PANEL_TITLES[panel] || panel;
}
$$("#rail-nav .rail-btn").forEach((btn) => {
  btn.addEventListener("click", () => goToPanel(btn.dataset.panel));
});

/* ---------- wiring ---------- */
$("#pg-temp").addEventListener("input", (e) => {
  $("#pg-temp-val").textContent = Number(e.target.value).toFixed(1);
});
$("#chat-input").addEventListener("input", (e) => autoGrow(e.target));
$("#chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("#chat-form").requestSubmit();
  }
});
$("#chat-form").addEventListener("submit", sendChat);
$("#chat-clear").addEventListener("click", () => {
  $("#chat-log").innerHTML =
    '<div class="empty-state" id="chat-empty"><div class="empty-glyph">▶</div><p>Start a conversation with a registered model.</p></div>';
  $("#chat-usage").textContent = "";
  state.hasChat = false;
});

/* ---------- boot ---------- */
initTheme();
renderPresets();
checkHealth();
loadModels();
loadSuites();
