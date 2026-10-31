"use strict";
// Builds the page with createElement / createElementNS and textContent only. Model output is
// untrusted text: it is never turned into markup, script code or a link target.
(function () {
  const L = window.AgentLoopLogic;
  const SVGNS = "http://www.w3.org/2000/svg";
  const $ = (selector) => document.querySelector(selector);

  function el(tag, attrs, ...kids) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (key === "class") node.className = value;
      else if (value === true) node.setAttribute(key, "");
      else if (value !== false && value !== null && value !== undefined) node.setAttribute(key, value);
    }
    for (const kid of kids.flat()) {
      if (kid === null || kid === undefined || kid === false) continue;
      node.append(typeof kid === "object" ? kid : document.createTextNode(String(kid)));
    }
    return node;
  }
  function svg(tag, attrs, ...kids) {
    const node = document.createElementNS(SVGNS, tag);
    for (const [key, value] of Object.entries(attrs || {})) node.setAttribute(key, String(value));
    for (const kid of kids.flat()) {
      if (kid === null || kid === undefined) continue;
      node.append(typeof kid === "object" ? kid : document.createTextNode(String(kid)));
    }
    return node;
  }
  function setChildren(node, ...kids) {
    node.replaceChildren(...kids.flat().filter((k) => k !== null && k !== undefined && k !== false));
  }
  async function api(path) {
    const resp = await fetch(path, { credentials: "same-origin", cache: "no-store" });
    if (!resp.ok) throw new Error(path.split("?")[0] + " -> " + resp.status);
    return resp.json();
  }

  let skew = 0; // server clock minus ours, seconds
  const serverNow = () => Date.now() / 1000 + skew;
  function ago(ts) {
    const d = Math.max(0, serverNow() - ts);
    if (d < 90) return Math.round(d) + "s ago";
    if (d < 5400) return Math.round(d / 60) + "m ago";
    return Math.round(d / 3600) + "h ago";
  }

  // ------------------------------------------------------------------ state
  const STATUSES = ["running", "paused", "finished", "error", "stopped"];
  const ui = {
    tab: "runs", runQuery: "", statuses: new Set(), selected: null,
    showEvalRuns: false, kinds: new Set(["llm", "node", "pause", "run"]), onlyFailures: false, query: "",
    sent: new Set(), confirmAccept: false, ownerSig: "", ownerError: "", guidance: "", csrf: null,
    follow: true, sub: "timeline", diffMode: "unified", open: new Set(), closed: new Set(), expandAll: false,
  };
  let runs = [], runsById = {}, events = [], nextIndex = 0, items = [], lastSig = "";
  let patchCache = {}, evalReports = [], evalTicked = [], evalShown = null;

  const roleClass = (role) => (["lead", "worker", "reviewer", "escalation"].includes(role) ? role : "worker");

  function badge(run) {
    const good = run.status === "finished" && run.result === "approved";
    const cls = "badge " + run.status + (run.status === "finished" ? (good ? " good" : " bad") : "")
      + (run.status === "running" ? " pulse" : "");
    return el("span", { class: cls }, run.status === "finished" ? run.result || "finished" : run.status);
  }

  // ------------------------------------------------------------------ run list
  function renderChips() {
    setChildren($("#status-chips"), ...STATUSES.map((s) => {
      const b = el("button", { class: "chip" + (ui.statuses.has(s) ? " on" : ""), type: "button", "aria-pressed": ui.statuses.has(s) }, s);
      b.addEventListener("click", () => { ui.statuses.has(s) ? ui.statuses.delete(s) : ui.statuses.add(s); renderChips(); renderRunList(); });
      return b;
    }), (() => {
      const b = el("button", { class: "chip" + (ui.showEvalRuns ? " on" : ""), type: "button", "aria-pressed": ui.showEvalRuns }, "eval attempts");
      b.addEventListener("click", () => { ui.showEvalRuns = !ui.showEvalRuns; renderChips(); renderRunList(); });
      return b;
    })());
  }

  const isEvalRun = (r) => r.id.startsWith("evals/");
  // evals/<label>-<stamp>/<stamp>-<task>-<n>  ->  "<label> · <task>-<n>"
  function runLabel(id) {
    const m = /^evals\/(.+?)-\d{8}-\d{6}-[0-9a-f]{4}\/\d{8}-\d{6}-[0-9a-f]{4}-(.+)$/.exec(id);
    return m ? m[1] + " · " + m[2] : id;
  }
  function visibleRuns() {
    return L.sortRuns(runs).filter((r) => (ui.showEvalRuns || !isEvalRun(r))
      && L.matchRun(r, { query: ui.runQuery, statuses: ui.statuses.size ? ui.statuses : null }));
  }

  function renderRunList() {
    const list = $("#run-list");
    const shown = visibleRuns();
    if (!shown.length) {
      setChildren(list, el("p", { class: "muted pad" }, runs.length
        ? "No runs match."
        : "No traced runs yet. Runs started with this version write runs/<thread>/trace.jsonl."));
      return;
    }
    setChildren(list, ...shown.map((run) => {
      const item = el("div", { class: "item" + (run.id === ui.selected ? " sel" : ""), tabindex: "0", role: "button", "data-id": run.id },
        el("div", { class: "title" }, badge(run), runLabel(run.id)),
        el("div", { class: "sub" }, (run.task || "(no task)").slice(0, 110)),
        el("div", { class: "sub" }, [run.kind, run.calls + " calls", ago(run.updated)].join(" · ")));
      item.addEventListener("click", () => openRun(run.id));
      item.addEventListener("keydown", (e) => { if (e.key === "Enter") openRun(run.id); });
      return item;
    }));
  }

  async function refreshRuns() {
    const data = await api("/api/runs");
    skew = data.now - Date.now() / 1000;
    runs = data.runs;
    runsById = Object.fromEntries(runs.map((r) => [r.id, r]));
    renderRunList();
    if (ui.selected) renderHead();
  }

  // ------------------------------------------------------------------ one run
  function hashFor(id, sub) {
    return "run=" + encodeURIComponent(id) + (sub && sub !== "timeline" ? "&view=" + sub : "");
  }
  function openRun(id) {
    if (window.location.hash !== "#" + hashFor(id, ui.sub)) window.location.hash = hashFor(id, ui.sub);
    else selectRun(id);
  }

  function selectRun(id) {
    ui.selected = id; events = []; nextIndex = 0; items = []; lastSig = ""; ui.open = new Set(); ui.closed = new Set();
    ui.confirmAccept = false; ui.ownerSig = ""; ui.ownerError = ""; ui.guidance = "";
    setChildren($("#run-view"), el("div", { id: "run-head", class: "head" }), el("div", { id: "run-owner" }), el("div", { id: "run-tools" }),
      el("div", { id: "run-body" }));
    renderRunList(); renderHead(); renderTools(); pollTrace(true);
  }

  function stat(value, name, extra) {
    return el("div", { class: "stat" }, el("b", {}, String(value)), el("span", {}, name), extra || null);
  }

  function renderHead() {
    const head = $("#run-head");
    if (!head || !ui.selected) return;
    const run = runsById[ui.selected] || { id: ui.selected, status: "?", kind: "", task: "" };
    const s = L.runStats(items);
    const startItem = items.find((i) => i.kind === "start");
    const cap = (startItem && startItem.maxRejections) || 3;
    const meter = el("span", { class: "meter", "aria-label": s.rejections + " rejections" },
      Array.from({ length: cap }, (_, i) => el("i", { class: i < s.rejections ? "on" : "" })));
    const taskBox = el("div", { class: "task" + (ui.taskOpen ? " open" : ""), title: "click to expand" }, (startItem && startItem.task) || run.task || "");
    taskBox.addEventListener("click", () => { ui.taskOpen = !ui.taskOpen; taskBox.classList.toggle("open", ui.taskOpen); });
    const models = startItem ? Object.entries(startItem.models || {}).map(([role, m]) =>
      el("span", { class: "model" }, el("b", {}, role), String(m).replace(/^ollama:/, ""))) : [];
    const live = items.filter((i) => i.kind === "llm" && !i.done);
    const banner = live.length && run.status === "running"
      ? el("div", { class: "banner", id: "live-banner" }, "Waiting on the ", el("b", {}, live[live.length - 1].role), " model · ",
        el("span", { id: "live-secs" }, Math.round(serverNow() - live[live.length - 1].ts) + "s")) : null;
    setChildren(head, 
      el("div", { class: "head-top" }, el("h2", {}, runLabel(ui.selected)), badge(run),
        el("span", { class: "muted" }, run.kind || ""),
        exportButton()),
      el("div", { class: "models" }, ...models), taskBox, banner,
      el("div", { class: "stats" },
        stat(s.laps, "worker laps"), stat(s.rejections, "rejections", meter), stat(s.calls, "model calls"),
        stat(L.formatCount(s.tokens.total), "tokens", el("span", { class: "muted small" }, L.formatCount(s.tokens.input) + " in · " + L.formatCount(s.tokens.output) + " out")),
        stat(L.formatSeconds(s.wallSeconds), "wall time"),
        stat(s.checks.passed + "/" + (s.checks.passed + s.checks.failed), "checks passed")));
    renderOwner();
  }

  // ---- answering a paused run (the only thing this page can write)
  const STOP_WORDS = ["stop", "quit", "end", "halt", "abort"], ACCEPT_WORDS = ["accept", "approve", "ship"];

  // The pause the run is waiting on: the latest one that has a nonce and nothing after it answering it.
  function pendingPause() {
    for (let i = items.length - 1; i >= 0; i--) {
      const it = items[i];
      if (it.kind === "owner" || it.kind === "end" || it.kind === "crash") return null;
      if (it.kind === "pause") return it.nonce ? it : null;
    }
    return null;
  }

  async function sendOwnerReply(nonce, text) {
    ui.sent.add(nonce); ui.confirmAccept = false; ui.ownerError = "";
    renderOwner(true);
    try {
      if (!ui.csrf) ui.csrf = (await api("/api/session")).csrf;
      const resp = await fetch("/api/owner_reply", {
        method: "POST", credentials: "same-origin",
        headers: { "Content-Type": "application/json", "X-CSRF": ui.csrf },
        body: JSON.stringify({ id: ui.selected, nonce, text }),
      });
      if (!resp.ok) {
        let why = "HTTP " + resp.status;
        try { why = (await resp.json()).error || why; } catch (_) { /* keep the status */ }
        throw new Error(why);
      }
    } catch (err) {
      ui.sent.delete(nonce); ui.ownerError = "Could not send: " + err.message;
      renderOwner(true);
    }
  }

  function renderOwner(force) {
    const box = $("#run-owner");
    if (!box) return;
    const waiting = pendingPause();
    const sig = waiting ? [waiting.nonce, ui.sent.has(waiting.nonce), ui.confirmAccept, ui.ownerError].join("|") : "";
    if (!force && sig === ui.ownerSig) return;
    ui.ownerSig = sig;
    if (!waiting) { setChildren(box); return; }

    const info = waiting.info || {};
    const lastCheck = items.slice().reverse().find((i) => i.kind === "node" && i.node === "check");
    const testsPassed = !!(lastCheck && lastCheck.update && lastCheck.update.tests_passed === true);
    const nonce = waiting.nonce;
    let controls;
    if (ui.sent.has(nonce)) {
      controls = el("p", { class: "muted" }, "Sent. Waiting for the run to pick it up…");
    } else {
      const stop = el("button", { class: "btn", type: "button" }, "Stop run");
      stop.addEventListener("click", () => sendOwnerReply(nonce, "stop"));
      let accept;
      if (!ui.confirmAccept) {
        accept = el("button", { class: "btn danger", type: "button" }, "Accept anyway…");
        accept.addEventListener("click", () => { ui.confirmAccept = true; renderOwner(true); });
      } else {
        const yes = el("button", { class: "btn danger", type: "button" }, testsPassed ? "Yes, accept" : "Yes, accept although the tests failed");
        yes.addEventListener("click", () => sendOwnerReply(nonce, "accept"));
        const no = el("button", { class: "btn", type: "button" }, "Cancel");
        no.addEventListener("click", () => { ui.confirmAccept = false; renderOwner(true); });
        accept = el("span", { class: "row" }, yes, no);
      }
      const guidance = el("textarea", { rows: "3", maxlength: "2000", "aria-label": "guidance for the worker",
        placeholder: "Or tell the worker what to try next…" });
      guidance.value = ui.guidance;
      guidance.addEventListener("input", () => { ui.guidance = guidance.value; });
      const send = el("button", { class: "btn primary", type: "button" }, "Send guidance");
      send.addEventListener("click", () => {
        const text = guidance.value.trim();
        const norm = text.toLowerCase().replace(/^[ .!?]+|[ .!?]+$/g, "");
        if (!text) { ui.ownerError = "Write something first."; renderOwner(true); return; }
        if (STOP_WORDS.includes(norm) || ACCEPT_WORDS.includes(norm)) {
          ui.ownerError = "That word ends the run. Use the Stop or Accept buttons, or add more words.";
          renderOwner(true); return;
        }
        ui.guidance = "";
        sendOwnerReply(nonce, text);
      });
      controls = el("div", {}, el("div", { class: "row" }, stop, accept), el("div", { class: "label" }, "guidance"), guidance,
        el("div", { class: "row pad-top" }, send), ui.ownerError ? el("p", { class: "error-text" }, ui.ownerError) : null);
    }
    setChildren(box, el("div", { class: "card owner" },
      el("h3", {}, "This run is waiting for you"),
      el("p", { class: "muted" }, "Rejected " + (info.rejections == null ? "?" : info.rejections) + " times. "
        + (testsPassed ? "The last tests passed." : "The last tests failed.")),
      info.review ? block("last review", String(info.review)) : null,
      info.test_output ? block("last test output", String(info.test_output)) : null,
      controls));
  }

  function exportButton() {
    const b = el("button", { class: "btn", type: "button" }, "Export report");
    b.addEventListener("click", () => window.location.assign("/api/export?id=" + encodeURIComponent(ui.selected)));
    return b;
  }

  function renderTools() {
    const tools = $("#run-tools");
    if (!tools) return;
    const kindBox = (kind, label) => {
      const box = el("input", { type: "checkbox", checked: ui.kinds.has(kind) });
      box.addEventListener("change", () => { box.checked ? ui.kinds.add(kind) : ui.kinds.delete(kind); renderBody(true); });
      return el("label", {}, box, label);
    };
    const fail = el("input", { type: "checkbox", checked: ui.onlyFailures });
    fail.addEventListener("change", () => { ui.onlyFailures = fail.checked; renderBody(true); });
    const follow = el("input", { type: "checkbox", checked: ui.follow, id: "follow-box" });
    follow.addEventListener("change", () => { ui.follow = follow.checked; });
    const search = el("input", { type: "search", placeholder: "Search in this run", value: ui.query, "aria-label": "search in this run", autocomplete: "off", id: "event-search" });
    search.addEventListener("input", () => { ui.query = search.value; renderBody(true); });
    const expand = el("button", { class: "btn-ghost", type: "button" }, ui.expandAll ? "Collapse all" : "Expand all");
    expand.addEventListener("click", () => toggleExpand());
    const sub = (name, label) => {
      const b = el("button", { class: "subtab" + (ui.sub === name ? " on" : ""), type: "button" }, label);
      b.addEventListener("click", () => setSub(name));
      return b;
    };
    setChildren(tools, 
      el("div", { class: "subtabs" }, sub("timeline", "Timeline"), sub("charts", "Charts"), sub("patch", "Patch")),
      ui.sub === "timeline" ? el("div", { class: "tools" }, kindBox("llm", "model calls"), kindBox("node", "node results"),
        kindBox("pause", "pauses"), kindBox("run", "start / end"), el("label", {}, fail, "only failures"),
        search, expand, el("label", {}, follow, "follow")) : null);
  }

  function setSub(name) {
    ui.sub = name; lastSig = "";
    window.history.replaceState(null, "", "#" + hashFor(ui.selected, name));
    renderTools(); renderBody(true);
  }
  function toggleExpand() { ui.expandAll = !ui.expandAll; ui.open = new Set(); ui.closed = new Set(); renderTools(); renderBody(true); }

  // ---- timeline cards
  function copyButton(text) {
    const b = el("button", { type: "button" }, "copy");
    b.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(text); b.textContent = "copied"; } catch (_) { b.textContent = "copy failed"; }
      setTimeout(() => { b.textContent = "copy"; }, 1200);
    });
    return b;
  }
  const block = (label, text) => [el("div", { class: "label" }, label, copyButton(text)), el("pre", {}, text)];

  function card(item, kind, title, meta, kids, openByDefault) {
    const d = el("details", { class: "event " + kind, "data-key": item.key },
      el("summary", {}, el("b", {}, title), el("span", { class: "meta" }, meta || "")),
      kids.length ? el("div", { class: "body" }, ...kids) : null);
    const wanted = ui.expandAll || ui.open.has(item.key) || (openByDefault && !ui.closed.has(item.key));
    if (wanted) d.setAttribute("open", "");
    d.addEventListener("toggle", () => {
      if (d.open) { ui.open.add(item.key); ui.closed.delete(item.key); }
      else { ui.open.delete(item.key); ui.closed.add(item.key); }
    });
    return d;
  }

  function renderItem(it) {
    switch (it.kind) {
      case "start":
        return card(it, "", "Run started · " + it.runKind, "", [...block("task", it.task), ...block("models", JSON.stringify(it.models, null, 2))], false);
      case "resumed": return card(it, "pause", "Resumed", "", []);
      case "llm": {
        const meta = (it.retries ? "retried after a server error · " : "") + (it.done ? (it.error ? "failed after " + L.formatSeconds(it.seconds)
          : L.formatSeconds(it.seconds) + " · " + L.formatCount((it.tokens || {}).total_tokens || 0) + " tokens")
          : "thinking… " + Math.max(0, Math.round(serverNow() - it.ts)) + "s");
        const kids = [...block("prompt", it.prompt)];
        if (it.reply !== null) kids.push(...block("reply", it.reply));
        if (it.error) kids.push(...block("error", it.error));
        const d = card(it, it.error ? "bad" : "role-" + roleClass(it.role), it.role + " · " + it.model.replace(/^ollama:/, ""), meta, kids, false);
        if (!it.done) { d.dataset.ts = String(it.ts); d.dataset.live = "1"; }
        return d;
      }
      case "node": {
        const u = it.update || {};
        const out = L.outcome(it);
        let title = it.node;
        if (it.node === "apply") title += Array.isArray(u.apply_errors) && u.apply_errors.length
          ? " · " + u.apply_errors.length + " block(s) rejected" : " · " + ((u.changed || []).length) + " file(s) written";
        if (it.node === "check") title += u.tests_passed ? " · tests passed" : " · tests failed";
        if (it.node === "reviewer") title += u.approved ? " · approved" : " · rejected";
        const kids = Object.entries(u).flatMap(([k, v]) => block(k, typeof v === "string" ? v : JSON.stringify(v, null, 2)));
        return card(it, out, title, L.formatSeconds(it.seconds), kids, it.node === "check" || out === "bad");
      }
      case "pause":
        return card(it, "pause", "Paused for the owner (#" + it.number + ")", "",
          Object.entries(it.info || {}).filter(([k]) => k !== "options").flatMap(([k, v]) => block(k, typeof v === "string" ? v : JSON.stringify(v))), true);
      case "owner":
        return card(it, "pause", L.replyTitle(it.source), "", block("reply", it.text), true);
      case "end": {
        const kids = [el("p", {}, "result: " + it.result + " · rejections " + it.rejections + " · pauses " + it.pauses)];
        if (it.changed.length) kids.push(...block("changed files", it.changed.join("\n")));
        if (it.patch) {
          const b = el("button", { class: "btn", type: "button" }, "View patch");
          b.addEventListener("click", () => setSub("patch"));
          kids.push(b);
        }
        return card(it, L.outcome(it), "Run finished · " + it.result, "", kids, true);
      }
      case "crash": return card(it, "bad", "Run crashed", "", block("error", it.error), true);
      default: return card(it, "", it.kind, "", [], false);
    }
  }

  // ---- charts
  function barChart(title, entries, fmt, classFor) {
    const g = L.barGeometry(entries, { width: 360, height: 170, left: 46, bottom: 34 });
    const root = svg("svg", { viewBox: "0 0 360 170", role: "img", "aria-label": title });
    for (const t of g.ticks) {
      root.append(svg("line", { class: "axis", x1: 46, x2: 352, y1: t.y, y2: t.y }),
        svg("text", { x: 40, y: t.y + 4, "text-anchor": "end" }, fmt(t.value)));
    }
    for (const b of g.bars) {
      root.append(svg("rect", { class: "bar " + (classFor ? classFor(b.label) : ""), x: b.x, y: b.y, width: b.w, height: Math.max(0, b.h), rx: 3 }),
        svg("text", { x: b.x + b.w / 2, y: 160, "text-anchor": "middle" }, b.label.length > 11 ? b.label.slice(0, 10) + "…" : b.label));
    }
    const table = el("table", {}, el("tbody", {}, ...entries.map((e) => el("tr", {}, el("td", {}, e.label), el("td", {}, fmt(e.value))))));
    return el("div", { class: "card" }, el("h3", {}, title), root,
      el("details", { class: "data" }, el("summary", {}, "data"), el("div", { class: "scroll" }, table)));
  }

  function renderCharts(body) {
    const s = L.runStats(items);
    const roles = Object.keys(s.byRole);
    if (!roles.length) { setChildren(body, el("p", { class: "muted pad" }, "No finished model calls yet.")); return; }
    const timeline = L.timelineSpans(items, { width: 700, now: serverNow() });
    const strip = svg("svg", { viewBox: "0 0 700 70", role: "img", "aria-label": "model calls over time" });
    strip.append(svg("line", { class: "axis", x1: 0, x2: 700, y1: 52, y2: 52 }));
    for (const sp of timeline.spans) {
      strip.append(svg("rect", { class: "bar " + roleClass(sp.role), x: sp.x, y: 10, width: sp.w, height: 34, rx: 2, opacity: sp.running ? 0.55 : 1 },
        svg("title", {}, sp.role + " · " + L.formatSeconds(sp.seconds))));
    }
    strip.append(svg("text", { x: 0, y: 66 }, "start"), svg("text", { x: 700, y: 66, "text-anchor": "end" }, L.formatSeconds(timeline.t1 - timeline.t0)));
    const legend = el("div", { class: "legend" }, ["lead", "worker", "reviewer", "escalation"].map((r) =>
      el("span", {}, el("i", { class: "bar " + r }), r)));
    setChildren(body, el("div", { class: "grid" },
      el("div", { class: "card wide" }, el("h3", {}, "Model calls over time"), strip, legend),
      barChart("Seconds by role", roles.map((r) => ({ label: r, value: s.byRole[r].seconds })), (v) => L.formatSeconds(v), roleClass),
      barChart("Tokens by role", roles.map((r) => ({ label: r, value: s.byRole[r].tokens })), (v) => L.formatCount(v), roleClass),
      barChart("Seconds by node", Object.entries(s.nodeSeconds).map(([n, v]) => ({ label: n, value: v })), (v) => L.formatSeconds(v))));
  }

  // ---- patch
  async function renderPatch(body) {
    const id = ui.selected;
    setChildren(body, el("p", { class: "muted pad" }, "Loading patch…"));
    let text = patchCache[id];
    if (text === undefined) {
      try { text = (await api("/api/patch?id=" + encodeURIComponent(id))).patch; } catch (_) { text = null; }
      if (text !== null) patchCache[id] = text;
    }
    if (id !== ui.selected || ui.sub !== "patch") return;
    if (!text) { setChildren(body, el("p", { class: "muted pad" }, "No patch for this run. Only project tasks that finished write one.")); return; }
    const parsed = L.parseDiff(text);
    const mode = el("div", { class: "row" }, ["unified", "side-by-side"].map((m) => {
      const b = el("button", { class: "chip" + (ui.diffMode === m ? " on" : ""), type: "button" }, m);
      b.addEventListener("click", () => { ui.diffMode = m; renderPatch(body); });
      return b;
    }));
    const files = parsed.files.map((f) => el("div", { class: "diff-file" },
      el("div", { class: "file-head" }, el("b", {}, f.path), el("span", { class: "badge" }, f.status),
        el("span", { class: "add" }, "+" + f.additions), el("span", { class: "del" }, "−" + f.deletions)),
      ...f.hunks.map((h) => el("div", { class: "hunk" }, el("div", { class: "hunk-head" }, h.header),
        ui.diffMode === "unified" ? h.lines.map((l) => el("div", { class: "dl" },
          el("span", { class: "no" }, l.oldNo ?? ""), el("span", { class: "no" }, l.newNo ?? ""),
          el("span", { class: l.type === "add" || l.type === "del" ? l.type : l.type === "meta" ? "meta" : "" },
            (l.type === "add" ? "+" : l.type === "del" ? "-" : " ") + l.text)))
          : L.sideBySide(h).map((r) => r.meta ? el("div", { class: "dl" }, el("span", {}), el("span", {}), el("span", { class: "meta" }, r.meta.text))
            : el("div", { class: "dl sbs" },
              el("span", { class: "no" }, r.left ? r.left.oldNo ?? "" : ""), el("span", { class: r.left ? (r.left.type === "del" ? "del" : "") : "blank" }, r.left ? r.left.text : ""),
              el("span", { class: "no" }, r.right ? r.right.newNo ?? "" : ""), el("span", { class: r.right ? (r.right.type === "add" ? "add" : "") : "blank" }, r.right ? r.right.text : "")))))));
    setChildren(body, el("div", { class: "tools" }, mode,
      el("span", { class: "muted" }, parsed.files.length + " file(s)" + (parsed.truncated ? " · truncated" : ""))), ...files);
  }

  function renderBody(force) {
    const body = $("#run-body");
    if (!body) return;
    const sig = [ui.sub, items.length, items.filter((i) => i.kind === "llm" && i.done).length, [...ui.kinds].join(), ui.onlyFailures, ui.query, ui.expandAll, ui.diffMode].join("|");
    if (!force && sig === lastSig) return;
    lastSig = sig;
    if (ui.sub === "charts") return renderCharts(body);
    if (ui.sub === "patch") return renderPatch(body);
    const quiet = (it) => it.kind === "node" && ["lead", "worker", "escalate"].includes(it.node) && !Object.keys(it.update || {}).length;
    const shown = L.filterItems(items, { kinds: ui.kinds, onlyFailures: ui.onlyFailures, query: ui.query }).filter((it) => !quiet(it));
    const scroll = window.scrollY, nearBottom = window.innerHeight + scroll >= document.body.scrollHeight - 80;
    setChildren(body, ...(shown.length ? shown.map(renderItem) : [el("p", { class: "muted pad" }, items.length ? "Nothing matches these filters." : "Waiting for the first event…")]));
    if (ui.follow && nearBottom) window.scrollTo(0, document.body.scrollHeight);
  }

  let tracePolling = false;
  async function pollTrace(first) {
    if (tracePolling || !ui.selected) return;
    tracePolling = true;
    const mine = ui.selected;
    try {
      for (;;) {
        const data = await api("/api/trace?id=" + encodeURIComponent(mine) + "&after=" + nextIndex);
        if (mine !== ui.selected) return;
        events.push(...data.events); nextIndex = data.next;
        if (nextIndex >= data.total) break;
      }
      items = L.buildItems(events);
      renderHead(); renderBody(!!first);
    } catch (err) {
      const body = $("#run-body");
      if (body && !events.length) setChildren(body, el("p", { class: "error-text pad" }, "Could not read this run: " + err.message));
    } finally { tracePolling = false; }
  }

  function tickLive() {
    for (const d of document.querySelectorAll("details[data-live]")) {
      const meta = d.querySelector(".meta");
      if (meta) meta.textContent = "thinking… " + Math.max(0, Math.round(serverNow() - Number(d.dataset.ts))) + "s";
    }
    const secs = $("#live-secs"), live = items.filter((i) => i.kind === "llm" && !i.done);
    if (secs && live.length) secs.textContent = Math.round(serverNow() - live[live.length - 1].ts) + "s";
  }

  // ------------------------------------------------------------------ evals
  async function refreshEvals() {
    const data = await api("/api/evals");
    evalReports = data.reports;
    const list = $("#eval-list");
    if (!evalReports.length) { setChildren(list, el("p", { class: "muted pad" }, "No eval reports yet.")); return; }
    setChildren(list, ...evalReports.map((r) => {
      const box = el("input", { type: "checkbox", "aria-label": "compare " + r.name, checked: evalTicked.includes(r.name) });
      box.addEventListener("click", (e) => e.stopPropagation());
      box.addEventListener("change", () => {
        evalTicked = evalTicked.filter((n) => n !== r.name);
        if (box.checked) evalTicked.push(r.name);
        evalTicked = evalTicked.slice(-2);
        refreshEvals(); if (evalShown) showEval(evalShown);
      });
      const item = el("div", { class: "item" + (r.name === evalShown ? " sel" : ""), tabindex: "0", role: "button" },
        el("div", { class: "title" }, box, r.label || r.name,
          el("span", { class: "badge " + (r.finished ? (r.rate >= 0.8 ? "good" : "") : "running pulse") }, r.finished ? Math.round(r.rate * 100) + "%" : "running")),
        el("div", { class: "sub" }, r.name),
        el("div", { class: "sub" }, r.done + " attempts · " + r.started));
      item.addEventListener("click", () => { window.location.hash = "eval=" + encodeURIComponent(r.name); });
      item.addEventListener("keydown", (e) => { if (e.key === "Enter") item.click(); });
      return item;
    }));
  }

  async function showEval(name) {
    evalShown = name;
    const view = $("#eval-view");
    let data;
    try { data = await api("/api/eval?name=" + encodeURIComponent(name)); }
    catch (_) { setChildren(view, el("p", { class: "error-text pad" }, "Could not read that report.")); return; }
    if (name !== evalShown) return;
    const fmt = (n, d) => (n == null ? "-" : n.toFixed(d));
    const rows = data.summary.map((r) => el("tr", { class: r.task === "ALL" ? "all" : "" },
      el("td", {}, r.task), el("td", {}, r.passes + "/" + r.n), el("td", {}, Math.round(r.rate * 100) + "%"),
      el("td", {}, fmt(r.mean_laps_to_pass, 1)), el("td", {}, String(r.escalated)), el("td", {}, String(r.errors)),
      el("td", {}, fmt(r.mean_tokens, 0)), el("td", {}, fmt(r.mean_seconds, 1))));
    const tasks = data.summary.filter((r) => r.task !== "ALL");
    const ordered = evalReports.slice().sort((a, b) => String(a.started).localeCompare(String(b.started)));
    const trend = L.trendPoints(ordered.map((r) => ({ label: r.label || r.name, value: r.rate })), { width: 360, height: 150, left: 40, right: 12, top: 12, bottom: 24 });
    const trendSvg = svg("svg", { viewBox: "0 0 360 150", role: "img", "aria-label": "overall pass rate across reports" });
    trendSvg.append(svg("line", { class: "axis", x1: 40, x2: 348, y1: 12, y2: 12 }), svg("line", { class: "axis", x1: 40, x2: 348, y1: 126, y2: 126 }),
      svg("text", { x: 34, y: 16, "text-anchor": "end" }, "100%"), svg("text", { x: 34, y: 130, "text-anchor": "end" }, "0%"));
    if (trend.points.length) trendSvg.append(svg("path", { class: "line", d: trend.path }));
    for (const p of trend.points) trendSvg.append(svg("circle", { class: "dot", cx: p.x, cy: p.y, r: 4 }, svg("title", {}, p.label + " · " + Math.round(p.value * 100) + "%")));
    const attempts = data.report.results.map((a) => el("tr", {},
      el("td", {}, a.task + " #" + a.attempt), el("td", {}, a.passed ? "pass" : "fail"), el("td", {}, a.result),
      el("td", {}, String(a.laps)), el("td", {}, a.escalated ? "yes" : "no"), el("td", { title: a.error || "" }, a.error ? a.error.slice(0, 40) : ""),
      el("td", {}, String(a.total_tokens)), el("td", {}, String(a.seconds))));
    const head = (cols) => el("thead", {}, el("tr", {}, ...cols.map((h) => el("th", {}, h))));
    const compare = evalTicked.length === 2 ? compareBlock() : null;
    setChildren(view, 
      el("h2", {}, data.report.label || name),
      el("p", { class: "muted" }, "models " + Object.entries(data.report.config.models).map(([k, v]) => k + "=" + v.replace(/^ollama:/, "")).join(", ")
        + (Object.keys(data.report.config.model_options || {}).length
          ? " · options " + Object.entries(data.report.config.model_options).map(([r, o]) => r + " " + JSON.stringify(o)).join("; ") : "")
        + (data.report.config.override ? " · patch " + data.report.config.override : "")
        + " · budget " + (data.report.budget || "none") + " · " + (data.report.finished ? "finished" : "not finished")),
      compare,
      el("div", { class: "grid" },
        barChart("Pass rate by task (%)", tasks.map((r) => ({ label: r.task, value: r.rate * 100 })), (v) => Math.round(v) + "%", null),
        barChart("Mean seconds by task", tasks.map((r) => ({ label: r.task, value: r.mean_seconds || 0 })), (v) => L.formatSeconds(v), null),
        el("div", { class: "card" }, el("h3", {}, "Overall pass rate across reports"), trendSvg)),
      el("h3", { class: "pad" }, "By task"),
      el("div", { class: "scroll" }, el("table", {}, head(["task", "pass", "rate", "laps to pass", "escalated", "errors", "tokens", "secs"]), el("tbody", {}, ...rows))),
      el("h3", { class: "pad" }, "Every attempt"),
      el("div", { class: "scroll" }, el("table", {}, head(["attempt", "outcome", "result", "laps", "escalated", "error", "tokens", "secs"]), el("tbody", {}, ...attempts))));
  }

  function compareBlock() {
    const out = el("pre", {}, "…");
    api("/api/compare?a=" + encodeURIComponent(evalTicked[0]) + "&b=" + encodeURIComponent(evalTicked[1]))
      .then((d) => { out.textContent = d.text; }).catch(() => { out.textContent = "Could not compare."; });
    return el("div", { class: "card" }, el("h3", {}, "Comparison"), out);
  }

  // ------------------------------------------------------------------ tabs, hash, keys, timers
  function showTab(name) {
    ui.tab = name;
    $("#view-runs").hidden = name !== "runs";
    $("#view-evals").hidden = name !== "evals";
    $("#tab-runs").classList.toggle("on", name === "runs");
    $("#tab-evals").classList.toggle("on", name === "evals");
    if (name === "evals") refreshEvals().catch(() => {});
  }

  function route() {
    const params = new URLSearchParams(window.location.hash.replace(/^#/, ""));
    const run = params.get("run"), report = params.get("eval");
    if (run) {
      if (["timeline", "charts", "patch"].includes(params.get("view"))) ui.sub = params.get("view");
      showTab("runs"); selectRun(run);
    } else if (report) { showTab("evals"); refreshEvals().then(() => showEval(report)).catch(() => {}); }
    else if (params.has("evals")) showTab("evals");
  }

  $("#tab-runs").addEventListener("click", () => showTab("runs"));
  $("#tab-evals").addEventListener("click", () => showTab("evals"));
  $("#help-btn").addEventListener("click", () => { $("#help").hidden = false; });
  $("#help-close").addEventListener("click", () => { $("#help").hidden = true; });
  $("#run-search").addEventListener("input", (e) => { ui.runQuery = e.target.value; renderRunList(); });
  window.addEventListener("hashchange", route);

  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const typing = ["INPUT", "TEXTAREA", "SELECT"].includes((document.activeElement || {}).tagName);
    if (e.key === "Escape") { $("#help").hidden = true; if (typing) document.activeElement.blur(); return; }
    if (typing) return;
    if (e.key === "?") { $("#help").hidden = false; return; }
    if (e.key === "/" && ui.tab === "runs") { e.preventDefault(); $("#run-search").focus(); return; }
    if (ui.tab !== "runs") return;
    if (e.key === "j" || e.key === "k") {
      const shown = visibleRuns(), at = shown.findIndex((r) => r.id === ui.selected);
      const next = shown[Math.min(shown.length - 1, Math.max(0, at + (e.key === "j" ? 1 : -1)))];
      if (next) openRun(next.id);
    } else if (e.key === "e" && ui.selected) toggleExpand();
    else if (e.key === "f" && ui.selected) { ui.follow = !ui.follow; const box = $("#follow-box"); if (box) box.checked = ui.follow; }
    else if (ui.selected && ["1", "2", "3"].includes(e.key)) setSub(["timeline", "charts", "patch"][Number(e.key) - 1]);
  });

  renderChips();
  refreshRuns().then(route).catch(() => {});
  route();
  setInterval(() => { if (ui.tab === "runs") refreshRuns().catch(() => {}); }, 3000);
  setInterval(() => { if (ui.tab === "evals") refreshEvals().catch(() => {}); }, 5000);
  setInterval(() => { if (ui.selected && ui.tab === "runs") pollTrace(false); }, 1500);
  setInterval(tickLive, 1000);
})();
