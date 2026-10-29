"use strict";
// Everything below builds the page with textContent and createElement. Model output is
// untrusted text: it is never turned into markup, script code or a link target.

const $ = (selector) => document.querySelector(selector);

function el(tag, attrs, ...kids) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (key === "class") node.className = value;
    else node.setAttribute(key, value);
  }
  for (const kid of kids) {
    if (kid === null || kid === undefined || kid === false) continue;
    node.append(typeof kid === "object" ? kid : document.createTextNode(String(kid)));
  }
  return node;
}

async function api(path) {
  const resp = await fetch(path, { credentials: "same-origin", cache: "no-store" });
  if (!resp.ok) throw new Error(path + " -> " + resp.status);
  return resp.json();
}

let skew = 0;                       // server clock minus ours, in seconds
const serverNow = () => Date.now() / 1000 + skew;
const secs = (n) => (n == null ? "-" : n < 100 ? n.toFixed(1) + "s" : Math.round(n) + "s");
function ago(ts) {
  const d = Math.max(0, serverNow() - ts);
  if (d < 90) return Math.round(d) + "s ago";
  if (d < 5400) return Math.round(d / 60) + "m ago";
  return Math.round(d / 3600) + "h ago";
}

// ---------------------------------------------------------------- runs list
let selected = null;
let runsById = {};

function badge(run) {
  const good = run.status === "finished" && run.result === "approved";
  const cls = "badge " + run.status + (run.status === "finished" ? (good ? " good" : " bad") : "");
  const text = run.status === "finished" ? (run.result || "finished") : run.status;
  return el("span", { class: cls + (run.status === "running" ? " pulse" : "") }, text);
}

async function refreshRuns() {
  const data = await api("/api/runs");
  skew = data.now - Date.now() / 1000;
  runsById = Object.fromEntries(data.runs.map((r) => [r.id, r]));
  const list = $("#run-list");
  list.replaceChildren(...data.runs.map((run) => {
    const item = el("div", { class: "item" + (run.id === selected ? " sel" : ""), tabindex: "0" },
      el("div", { class: "title" }, badge(run), run.id),
      el("div", { class: "sub" }, (run.task || "(no task)").slice(0, 120)),
      el("div", { class: "sub" }, run.kind + " · " + run.calls + " model calls · " + ago(run.updated)));
    item.addEventListener("click", () => selectRun(run.id));
    item.addEventListener("keydown", (e) => { if (e.key === "Enter") selectRun(run.id); });
    return item;
  }));
  if (!data.runs.length) list.replaceChildren(el("p", { class: "muted pad" },
    "No traced runs yet. Runs started with this version write runs/<thread>/trace.jsonl."));
}

// ---------------------------------------------------------------- one run
let events = [], nextIndex = 0, pending = {}, follow = true, view = null;

function selectRun(id) {
  selected = id; events = []; nextIndex = 0; pending = {};
  view = null;
  const section = $("#run-view");
  const timeline = el("div", { id: "timeline" });
  const banner = el("div", { id: "banner", class: "banner", hidden: "" });
  const stats = el("div", { id: "stats", class: "stats" });
  const followBox = el("input", { type: "checkbox", id: "follow", checked: "" });
  followBox.addEventListener("change", () => { follow = followBox.checked; });
  section.replaceChildren(el("h2", {}, id), stats, banner,
    el("label", { class: "row muted" }, followBox, "follow new events"), timeline);
  view = { timeline, banner, stats };
  refreshRuns();
  pollTrace();
}

function jsonBlock(label, text) {
  return [el("div", { class: "label" }, label), el("pre", {}, text)];
}

function card(kind, title, meta, bodyKids, open) {
  const hasBody = bodyKids && bodyKids.length;
  if (!hasBody) return el("div", { class: "event " + kind }, el("div", { class: "head" }, el("b", {}, title), el("span", { class: "meta" }, meta || "")));
  const d = el("details", { class: "event " + kind },
    el("summary", {}, el("b", {}, title), el("span", { class: "meta" }, meta || "")),
    el("div", { class: "body" }, ...bodyKids));
  if (open) d.setAttribute("open", "");
  return d;
}

function setMeta(node, text) {
  const meta = node.querySelector(".meta");
  if (meta) meta.textContent = text;
}

function render(e) {
  const t = view.timeline;
  switch (e.event) {
    case "run_start":
      t.append(card("", "Run started · " + e.kind, "",
        [...jsonBlock("task", e.task), ...jsonBlock("models", JSON.stringify(e.models, null, 2))], true));
      break;
    case "resumed":
      t.append(card("pause", "Resumed", "", []));
      break;
    case "llm_start": {
      const node = card("llm", e.role + " · " + e.model, "thinking...", jsonBlock("prompt", e.prompt));
      node.dataset.ts = String(e.ts);
      (pending[e.role] = pending[e.role] || []).push(node);
      t.append(node);
      break;
    }
    case "llm_end":
    case "llm_error": {
      const node = (pending[e.role] || []).shift();
      if (!node) break;
      delete node.dataset.ts;
      const body = node.querySelector(".body");
      if (e.event === "llm_end") {
        const tok = e.tokens ? e.tokens.total_tokens : 0;
        setMeta(node, secs(e.seconds) + " · " + tok + " tokens");
        body.append(...jsonBlock("reply", e.reply));
      } else {
        node.classList.add("bad");
        setMeta(node, "failed after " + secs(e.seconds));
        body.append(...jsonBlock("error", e.error));
      }
      break;
    }
    case "node": renderNode(e); break;
    case "pause":
      t.append(card("pause", "Paused for the owner (#" + e.number + ")", "",
        Object.entries(e.info || {}).flatMap(([k, v]) => jsonBlock(k, typeof v === "string" ? v : JSON.stringify(v))), true));
      break;
    case "owner_call":
      t.append(card("pause", "Owner call", "", jsonBlock("reply", e.text), true));
      break;
    case "run_end": {
      const good = e.result === "approved";
      const kids = [el("div", {}, "result: " + e.result + " · rejections " + e.rejections + " · pauses " + e.pauses)];
      if (e.changed && e.changed.length) kids.push(...jsonBlock("changed", e.changed.join("\n")));
      const n = card(good ? "good" : "bad", "Run finished · " + e.result, "", kids, true);
      if (e.patch) {
        const btn = el("button", { class: "act", type: "button" }, "show patch");
        btn.addEventListener("click", async () => {
          const data = await api("/api/patch?id=" + encodeURIComponent(selected));
          btn.replaceWith(...jsonBlock("changes.patch", data.patch || "(not found)"));
        });
        n.querySelector(".body").append(btn);
      }
      t.append(n);
      break;
    }
    case "run_error":
      t.append(card("bad", "Run crashed", "", jsonBlock("error", e.error), true));
      break;
    default:
      break;
  }
}

function renderNode(e) {
  const u = e.update || {};
  const kids = Object.entries(u).flatMap(([k, v]) =>
    jsonBlock(k, typeof v === "string" ? v : JSON.stringify(v, null, 2)));
  let kind = "";
  if (e.node === "check" || e.node === "reviewer") kind = u.tests_passed === true || u.approved === true ? "good"
    : u.tests_passed === false || u.approved === false ? "bad" : "";
  const title = e.node + (e.node === "check" ? (u.tests_passed ? " · tests passed" : " · tests failed") : "")
    + (e.node === "reviewer" ? (u.approved ? " · approved" : " · rejected") : "");
  const open = e.node === "check" || e.node === "apply" && (u.apply_errors || []).length > 0;
  view.timeline.append(card(kind, title, secs(e.seconds), kids, open));
}

function updateHeader() {
  const done = events.filter((e) => e.event === "llm_end");
  const tokens = done.reduce((n, e) => n + ((e.tokens || {}).total_tokens || 0), 0);
  const laps = done.filter((e) => e.role === "worker" || e.role === "escalation").length;
  const reviews = events.filter((e) => e.event === "node" && e.node === "reviewer");
  const rej = reviews.length ? (reviews[reviews.length - 1].update || {}).rejections : 0;
  const run = runsById[selected] || {};
  const stat = (value, name) => el("div", { class: "stat" }, el("b", {}, String(value)), el("span", {}, name));
  view.stats.replaceChildren(stat(run.status || "-", "status"), stat(laps, "worker laps"),
    stat(rej == null ? 0 : rej, "rejections"), stat(done.length, "model calls"), stat(tokens, "tokens"));
  const live = Object.entries(pending).flatMap(([role, nodes]) => nodes.map((n) => [role, n]));
  if (live.length && run.status === "running") {
    const [role, node] = live[live.length - 1];
    view.banner.hidden = false;
    view.banner.textContent = "Waiting on the " + role + " model · " + Math.round(serverNow() - Number(node.dataset.ts)) + "s so far";
  } else {
    view.banner.hidden = true;
  }
  for (const nodes of Object.values(pending)) for (const n of nodes) {
    if (n.dataset.ts) setMeta(n, "thinking... " + Math.round(serverNow() - Number(n.dataset.ts)) + "s");
  }
}

let tracePolling = false;
async function pollTrace() {
  if (tracePolling || !selected) return;
  tracePolling = true;
  const mine = selected;
  try {
    for (;;) {
      const data = await api("/api/trace?id=" + encodeURIComponent(mine) + "&after=" + nextIndex);
      if (mine !== selected) return;
      const stickToBottom = follow;
      for (const e of data.events) { events.push(e); render(e); }
      nextIndex = data.next;
      if (data.events.length && stickToBottom) window.scrollTo(0, document.body.scrollHeight);
      if (nextIndex >= data.total) break;
    }
    updateHeader();
  } catch (err) {
    if (view) view.banner.hidden = false, view.banner.textContent = "Could not read the trace: " + err.message;
  } finally {
    tracePolling = false;
  }
}

// ---------------------------------------------------------------- evals
let ticked = [];

async function refreshEvals() {
  const data = await api("/api/evals");
  const list = $("#eval-list");
  list.replaceChildren(...data.reports.map((r) => {
    const box = el("input", { type: "checkbox", "aria-label": "compare " + r.name });
    box.checked = ticked.includes(r.name);
    box.addEventListener("change", () => {
      ticked = ticked.filter((n) => n !== r.name);
      if (box.checked) ticked.push(r.name);
      ticked = ticked.slice(-2);
      refreshEvals();
      renderCompareButton();
    });
    const item = el("div", { class: "item" },
      el("div", { class: "row" }, box, el("div", { class: "title" }, r.label || r.name)),
      el("div", { class: "sub" }, r.name),
      el("div", { class: "sub" }, (r.finished ? "finished" : "running or stopped") + " · " + r.done + " attempts done · " + r.started));
    item.addEventListener("click", (e) => { if (e.target !== box) showEval(r.name); });
    return item;
  }));
  if (!data.reports.length) list.replaceChildren(el("p", { class: "muted pad" }, "No eval reports yet."));
}

function renderCompareButton() {
  const view = $("#eval-view");
  let holder = $("#compare-bar");
  if (!holder) { holder = el("div", { id: "compare-bar", class: "row" }); view.prepend(holder); }
  holder.replaceChildren();
  if (ticked.length === 2) {
    const btn = el("button", { class: "act", type: "button" }, "compare " + ticked[0] + " → " + ticked[1]);
    btn.addEventListener("click", async () => {
      const data = await api("/api/compare?a=" + encodeURIComponent(ticked[0]) + "&b=" + encodeURIComponent(ticked[1]));
      const out = $("#compare-out") || el("pre", { id: "compare-out" });
      out.textContent = data.text;
      holder.after(out);
    });
    holder.append(btn);
  }
}

async function showEval(name) {
  const data = await api("/api/eval?name=" + encodeURIComponent(name));
  const head = ["task", "pass", "rate", "laps to pass", "escalated", "errors", "tokens", "secs"];
  const fmt = (n, d) => (n == null ? "-" : n.toFixed(d));
  const rows = data.summary.map((r) => el("tr", { class: r.task === "ALL" ? "all" : "" },
    el("td", {}, r.task), el("td", {}, r.passes + "/" + r.n), el("td", {}, Math.round(r.rate * 100) + "%"),
    el("td", {}, fmt(r.mean_laps_to_pass, 1)), el("td", {}, String(r.escalated)), el("td", {}, String(r.errors)),
    el("td", {}, fmt(r.mean_tokens, 0)), el("td", {}, fmt(r.mean_seconds, 1))));
  const attempts = data.report.results.map((a) => el("tr", {},
    el("td", {}, a.task + " #" + a.attempt), el("td", {}, a.passed ? "pass" : "fail"), el("td", {}, a.result),
    el("td", {}, String(a.laps)), el("td", {}, a.escalated ? "yes" : "no"), el("td", {}, a.error || ""),
    el("td", {}, String(a.total_tokens)), el("td", {}, String(a.seconds))));
  const section = $("#eval-view");
  section.replaceChildren(el("h2", {}, data.report.label || name),
    el("p", { class: "muted" }, "models: " + JSON.stringify(data.report.config.models)
      + " · budget " + (data.report.budget || "none") + " · " + (data.report.finished ? "finished" : "not finished")),
    el("table", {}, el("thead", {}, el("tr", {}, ...head.map((h) => el("th", {}, h)))), el("tbody", {}, ...rows)),
    el("h3", {}, "every attempt"),
    el("table", {}, el("thead", {}, el("tr", {}, ...["attempt", "outcome", "result", "laps", "escalated", "error", "tokens", "secs"].map((h) => el("th", {}, h)))),
      el("tbody", {}, ...attempts)));
  renderCompareButton();
}

// ---------------------------------------------------------------- tabs and timers
function showTab(name) {
  $("#view-runs").hidden = name !== "runs";
  $("#view-evals").hidden = name !== "evals";
  $("#tab-runs").classList.toggle("on", name === "runs");
  $("#tab-evals").classList.toggle("on", name === "evals");
  if (name === "evals") { refreshEvals(); renderCompareButton(); }
}
$("#tab-runs").addEventListener("click", () => showTab("runs"));
$("#tab-evals").addEventListener("click", () => showTab("evals"));

refreshRuns().catch(() => {});
setInterval(() => { if (!$("#view-runs").hidden) refreshRuns().catch(() => {}); }, 3000);
setInterval(() => { if (!$("#view-evals").hidden) refreshEvals().catch(() => {}); }, 5000);
setInterval(() => { if (selected) pollTrace(); }, 1500);
setInterval(() => { if (selected && view) updateHeader(); }, 1000);
