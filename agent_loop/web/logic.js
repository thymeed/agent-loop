"use strict";
// Pure functions behind the dashboard: no DOM, no network, no globals. Tested under Node
// (jstests/logic.test.js). Everything here treats its input as untrusted text and never throws
// on malformed data; turning results into page elements is app.js's job.
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.AgentLoopLogic = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {

  const MAX_DIFF_LINES = 5000;
  const num = (v) => (typeof v === "number" && isFinite(v) ? v : 0);

  // ------------------------------------------------------------------ unified diff
  // Hunk headers say how many old/new lines follow, so body lines that merely look like
  // headers ("--- x", "+++ y", "@@ ...") are read as content, not as structure.
  function parseDiff(text) {
    const files = [];
    if (typeof text !== "string") return { files, truncated: false };
    const lines = text.split(/\r?\n/);
    let file = null, hunk = null, oldNo = 0, newNo = 0, oldLeft = 0, newLeft = 0, truncated = false;
    for (let i = 0; i < lines.length; i++) {
      if (i >= MAX_DIFF_LINES) { truncated = true; break; }
      const line = lines[i];
      if (hunk && (oldLeft > 0 || newLeft > 0)) {
        const ch = line.charAt(0);
        if (ch === "+" && newLeft > 0) {
          hunk.lines.push({ type: "add", text: line.slice(1), oldNo: null, newNo: newNo++ });
          newLeft--; file.additions++; continue;
        }
        if (ch === "-" && oldLeft > 0) {
          hunk.lines.push({ type: "del", text: line.slice(1), oldNo: oldNo++, newNo: null });
          oldLeft--; file.deletions++; continue;
        }
        if ((ch === " " || line === "") && oldLeft > 0 && newLeft > 0) {
          hunk.lines.push({ type: "ctx", text: line.slice(1), oldNo: oldNo++, newNo: newNo++ });
          oldLeft--; newLeft--; continue;
        }
        if (ch === "\\") { hunk.lines.push({ type: "meta", text: line.slice(0, 200) }); continue; }
        oldLeft = newLeft = 0; // the counts lied: stop treating lines as hunk body
      }
      if (hunk && line.charAt(0) === "\\") { hunk.lines.push({ type: "meta", text: line.slice(0, 200) }); continue; }
      let m = /^diff --git a\/(.+) b\/(.+)$/.exec(line);
      if (m) {
        file = { path: m[2], oldPath: m[1], status: "modified", additions: 0, deletions: 0, hunks: [] };
        files.push(file); hunk = null; continue;
      }
      if (!file) continue; // anything before the first file header is ignored
      if (line.startsWith("new file mode")) { file.status = "added"; continue; }
      if (line.startsWith("deleted file mode")) { file.status = "deleted"; continue; }
      m = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/.exec(line);
      if (m) {
        oldNo = +m[1]; newNo = +m[3];
        oldLeft = m[2] === undefined ? 1 : +m[2];
        newLeft = m[4] === undefined ? 1 : +m[4];
        hunk = { header: line.slice(0, 200), lines: [] };
        file.hunks.push(hunk);
      }
    }
    return { files, truncated };
  }

  // Pair removed and added lines so the two sides line up in a side-by-side view.
  function sideBySide(hunk) {
    const rows = [];
    let dels = [], adds = [];
    const flush = () => {
      const n = Math.max(dels.length, adds.length);
      for (let i = 0; i < n; i++) rows.push({ left: dels[i] || null, right: adds[i] || null });
      dels = []; adds = [];
    };
    for (const l of (hunk && hunk.lines) || []) {
      if (l.type === "del") dels.push(l);
      else if (l.type === "add") adds.push(l);
      else {
        flush();
        rows.push(l.type === "ctx" ? { left: l, right: l } : { meta: l });
      }
    }
    flush();
    return rows;
  }

  // ------------------------------------------------------------------ trace -> items
  // One item per thing worth showing: a model call (start and end merged), a node result, a
  // pause, an owner call, or the start/end of the run.
  function buildItems(events) {
    const items = [];
    const waiting = {}; // role -> calls started and not yet finished, oldest first
    for (const e of Array.isArray(events) ? events : []) {
      if (!e || typeof e !== "object") continue;
      switch (e.event) {
        case "llm_start": {
          const it = { kind: "llm", role: String(e.role), model: String(e.model || "?"), ts: num(e.ts),
            prompt: String(e.prompt || ""), reply: null, error: null, seconds: null, tokens: null, done: false };
          (waiting[it.role] = waiting[it.role] || []).push(it);
          items.push(it);
          break;
        }
        case "llm_end":
        case "llm_error": {
          const it = (waiting[String(e.role)] || []).shift();
          if (!it) break;
          it.done = true;
          it.seconds = num(e.seconds);
          if (e.event === "llm_end") { it.reply = String(e.reply || ""); it.tokens = e.tokens || {}; }
          else it.error = String(e.error || "error");
          break;
        }
        case "llm_retry": {
          const it = (waiting[String(e.role)] || [])[0];
          if (it) it.retries = (it.retries || 0) + 1;
          break;
        }
        case "node":
          items.push({ kind: "node", node: String(e.node), ts: num(e.ts), seconds: num(e.seconds), update: e.update || {} });
          break;
        case "pause":
          items.push({ kind: "pause", ts: num(e.ts), number: num(e.number), info: e.info || {}, nonce: e.nonce ? String(e.nonce) : null });
          break;
        case "owner_call":
          items.push({ kind: "owner", ts: num(e.ts), text: String(e.text || ""), source: String(e.source || "terminal") });
          break;
        case "resumed":
          items.push({ kind: "resumed", ts: num(e.ts) });
          break;
        case "run_start":
          items.push({ kind: "start", ts: num(e.ts), runKind: String(e.kind || ""), task: String(e.task || ""), models: e.models || {},
            maxRejections: num(e.max_rejections) || 3 });
          break;
        case "run_end":
          items.push({ kind: "end", ts: num(e.ts), result: String(e.result || ""), approved: !!e.approved, tests_passed: !!e.tests_passed,
            rejections: e.rejections, pauses: e.pauses, changed: e.changed || [], patch: e.patch || null, tokens: e.tokens || {} });
          break;
        case "run_error":
          items.push({ kind: "crash", ts: num(e.ts), error: String(e.error || "") });
          break;
        default:
          break;
      }
    }
    items.forEach((it, i) => { it.key = i; });
    return items;
  }

  // Title for an owner_call. A flag answer is an automatic stop at the rejection cap, not an owner's reply.
  function replyTitle(source) {
    const s = String(source || "terminal");
    return s === "flag" ? "Automatic reply · flag" : "Owner replied · " + s;
  }

  const CATEGORY = { llm: "llm", node: "node", pause: "pause", owner: "pause", resumed: "pause", start: "run", end: "run", crash: "run" };
  const category = (item) => CATEGORY[item.kind] || "run";

  // "good" / "bad" / "" for colouring and for the only-failures filter.
  function outcome(item) {
    switch (item.kind) {
      case "llm": return item.error ? "bad" : "";
      case "node": {
        const u = item.update || {};
        if (item.node === "check") return u.tests_passed === true ? "good" : u.tests_passed === false ? "bad" : "";
        if (item.node === "reviewer") return u.approved === true ? "good" : u.approved === false ? "bad" : "";
        if (item.node === "apply") return Array.isArray(u.apply_errors) && u.apply_errors.length ? "bad" : "";
        return "";
      }
      case "end": return item.approved ? "good" : "bad";
      case "crash": return "bad";
      default: return "";
    }
  }

  function itemText(item) {
    const parts = [item.kind, item.role, item.model, item.prompt, item.reply, item.error, item.node, item.task, item.text, item.result];
    try { if (item.update) parts.push(JSON.stringify(item.update)); if (item.info) parts.push(JSON.stringify(item.info)); }
    catch (_) { /* unserialisable: skip */ }
    return parts.filter((p) => typeof p === "string" && p).join("\n").toLowerCase();
  }

  // opts: { kinds: Set of llm|node|pause|run (omit for all), onlyFailures: bool, query: string }
  function filterItems(items, opts) {
    const o = opts || {};
    const query = typeof o.query === "string" ? o.query.trim().toLowerCase() : "";
    return items.filter((it) => {
      if (o.kinds && !o.kinds.has(category(it))) return false;
      if (o.onlyFailures && outcome(it) !== "bad") return false;
      return !query || itemText(it).includes(query);
    });
  }

  // ------------------------------------------------------------------ numbers for the header and charts
  function runStats(items) {
    const byRole = {};
    const nodeSeconds = {};
    let tokens = { input: 0, output: 0, total: 0 };
    let calls = 0, laps = 0, rejections = 0, passed = 0, failed = 0, first = null, last = null;
    for (const it of items) {
      const t0 = num(it.ts);
      if (t0 && (first === null || t0 < first)) first = t0;
      const t1 = t0 + (it.kind === "llm" || it.kind === "node" ? num(it.seconds) : 0);
      if (t1 && (last === null || t1 > last)) last = t1;
      if (it.kind === "llm" && it.done && !it.error) {
        calls++;
        const r = byRole[it.role] = byRole[it.role] || { calls: 0, seconds: 0, tokens: 0 };
        const tk = it.tokens || {};
        r.calls++; r.seconds += num(it.seconds); r.tokens += num(tk.total_tokens);
        tokens.input += num(tk.input_tokens); tokens.output += num(tk.output_tokens); tokens.total += num(tk.total_tokens);
        if (it.role === "worker" || it.role === "escalation") laps++;
      } else if (it.kind === "node") {
        nodeSeconds[it.node] = (nodeSeconds[it.node] || 0) + num(it.seconds);
        if (it.node === "reviewer" && it.update && typeof it.update.rejections === "number") rejections = it.update.rejections;
        if (it.node === "check") { if (it.update && it.update.tests_passed === true) passed++; else if (it.update && it.update.tests_passed === false) failed++; }
      }
    }
    return { calls, laps, rejections, tokens, byRole, nodeSeconds, checks: { passed, failed },
      wallSeconds: first !== null && last !== null ? Math.max(0, last - first) : 0 };
  }

  // 1, 2, 5 x 10^n at or above v, for friendly axis ends.
  function niceMax(v) {
    v = num(v);
    if (v <= 0) return 1;
    const pow = Math.pow(10, Math.floor(Math.log10(v)));
    for (const step of [1, 2, 5, 10]) if (step * pow >= v) return step * pow;
    return 10 * pow;
  }

  // entries: [{label, value}] -> bar rectangles inside a width x height box.
  function barGeometry(entries, opts) {
    const o = Object.assign({ width: 320, height: 160, left: 44, right: 8, top: 8, bottom: 28, gap: 8 }, opts || {});
    const list = (Array.isArray(entries) ? entries : []).map((e) => ({ label: String((e && e.label) || ""), value: Math.max(0, num(e && e.value)) }));
    const max = niceMax(Math.max(0, ...list.map((e) => e.value)));
    const plotW = Math.max(1, o.width - o.left - o.right), plotH = Math.max(1, o.height - o.top - o.bottom);
    const slot = list.length ? plotW / list.length : plotW;
    const barW = Math.max(1, slot - o.gap);
    const bars = list.map((e, i) => {
      const h = (e.value / max) * plotH;
      return { label: e.label, value: e.value, x: o.left + i * slot + o.gap / 2, y: o.top + plotH - h, w: barW, h };
    });
    const ticks = [0, 0.5, 1].map((f) => ({ value: max * f, y: o.top + plotH - f * plotH }));
    return { bars, max, ticks, box: o };
  }

  // Model calls laid out on a time axis. `now` is used for calls that have not finished.
  function timelineSpans(items, opts) {
    const o = Object.assign({ width: 640, left: 0, now: 0 }, opts || {});
    const calls = items.filter((it) => it.kind === "llm" && num(it.ts) > 0)
      .map((it) => ({ it, start: it.ts, dur: it.done ? num(it.seconds) : Math.max(0, num(o.now) - it.ts) }));
    if (!calls.length) return { spans: [], t0: 0, t1: 0 };
    const t0 = Math.min(...calls.map((c) => c.start));
    const t1 = Math.max(...calls.map((c) => c.start + c.dur), t0 + 1);
    const scale = (o.width - o.left) / (t1 - t0);
    const spans = calls.map((c) => ({
      x: o.left + (c.start - t0) * scale, w: Math.max(2, c.dur * scale), role: c.it.role, model: c.it.model,
      seconds: c.dur, running: !c.it.done, key: c.it.key,
    }));
    return { spans, t0, t1 };
  }

  // series: [{label, value (0..1)}] -> points and an SVG path, with no NaN for 0 or 1 points.
  function trendPoints(series, opts) {
    const o = Object.assign({ width: 320, height: 120, left: 8, right: 8, top: 8, bottom: 8 }, opts || {});
    const list = (Array.isArray(series) ? series : []).map((s) => ({ label: String((s && s.label) || ""), value: Math.min(1, Math.max(0, num(s && s.value))) }));
    const plotW = Math.max(1, o.width - o.left - o.right), plotH = Math.max(1, o.height - o.top - o.bottom);
    const points = list.map((s, i) => ({
      label: s.label, value: s.value,
      x: list.length === 1 ? o.left + plotW / 2 : o.left + (i / (list.length - 1)) * plotW,
      y: o.top + (1 - s.value) * plotH,
    }));
    const path = points.map((p, i) => (i ? "L" : "M") + p.x.toFixed(1) + " " + p.y.toFixed(1)).join(" ");
    return { points, path };
  }

  // ------------------------------------------------------------------ run list
  // opts: { query, statuses: Set (omit for all), kind }
  function matchRun(run, opts) {
    const o = opts || {};
    if (o.statuses && !o.statuses.has(run.status)) return false;
    if (o.kind && run.kind !== o.kind) return false;
    const q = typeof o.query === "string" ? o.query.trim().toLowerCase() : "";
    return !q || [run.id, run.task, run.result, run.kind].some((v) => typeof v === "string" && v.toLowerCase().includes(q));
  }

  const LIVE_FIRST = { running: 0, paused: 1, error: 2, stopped: 3, finished: 4 };
  function sortRuns(runs) {
    return runs.slice().sort((a, b) =>
      (LIVE_FIRST[a.status] ?? 9) - (LIVE_FIRST[b.status] ?? 9) || num(b.updated) - num(a.updated));
  }

  // ------------------------------------------------------------------ formatting
  function formatSeconds(n) {
    n = num(n);
    if (n < 100) return n.toFixed(1) + "s";
    if (n < 3600) return Math.floor(n / 60) + "m " + String(Math.round(n % 60)).padStart(2, "0") + "s";
    return Math.floor(n / 3600) + "h " + String(Math.floor((n % 3600) / 60)).padStart(2, "0") + "m";
  }
  function formatCount(n) {
    n = num(n);
    if (n < 1000) return String(Math.round(n));
    if (n < 100000) return (n / 1000).toFixed(1).replace(/\.0$/, "") + "k";
    return Math.round(n / 1000) + "k";
  }

  return { parseDiff, sideBySide, buildItems, replyTitle, category, outcome, itemText, filterItems, runStats, niceMax,
    barGeometry, timelineSpans, trendPoints, matchRun, sortRuns, formatSeconds, formatCount };
});
