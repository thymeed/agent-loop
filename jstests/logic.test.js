"use strict";
// Run with: node jstests/logic.test.js   (also run by _smoke_test.py when Node is installed)
const assert = require("node:assert/strict");
const L = require("../agent_loop/web/logic.js");

let checks = 0;
const check = (name, fn) => { fn(); checks++; };

const PATCH = [
  "diff --git a/calc.py b/calc.py",
  "--- a/calc.py",
  "+++ b/calc.py",
  "@@ -1,3 +1,3 @@",
  " def add(a, b):",
  "-    return a - b",
  "+    return a + b",
  " ",
  "diff --git a/new.py b/new.py",
  "new file mode 100644",
  "--- /dev/null",
  "+++ b/new.py",
  "@@ -0,0 +1,2 @@",
  "+z = 1",
  "+y = 2",
  "\\ No newline at end of file",
].join("\n");

check("parseDiff reads files, statuses, counts and line numbers", () => {
  const { files, truncated } = L.parseDiff(PATCH);
  assert.equal(truncated, false);
  assert.deepEqual(files.map((f) => [f.path, f.status, f.additions, f.deletions]),
    [["calc.py", "modified", 1, 1], ["new.py", "added", 2, 0]]);
  const lines = files[0].hunks[0].lines;
  assert.deepEqual(lines.map((l) => l.type), ["ctx", "del", "add", "ctx"]);
  assert.deepEqual([lines[1].oldNo, lines[1].newNo, lines[2].oldNo, lines[2].newNo], [2, null, null, 2]);
  assert.equal(files[1].hunks[0].lines.at(-1).type, "meta");
});

check("parseDiff treats header-looking body lines as content (the counts decide)", () => {
  const tricky = ["diff --git a/x.py b/x.py", "@@ -1 +1 @@", "--- not a header", "+++ not a header"].join("\n");
  const f = L.parseDiff(tricky).files[0];
  assert.deepEqual(f.hunks[0].lines.map((l) => [l.type, l.text]), [["del", "-- not a header"], ["add", "++ not a header"]]);
  assert.equal(f.hunks.length, 1);
});

check("parseDiff keeps hostile text as text and never throws", () => {
  const evil = "diff --git a/<img src=x onerror=alert(1)>.py b/<img src=x onerror=alert(1)>.py\n@@ -1 +1 @@\n-</pre><script>1</script>\n+ok";
  const f = L.parseDiff(evil).files[0];
  assert.equal(f.path, "<img src=x onerror=alert(1)>.py");
  assert.equal(f.hunks[0].lines[0].text, "</pre><script>1</script>");
  for (const bad of [null, undefined, 42, {}, [], "", "\u0000\u0000", "@@ -1 +1 @@", "diff --git a/x b/x\n@@ nonsense", "diff --git a/x b/x\n@@ -99999999999999999999 +1 @@\n+a"]) {
    assert.doesNotThrow(() => L.parseDiff(bad));
  }
  assert.deepEqual(L.parseDiff(null), { files: [], truncated: false });
  assert.equal(L.parseDiff("junk before\nmore junk\n" + PATCH).files.length, 2);
  assert.equal(L.parseDiff(PATCH.replace(/\n/g, "\r\n")).files[0].additions, 1);
});

check("parseDiff stops at the line cap instead of building a huge page", () => {
  const big = "diff --git a/x b/x\n@@ -1,0 +1,9000 @@\n" + "+line\n".repeat(9000);
  const r = L.parseDiff(big);
  assert.equal(r.truncated, true);
  assert.ok(r.files[0].hunks[0].lines.length <= 5000);
});

check("sideBySide pairs removals with additions and keeps context on both sides", () => {
  const rows = L.sideBySide(L.parseDiff(PATCH).files[0].hunks[0]);
  assert.equal(rows.length, 3);
  assert.equal(rows[1].left.text, "    return a - b");
  assert.equal(rows[1].right.text, "    return a + b");
  assert.equal(rows[0].left, rows[0].right);
  const uneven = L.sideBySide({ lines: [{ type: "del", text: "a" }, { type: "del", text: "b" }, { type: "add", text: "c" }] });
  assert.deepEqual(uneven.map((r) => [r.left && r.left.text, r.right && r.right.text]), [["a", "c"], ["b", null]]);
  assert.deepEqual(L.sideBySide(null), []);
});

const EVENTS = [
  { ts: 100, event: "run_start", kind: "code", task: "Write is_palindrome", models: { worker: "m" } },
  { ts: 101, event: "llm_start", role: "worker", model: "qwen", prompt: "PROMPT one" },
  { ts: 103, event: "llm_end", role: "worker", seconds: 2, reply: "REPLY one", tokens: { input_tokens: 10, output_tokens: 5, total_tokens: 15 } },
  { ts: 103, event: "node", node: "worker", seconds: 2, update: {} },
  { ts: 104, event: "node", node: "check", seconds: 1, update: { tests_passed: false, test_output: "AssertionError" } },
  { ts: 105, event: "llm_start", role: "reviewer", model: "qwen", prompt: "judge" },
  { ts: 107, event: "llm_end", role: "reviewer", seconds: 2, reply: "REJECTED", tokens: { total_tokens: 7 } },
  { ts: 107, event: "node", node: "reviewer", seconds: 2, update: { approved: false, rejections: 1 } },
  { ts: 108, event: "llm_start", role: "worker", model: "qwen", prompt: "PROMPT two" },
  { ts: 112, event: "llm_end", role: "worker", seconds: 4, reply: "REPLY two", tokens: { total_tokens: 20 } },
  { ts: 113, event: "node", node: "check", seconds: 1, update: { tests_passed: true } },
  { ts: 114, event: "llm_start", role: "worker", model: "qwen", prompt: "still thinking" },
];

check("buildItems merges start/end per role in order and keeps unfinished calls open", () => {
  const items = L.buildItems(EVENTS);
  const llm = items.filter((i) => i.kind === "llm");
  assert.deepEqual(llm.map((i) => [i.role, i.done, i.reply]), [["worker", true, "REPLY one"], ["reviewer", true, "REJECTED"], ["worker", true, "REPLY two"], ["worker", false, null]]);
  assert.equal(items.map((i) => i.key).join(), items.map((_, i) => i).join());
  assert.deepEqual(L.buildItems(null), []);
  assert.doesNotThrow(() => L.buildItems([null, 3, "x", { event: "llm_end", role: "ghost" }, { event: "unknown" }]));
  assert.equal(L.buildItems([{ event: "llm_start", role: "w", ts: 1 }, { event: "llm_error", role: "w", seconds: 1, error: "boom" }])[0].error, "boom");
});

check("a retried call stays one item and remembers it was retried", () => {
  const items = L.buildItems([
    { ts: 1, event: "llm_start", role: "worker", model: "m", prompt: "p" },
    { ts: 2, event: "llm_retry", role: "worker", wait: 5, error: "CUDA error" },
    { ts: 8, event: "llm_end", role: "worker", seconds: 7, reply: "r", tokens: { total_tokens: 3 } },
  ]);
  assert.equal(items.length, 1);
  assert.deepEqual([items[0].retries, items[0].done, items[0].reply], [1, true, "r"]);
  assert.doesNotThrow(() => L.buildItems([{ event: "llm_retry", role: "ghost" }]));
});

check("outcome and category classify items", () => {
  const items = L.buildItems(EVENTS);
  const checks_ = items.filter((i) => i.kind === "node" && i.node === "check").map(L.outcome);
  assert.deepEqual(checks_, ["bad", "good"]);
  assert.equal(L.outcome(items.find((i) => i.node === "reviewer")), "bad");
  assert.equal(L.category(items[0]), "run");
  assert.equal(L.category(items[1]), "llm");
});

check("filterItems: kinds, only failures and text search (case-insensitive, searches prompts and replies)", () => {
  const items = L.buildItems(EVENTS);
  assert.equal(L.filterItems(items, {}).length, items.length);
  assert.ok(L.filterItems(items, { kinds: new Set(["llm"]) }).every((i) => i.kind === "llm"));
  assert.deepEqual(L.filterItems(items, { onlyFailures: true }).map((i) => i.node || i.kind), ["check", "reviewer"]);
  assert.equal(L.filterItems(items, { query: "reply TWO" }).length, 1);
  assert.equal(L.filterItems(items, { query: "  assertionerror " }).length, 1);
  assert.equal(L.filterItems(items, { query: "no such text" }).length, 0);
  assert.equal(L.filterItems(items, { kinds: new Set(["llm"]), query: "judge" })[0].role, "reviewer");
});

check("runStats totals calls, tokens, laps, rejections and checks", () => {
  const s = L.runStats(L.buildItems(EVENTS));
  assert.equal(s.calls, 3);
  assert.equal(s.laps, 2);
  assert.equal(s.rejections, 1);
  assert.deepEqual(s.checks, { passed: 1, failed: 1 });
  assert.equal(s.tokens.total, 15 + 7 + 20);
  assert.deepEqual(s.byRole.worker, { calls: 2, seconds: 6, tokens: 35 });
  assert.equal(s.nodeSeconds.check, 2);
  assert.equal(s.wallSeconds, 114 - 100);
  assert.deepEqual(L.runStats([]).byRole, {});
});

check("niceMax and barGeometry scale safely", () => {
  assert.deepEqual([0, -3, NaN, undefined, 0.9, 1, 1.1, 3, 7, 14, 120, 999].map(L.niceMax), [1, 1, 1, 1, 1, 1, 2, 5, 10, 20, 200, 1000]);
  const g = L.barGeometry([{ label: "a", value: 30 }, { label: "b", value: 100 }, { label: "c", value: 0 }], { width: 300, height: 100, left: 0, right: 0, top: 0, bottom: 0, gap: 10 });
  assert.equal(g.max, 100);
  assert.equal(g.bars[1].h, 100);
  assert.equal(g.bars[0].h, 30);
  assert.equal(g.bars[2].h, 0);
  assert.ok(g.bars.every((b) => b.x >= 0 && b.x + b.w <= 300 && b.y >= 0 && isFinite(b.y)));
  for (const bad of [null, [], [{ label: "x", value: NaN }], [{}, null], [{ label: "n", value: -5 }]]) {
    const r = L.barGeometry(bad);
    assert.ok(r.bars.every((b) => [b.x, b.y, b.w, b.h].every(isFinite) && b.h >= 0));
  }
});

check("timelineSpans places calls on a time axis and sizes unfinished ones from `now`", () => {
  const items = L.buildItems(EVENTS);
  const t = L.timelineSpans(items, { width: 100, now: 118 });
  assert.equal(t.spans.length, 4);
  assert.equal(t.t0, 101);
  assert.ok(t.spans.every((s) => s.x >= 0 && s.x + s.w <= 100.0001 && s.w >= 2));
  const open = t.spans.find((s) => s.running);
  assert.equal(open.seconds, 4);
  assert.deepEqual(L.timelineSpans([], { width: 50 }), { spans: [], t0: 0, t1: 0 });
});

check("trendPoints gives finite coordinates for 0, 1 and many points", () => {
  assert.deepEqual(L.trendPoints([]).points, []);
  const one = L.trendPoints([{ label: "a", value: 0.5 }], { width: 100, height: 100, left: 0, right: 0, top: 0, bottom: 0 });
  assert.equal(one.points[0].x, 50);
  assert.equal(one.points[0].y, 50);
  assert.ok(!/NaN/.test(one.path));
  const many = L.trendPoints([{ label: "a", value: 0 }, { label: "b", value: 2 }, { label: "c", value: -1 }, { label: "d", value: NaN }]);
  assert.ok(many.points.every((p) => isFinite(p.x) && isFinite(p.y) && p.value >= 0 && p.value <= 1));
  assert.ok(many.path.startsWith("M"));
});

check("matchRun and sortRuns", () => {
  const runs = [
    { id: "old", status: "finished", kind: "code", task: "palindrome", result: "approved", updated: 5 },
    { id: "now", status: "running", kind: "project", task: "fix slugify", result: null, updated: 1 },
    { id: "wait", status: "paused", kind: "code", task: "csv", result: null, updated: 9 },
    { id: "new", status: "finished", kind: "code", task: "fizz", result: "approved", updated: 8 },
  ];
  assert.deepEqual(L.sortRuns(runs).map((r) => r.id), ["now", "wait", "new", "old"]);
  assert.equal(runs.filter((r) => L.matchRun(r, { query: "SLUG" })).length, 1);
  assert.equal(runs.filter((r) => L.matchRun(r, { statuses: new Set(["finished"]) })).length, 2);
  assert.equal(runs.filter((r) => L.matchRun(r, { kind: "project" })).length, 1);
  assert.equal(runs.filter((r) => L.matchRun(r, { query: "approved", statuses: new Set(["running"]) })).length, 0);
});

check("formatters", () => {
  assert.deepEqual([0, 2.345, 99.94, 100, 754, 3700, NaN].map(L.formatSeconds), ["0.0s", "2.3s", "99.9s", "1m 40s", "12m 34s", "1h 01m", "0.0s"]);
  assert.deepEqual([0, 999, 1000, 1234, 15700, 250000, null].map(L.formatCount), ["0", "999", "1k", "1.2k", "15.7k", "250k", "0"]);
});

console.log(`${checks} logic checks passed`);
