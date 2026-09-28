"use strict";
/* Přepis – recording -> transcript_clean.txt. Talks only to server.py on 127.0.0.1. */

// ------------------------------------------------------------------ helpers
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "html") el.innerHTML = v;
    else if (k in el && typeof v !== "string") el[k] = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return el;
}

const ICON = {
  play: '<svg viewBox="0 0 16 16"><path d="M4 2.5v11a.5.5 0 0 0 .77.42l8.5-5.5a.5.5 0 0 0 0-.84l-8.5-5.5A.5.5 0 0 0 4 2.5z"/></svg>',
  pause: '<svg viewBox="0 0 16 16"><rect x="3.5" y="2.5" width="3" height="11" rx="1"/><rect x="9.5" y="2.5" width="3" height="11" rx="1"/></svg>',
  back: '<svg viewBox="0 0 16 16"><path d="M8 3a5 5 0 1 1-4.9 6h1.53A3.5 3.5 0 1 0 8 4.5V7L4.5 3.75 8 .5V3z"/></svg>',
  fwd: '<svg viewBox="0 0 16 16"><path d="M8 3a5 5 0 1 0 4.9 6h-1.53A3.5 3.5 0 1 1 8 4.5V7l3.5-3.25L8 .5V3z"/></svg>',
};
function fill(el, ...kids) {
  el.replaceChildren();
  for (const c of kids.flat(Infinity)) if (c != null && c !== false) el.append(c.nodeType ? c : document.createTextNode(String(c)));
  return el;
}
const icon = (n) => { const s = document.createElement("span"); s.innerHTML = ICON[n]; return s.firstChild; };

async function jfetch(url, opt) {
  const r = await fetch(url, opt);
  const d = await r.json().catch(() => ({ error: r.statusText }));
  if (!r.ok) throw Object.assign(new Error(d.error || r.statusText), { status: r.status });
  return d;
}
const api = {
  get: (u) => jfetch(u),
  send: (m, u, b) => jfetch(u, { method: m, headers: { "Content-Type": "application/json" }, body: JSON.stringify(b || {}) }),
};

function fmtT(t) {
  t = Math.max(0, Math.floor(t || 0));
  const hh = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60;
  return hh ? `${hh}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}` : `${m}:${String(s).padStart(2, "0")}`;
}
function fmtTd(t) { t = Math.max(0, t || 0); return fmtT(t) + "." + Math.floor((t % 1) * 10); }
function fmtDur(t) {
  if (!t) return "";
  t = Math.round(t);
  if (t < 60) return `${t} s`;
  const m = Math.floor(t / 60), s = t % 60;
  return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : s ? `${m} min ${s} s` : `${m} min`;
}
function fmtSize(b) { return b > 1e9 ? (b / 1e9).toFixed(1) + " GB" : Math.round(b / 1e6) + " MB"; }
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
// file names that are just a timestamp (what most recorders write) read better as a date
function recTitle(id) {
  const m = id.match(/^(.*?)[ _-]?(\d{4})-(\d{2})-(\d{2})[ _T](\d{2})[-.:h](\d{2})(?:[-.:m](\d{2}))?$/);
  if (!m || +m[3] < 1 || +m[3] > 12) return id;
  const d = `${+m[4]} ${MONTHS[+m[3] - 1]} ${m[2]}, ${m[5]}:${m[6]}`;
  return m[1] ? `${m[1].replace(/[_-]+/g, " ").trim()}, ${d}` : d;
}

let toastTimer;
function toast(msg, bad = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast show" + (bad ? " bad" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.className = "toast"), bad ? 6000 : 2400);
}
async function copyText(s) {
  try { await navigator.clipboard.writeText(s); toast("Copied"); }
  catch { toast("Copy failed – select the text instead", true); }
}
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ------------------------------------------------------------------ app state
const S = {
  state: null,          // /api/state
  rid: null, tab: null,
  rec: null,            // /api/rec/<id>
  doc: null,            // /api/rec/<id>/doc
  tracks: null,
  jobLines: [], jobId: null,
};
const audio = $("#audio");
const preview = $("#preview");

// ------------------------------------------------------------------ polling + chrome
let pollTimer;
async function poll() {
  clearTimeout(pollTimer);
  try {
    const prev = S.state;
    S.state = await api.get("/api/state");
    renderLibrary();
    renderChrome();
    // the open recording's job moved on: refresh its view
    if (S.rid) {
      const before = prev && prev.recordings.find((r) => r.id === S.rid);
      const now = S.state.recordings.find((r) => r.id === S.rid);
      const sig = (r) => r ? JSON.stringify([r.stage, r.files, r.job && [r.job.state, r.job.step, r.job.done_steps]]) : "";
      if (now && sig(before) !== sig(now)) onRecChanged(before, now);
      else if (now && now.job && S.tab === "transcribe") renderJob(now.job);
    }
  } catch (e) {
    $("#gpuChip").textContent = "server not reachable";
  }
  const busy = S.state && (S.state.jobs.length || Object.keys(S.state.analysing || {}).length);
  pollTimer = setTimeout(poll, busy ? 1200 : 5000);
}

function renderChrome() {
  const g = S.state.gpu;
  const chip = $("#gpuChip");
  fill(chip, );
  if (!g.available) {
    chip.className = "gpu-chip";
    chip.append(h("span", { class: "dot" }), h("span", { class: "txt" }, "GPU: can't check"));
  } else {
    const free = g.free_mb / 1024;
    chip.className = "gpu-chip " + (free >= 6 ? "ok" : free >= 2 ? "low" : "bad");
    chip.append(h("span", { class: "dot" }), h("span", {}, `${free.toFixed(1)} GB free`), h("span", { class: "txt" }, ` of ${Math.round(g.total_mb / 1024)} GB`));
  }
  if (!$("#gpuPop").hidden) renderGpuPop();
  let warn = $("#gitWarn");
  if (S.state.work_ignored === false && !warn) {
    document.querySelector(".topbar").after(h("div", { id: "gitWarn", class: "note bad", role: "alert", style: { margin: "0", borderRadius: "0" } },
      h("strong", {}, "Transcripts are not protected from git."),
      "The work/ folder is not in .gitignore (or git already tracks files in it), so a commit could include private transcripts. Add work/ to .gitignore."));
  } else if (S.state.work_ignored !== false && warn) warn.remove();
  const jc = $("#jobChip");
  const running = S.state.jobs.find((j) => j.state === "running") || S.state.jobs[0];
  fill(jc, );
  if (running) {
    const pct = running.progress != null ? running.progress : null;
    const label = running.state === "blocked" ? "Waiting for GPU" : running.state === "queued" ? "Queued" :
      { audio: "Extracting audio", transcribe: "Transcribing", diarize: "Finding speakers" }[running.step] || "Working";
    jc.append(h("button", { class: "job-chip", onclick: () => go(running.rid, "transcribe") },
      h("span", { class: "dot" }), `${label} · ${recTitle(running.rid)}`,
      pct != null && running.step === "transcribe" ? h("span", { class: "bar" }, h("i", { style: { width: pct + "%" } })) : null));
  }
}

function gpuBlock(g, needGb) {
  if (!g || !g.available) return h("p", { class: "muted small" }, g ? g.reason : "");
  const used = g.used_mb / g.total_mb * 100;
  const others = g.procs.filter((p) => !p.ours && (p.mb == null || p.mb >= 100));
  return h("div", {},
    h("div", { class: "vram", title: `${(g.used_mb / 1024).toFixed(1)} GB used` },
      h("i", { style: { width: used + "%", background: "var(--ink-2)" } })),
    h("div", { class: "small muted" }, `${(g.free_mb / 1024).toFixed(1)} GB free of ${(g.total_mb / 1024).toFixed(1)} GB` +
      (needGb ? ` · needs about ${needGb} GB` : "")),
    others.length ? h("ul", { class: "holders" }, others.map((p) =>
      h("li", {},
        h("div", { class: "what" }, `${p.what}`, h("span", { class: "muted" }, `  ${p.mb != null ? (p.mb / 1024).toFixed(1) + " GB" : ""} · pid ${p.pid}`)),
        p.unit ? h("div", { class: "small" }, `${p.user_unit ? "user service" : "system service"} ${p.unit}`) : null,
        p.container ? h("div", { class: "small" }, `in container ${p.container}`) : null,
        p.cmd ? h("div", { class: "cmd" }, p.cmd) : null,
        p.stop_hint ? h("div", { class: "small", style: { marginTop: "6px" } }, "To free it yourself: ",
          h("span", { class: "copyline" }, h("code", {}, p.stop_hint),
            h("button", { class: "btn sm quiet", onclick: () => copyText(p.stop_hint) }, "Copy"))) : null)))
      : h("p", { class: "small muted" }, "No other program is using GPU memory."),
    g.unattributed_mb > 300 ? h("p", { class: "small muted" },
      `${(g.unattributed_mb / 1024).toFixed(1)} GB is used by the display or by programs that can't be listed.`) : null);
}

function renderGpuPop() {
  const pop = $("#gpuPop");
  fill(pop, h("h3", {}, "GPU memory"), gpuBlock(S.state.gpu),
    h("p", { class: "small muted" }, "Transcription with large-v3 needs about 6 GB free on NVIDIA and 8 GB on other GPUs, speaker detection about 2 GB. This page never stops other programs."));
}
$("#gpuChip").addEventListener("click", (e) => {
  e.stopPropagation();
  const pop = $("#gpuPop");
  pop.hidden = !pop.hidden;
  if (!pop.hidden) renderGpuPop();
});
document.addEventListener("click", (e) => {
  if (!$("#gpuPop").hidden && !e.target.closest("#gpuPop")) $("#gpuPop").hidden = true;
  if ($("#library").classList.contains("open") && !e.target.closest("#library") && !e.target.closest("#libToggle")) toggleLib(false);
});
function toggleLib(open) {
  $("#library").classList.toggle("open", open);
  $("#libToggle").setAttribute("aria-expanded", open);
}
$("#libToggle").addEventListener("click", () => toggleLib(!$("#library").classList.contains("open")));

const STAGE_TEXT = {
  new: ["New", "new"], audio: ["Audio ready", "work"], transcribed: ["Transcribed", "work"],
  diarized: ["Speakers found", "work"], named: ["Ready to review", "review"], reviewing: ["In review", "review"],
  done: ["Done", "done"],
};
function stageOf(r) {
  const j = r.job;
  if (j && j.state === "running") {
    const s = { audio: "Extracting audio", transcribe: "Transcribing", diarize: "Finding speakers" }[j.step] || "Working";
    return [j.step === "transcribe" && j.progress != null ? `${s} ${Math.round(j.progress)}%` : s, "run"];
  }
  if (j && j.state === "queued") return ["Queued", "run"];
  if (j && j.state === "blocked") return ["Waiting for GPU memory", "bad"];
  if (j && j.state === "failed") return ["Failed", "bad"];
  if (r.files.clean === "stale" && r.stage !== "done") return ["Edited since export", "review"];
  return STAGE_TEXT[r.stage] || [r.stage, ""];
}

function renderLibrary() {
  const lib = $("#library");
  const recs = S.state.recordings;
  const vd = S.state.settings.videos_dir;

  const item = (r) => {
    const [txt, cls] = stageOf(r);
    return h("button", {
      class: "lib-item", "aria-current": r.id === S.rid ? "true" : "false",
      onclick: () => { go(r.id); toggleLib(false); },
    },
      h("span", { class: "t" }, recTitle(r.id)),
      h("span", { class: "d num" }, r.duration ? fmtT(r.duration) : ""),
      h("span", { class: "s" }, h("span", { class: "stage " + cls }, txt),
        r.files.clean === "other" && r.stage !== "done" ? h("span", { class: "extra" }, "hand-made transcript") : null,
        r.media && r.media.growing ? h("span", { class: "extra" }, "still recording") : null));
  };
  fill(lib, 
    h("div", { class: "lib-head" }, h("h2", {}, "Recordings"), h("small", { title: vd }, vd.replace(/^\/home\/[^/]+/, "~"))),
    recs.map(item),
    recs.length ? null : h("p", { class: "muted small", style: { padding: "0 10px" } }, "No recordings found in this folder."));
}

// ------------------------------------------------------------------ routing
function go(rid, tab) {
  location.hash = rid ? `#/r/${encodeURIComponent(rid)}${tab ? "/" + tab : ""}` : "#/";
}
window.addEventListener("hashchange", route);

async function route() {
  const m = location.hash.match(/^#\/r\/([^/]+)(?:\/(\w+))?/);
  const rid = m ? decodeURIComponent(m[1]) : null;
  const tab = m && m[2];
  if (rid !== S.rid) {
    leaveReview();
    S.rid = rid; S.rec = null; S.doc = null; S.tracks = null; S.jobLines = []; S.jobId = null;
    if (!audio.paused) audio.pause();
    audio.removeAttribute("src");
  }
  if (S.state) renderLibrary();
  if (!rid) return renderHome();
  try {
    S.rec = await api.get(`/api/rec/${encodeURIComponent(rid)}`);
  } catch (e) {
    fill($("#main"), h("div", { class: "empty" }, h("h1", {}, "Recording not found"), h("p", {}, e.message)));
    return;
  }
  S.tab = tab || defaultTab(S.rec);
  if (S.tab !== "review") leaveReview();
  renderRec();
}

function defaultTab(r) {
  if (r.job && ["queued", "running", "blocked", "failed"].includes(r.job.state)) return "transcribe";
  if (!r.files.transcript) return r.files.audio ? "transcribe" : "track";
  if (!r.files.speakers) return "speakers";
  return "review";
}

function onRecChanged(before, now) {
  // job finished a step, or the files changed underneath us
  api.get(`/api/rec/${encodeURIComponent(S.rid)}`).then((rec) => {
    const wasTranscribing = S.tab === "transcribe";
    S.rec = rec;
    if (wasTranscribing && now.job && now.job.state === "done" && !(before && before.job && before.job.state === "done")) {
      toast(rec.files.diarized ? "Transcribed. Now name the speakers." : "Transcribed.");
    }
    if (S.tab !== "review") renderRec();
    else renderSteps();
  });
}

function renderHome() {
  const recs = (S.state && S.state.recordings) || [];
  const todo = recs.filter((r) => r.stage === "new");
  const open = recs.filter((r) => ["named", "reviewing", "transcribed", "diarized"].includes(r.stage));
  fill($("#main"), h("div", { class: "empty" },
    h("h1", {}, "From a recording to a clean transcript"),
    h("p", {}, `Pick a recording on the left. Audio and video files in ${((S.state && S.state.settings.videos_dir) || "your Videos folder").replace(/^\/home\/[^/]+/, "~")} show up there on their own. Each one goes through five steps: choose the audio track, transcribe, name the speakers, correct the text while listening, save transcript_clean.txt.`),
    open.length ? h("p", {}, "Waiting for you: ", open.slice(0, 3).map((r, i) => [i ? ", " : "", h("a", { href: `#/r/${encodeURIComponent(r.id)}` }, recTitle(r.id))])) : null,
    todo.length ? h("p", {}, "Not transcribed yet: ", todo.slice(0, 4).map((r, i) => [i ? ", " : "", h("a", { href: `#/r/${encodeURIComponent(r.id)}` }, recTitle(r.id))])) : null));
}

// ------------------------------------------------------------------ recording page
const TABS = [["track", "Audio track"], ["transcribe", "Transcribe"], ["speakers", "Speakers"], ["review", "Review"], ["export", "Export"]];

function tabState(r) {
  const f = r.files;
  const job = r.job && ["queued", "running", "blocked", "failed"].includes(r.job.state);
  return {
    track: { enabled: true, done: f.audio },
    transcribe: { enabled: f.audio || job || f.transcript, done: f.transcript && (f.diarized || f.speakers) },
    speakers: { enabled: f.transcript, done: f.speakers },
    review: { enabled: f.transcript, done: f.review },
    export: { enabled: f.transcript, done: f.clean === "current" },
  };
}

function renderSteps() {
  const el = $(".steps");
  if (!el) return;
  const ts = tabState(S.rec);
  fill(el, ...TABS.map(([k, label], i) => h("a", {
    href: `#/r/${encodeURIComponent(S.rid)}/${k}`, class: ts[k].done ? "done" : "",
    "aria-current": S.tab === k ? "page" : null, "aria-disabled": ts[k].enabled ? null : "true",
  }, h("span", { class: "n" }, ts[k].done ? "✓" : i + 1), label)));
}

function renderRec() {
  const r = S.rec;
  const m = r.media;
  const meta = r.meta || {};
  const sub = [];
  if (r.duration) sub.push(fmtDur(r.duration));
  if (m) sub.push(m.name + " · " + fmtSize(m.size));
  else if (meta.source) sub.push("original file is gone: " + meta.source);
  if (meta.audio_track) sub.push(`audio from track ${meta.audio_track}`);
  const head = h("section", { class: "rec-head" },
    h("h1", {}, recTitle(r.id)),
    h("div", { class: "sub" }, sub.map((s) => h("span", {}, s))),
    h("nav", { class: "steps", "aria-label": "Steps" }));
  const body = h("div", { id: "tabBody" });
  fill($("#main"), head, body);
  renderSteps();
  ({ track: renderTrack, transcribe: renderTranscribe, speakers: renderSpeakers, review: renderReview, export: renderExport }[S.tab] || renderTrack)(body);
}

// ------------------------------------------------------------------ step 1: track
async function renderTrack(body) {
  const r = S.rec;
  const s = S.state ? S.state.settings : { language: "", model: "large-v3" };
  const pane = h("section", { class: "pane" });
  fill(body, pane);
  pane.append(h("h2", {}, "Which audio track should be transcribed?"));
  if (!r.media) {
    pane.append(h("p", { class: "lead" }, r.files.audio
      ? `The original recording is not available any more. The extracted audio.wav${r.meta.audio_track ? ` (track ${r.meta.audio_track})` : ""} is used.`
      : "The original recording is not available, so there is nothing to transcribe."));
    return;
  }
  if (r.files.audio) {
    const t = r.meta.audio_track;
    pane.append(h("div", { class: "note good" },
      h("strong", {}, t ? `Audio was taken from track ${t}${r.meta.audio_tracks ? ` of ${r.meta.audio_tracks}` : ""}.` : "Audio was taken from the first track."),
      h("p", {}, "To use a different track, start over from the audio. The current audio, transcript, speaker names and edits are moved to a previous/ folder, not deleted."),
      h("div", { class: "row", style: { marginTop: "10px" } },
        h("button", { class: "btn", onclick: () => redo("audio") }, "Start over with another track"))));
  }
  const lead = h("p", { class: "lead" });
  pane.append(lead);
  const list = h("div", { class: "tracks" }, h("p", { class: "muted" }, h("span", { class: "spinner" }), " Measuring the tracks…"));
  pane.append(list);
  let tr = S.tracks;
  if (!tr) {
    try { tr = S.tracks = await api.get(`/api/rec/${encodeURIComponent(r.id)}/tracks`); }
    catch (e) { fill(list, h("div", { class: "note bad" }, e.message)); return; }
  }
  if (S.tab !== "track" || S.rec !== r) return;
  lead.textContent = tr.tracks.length > 1
    ? `This file has ${tr.tracks.length} audio tracks. Pick the one with the speech you want in the transcript; listen to be sure.`
    : "This file has one audio track.";
  const chosen = { n: r.meta.audio_track || tr.default || 1 };
  const previewAt = Math.min(30, (tr.duration || 0) * 0.15);
  const meterPct = (db) => Math.max(0, Math.min(100, (db + 60) / 60 * 100));
  fill(list, ...tr.tracks.map((t) => {
    const tags = [];
    if (t.silent) tags.push(h("span", { class: "tag" }, "silent"));
    if (t.same_as) tags.push(h("span", { class: "tag" }, `as loud as track ${t.same_as}`));
    const radio = h("input", { type: "radio", name: "track", value: t.n, checked: t.n === chosen.n, disabled: r.files.audio,
      onchange: () => { chosen.n = t.n; startBtn.textContent = `Transcribe track ${t.n}`; } });
    return h("label", { class: "track" },
      radio,
      h("div", {},
        h("div", { class: "name" }, `Track ${t.n}`),
        tags.length ? h("div", { class: "tags" }, tags) : null),
      h("button", { class: "btn sm", type: "button", disabled: t.silent,
        onclick: (e) => { e.preventDefault(); playPreview(r.id, t.n, previewAt, e.currentTarget); } }, icon("play"), "Listen"),
      h("div", { class: "meter", title: `mean ${t.mean_db} dB, peak ${t.max_db} dB` },
        h("i", { style: { width: meterPct(t.mean_db) + "%" } }), h("b", { style: { left: meterPct(t.max_db) + "%" } })),
      h("div", { class: "meter-label" }, t.silent ? "no sound" : `average ${Math.round(t.mean_db)} dB, peaks at ${Math.round(t.max_db)} dB`));
  }));
  if (tr.tracks.every((t) => t.silent)) pane.append(h("div", { class: "note warn" }, h("strong", {}, "Every track is silent."),
    h("p", {}, "There is nothing to transcribe in this recording.")));
  if (r.files.audio) return;

  const lang = h("input", { value: s.language || "", size: 6, placeholder: "auto" });
  const model = h("select", {}, ["large-v3", "turbo", "medium", "small"].map((x) => h("option", { value: x, selected: x === s.model }, x)));
  const diar = h("input", { type: "checkbox", checked: true });
  const remember = h("input", { type: "checkbox" });
  const startBtn = h("button", { class: "btn primary", onclick: async () => {
    startBtn.disabled = true;
    try {
      if (remember.checked) await api.send("PUT", "/api/settings", { track: chosen.n, language: lang.value.trim(), model: model.value });
      await api.send("POST", `/api/rec/${encodeURIComponent(r.id)}/start`,
        { track: chosen.n, language: lang.value.trim(), model: model.value, diarize: diar.checked });
      await poll();
      go(r.id, "transcribe");
    } catch (e) { toast(e.message, true); startBtn.disabled = false; }
  } }, `Transcribe track ${chosen.n}`);
  pane.append(
    h("div", { class: "row", style: { marginTop: "22px" } }, startBtn,
      r.media.growing ? h("span", { class: "muted small" }, "This file is still being written.") : null),
    h("details", { class: "more" }, h("summary", {}, "Transcription settings"),
      h("div", { class: "row" },
        h("label", { class: "field" }, "Language (e.g. en, cs; empty = detect)", lang),
        h("label", { class: "field" }, "Model", model),
        h("label", { class: "check" }, diar, "Find speakers"),
        h("label", { class: "check" }, remember, "Use these by default"))));
}

let previewBtn = null;
function playPreview(rid, track, t, btn) {
  if (previewBtn === btn && !preview.paused) { preview.pause(); return; }
  if (!audio.paused) audio.pause();
  preview.src = `/api/rec/${encodeURIComponent(rid)}/preview?track=${track}&t=${t.toFixed(1)}`;
  preview.play().catch((e) => toast(e.message, true));
  if (previewBtn) fill(previewBtn, icon("play"), "Listen");
  previewBtn = btn;
  fill(btn, icon("pause"), "Stop");
  preview.onended = preview.onpause = () => { fill(btn, icon("play"), "Listen"); };
}

async function redo(from) {
  const what = { audio: "the audio, transcript, speaker names and your edits", transcribe: "the transcript, speaker names and your edits",
    diarize: "the speaker detection, speaker names and your edits" }[from];
  if (!confirm(`This moves ${what} for this recording into a previous/ folder inside its work folder (nothing is deleted). Continue?`)) return;
  try {
    const d = await api.send("POST", `/api/rec/${encodeURIComponent(S.rid)}/redo`, { from });
    toast(d.moved_to ? `Moved to ${d.moved_to}` : "Nothing to move");
    S.rec = await api.get(`/api/rec/${encodeURIComponent(S.rid)}`);
    S.doc = null;
    await poll();
    if (from === "audio") { S.tab = "track"; go(S.rid, "track"); renderRec(); }
    else startJob({});
  } catch (e) { toast(e.message, true); }
}

async function startJob(extra) {
  const s = S.state.settings;
  const m = S.rec.meta || {};
  try {
    await api.send("POST", `/api/rec/${encodeURIComponent(S.rid)}/start`,
      { track: m.audio_track || null, language: m.language || s.language, model: m.model || s.model, diarize: true, ...extra });
    S.rec = await api.get(`/api/rec/${encodeURIComponent(S.rid)}`);
    await poll();
    S.tab = "transcribe";
    if (location.hash.endsWith("/transcribe")) renderRec(); else go(S.rid, "transcribe");
  } catch (e) { toast(e.message, true); }
}

// ------------------------------------------------------------------ step 2: transcribe
function renderTranscribe(body) {
  const pane = h("section", { class: "pane", id: "jobPane" });
  fill(body, pane);
  const r = S.rec;
  const job = r.job;
  if (job) return renderJob(job);
  pane.append(h("h2", {}, r.files.transcript ? "Transcribed" : "Transcribe"));
  if (!r.files.transcript) {
    pane.append(h("p", { class: "lead" }, r.files.audio ? "The audio is ready." : "Choose the audio track first."),
      r.files.audio ? h("button", { class: "btn primary", onclick: () => startJob({}) }, "Transcribe") :
        h("a", { class: "btn primary", href: `#/r/${encodeURIComponent(r.id)}/track` }, "Choose the track"));
    return;
  }
  const m = r.meta;
  pane.append(h("ul", { class: "pipeline" },
    pipelineItem("done", "Audio", m.audio_track ? `track ${m.audio_track}` : "first track"),
    pipelineItem("done", "Transcript", [m.model, m.language, m.device].filter(Boolean).join(", ")),
    pipelineItem(r.files.diarized ? "done" : "", "Speakers", r.files.diarized ? "found" : "not detected yet")));
  if (!r.files.diarized) pane.append(h("div", { class: "row" },
    h("button", { class: "btn primary", onclick: () => startJob({}) }, "Find speakers"),
    h("a", { class: "btn", href: `#/r/${encodeURIComponent(r.id)}/speakers` }, "Skip: only one person talks")));
  else pane.append(h("div", { class: "row" }, h("a", { class: "btn primary", href: `#/r/${encodeURIComponent(r.id)}/speakers` }, "Name the speakers")));
  pane.append(h("h3", {}, "Start over"),
    h("p", { class: "muted small" }, "Earlier results and your edits are moved to previous/ inside the work folder."),
    h("div", { class: "row" },
      r.files.diarized ? h("button", { class: "btn sm", onclick: () => redo("diarize") }, "Find speakers again") : null,
      h("button", { class: "btn sm", onclick: () => redo("transcribe") }, "Transcribe again"),
      h("button", { class: "btn sm", onclick: () => redo("audio") }, "Use another audio track")));
}

function pipelineItem(state, label, det, extra) {
  return h("li", { class: state }, h("span", { class: "ic" }, state === "done" ? "✓" : state === "bad" ? "!" : ""),
    h("div", {}, h("div", { class: "lbl" }, label), det ? h("div", { class: "det" }, det) : null, extra || null));
}

let jobPollTimer;
async function renderJob(job) {
  const pane = $("#jobPane");
  if (!pane) return;
  clearTimeout(jobPollTimer);
  if (S.jobId !== job.id) { S.jobId = job.id; S.jobLines = []; }
  try {
    const since = S.jobLines.length ? S.jobLines[S.jobLines.length - 1].n + 1 : 0;
    const d = await api.get(`/api/job/${job.id}?since=${since}`);
    S.jobLines.push(...d.lines);
    job = d;
  } catch { /* job gone after a restart */ }
  if (!$("#jobPane")) return;
  const names = { audio: "Extract the audio track", transcribe: "Transcribe (Whisper)", diarize: "Find who is speaking" };
  const steps = job.steps.map((s) => {
    let st = job.done_steps.includes(s) ? "done" : job.step === s ? (job.state === "running" ? "run" : job.state === "blocked" ? "wait" : job.state === "failed" ? "bad" : "") : "";
    let det = null, extra = null;
    if (st === "run") {
      const last = [...S.jobLines].reverse().find((l) => l.kind === "step");
      det = last ? last.text.replace(/^\w+: /, "") : "starting…";
      if (s === "transcribe" && job.progress != null) extra = h("div", { class: "progress" }, h("i", { style: { width: job.progress + "%" } }));
    }
    if (s === "audio" && S.rec.meta.audio_track) det = det || `track ${S.rec.meta.audio_track}`;
    return pipelineItem(st, names[s], det, extra);
  });
  const head = job.state === "done" ? "Done" : job.state === "blocked" ? "Waiting for GPU memory" :
    job.state === "failed" ? "Something went wrong" : job.state === "cancelled" ? "Cancelled" : job.state === "queued" ? "Queued" : "Working on it";
  const kids = [h("h2", {}, head)];
  if (job.state === "queued" && job.queue_pos > 1) kids.push(h("p", { class: "lead" }, `Another recording is using the GPU; this one is number ${job.queue_pos} in line.`));
  if (job.state === "running") kids.push(h("p", { class: "lead" }, "You can leave this page; the work goes on in the background."));
  kids.push(h("ul", { class: "pipeline" }, steps));
  if (job.state === "blocked" && job.blocked) {
    const b = job.blocked;
    kids.push(h("div", { class: "note bad" },
      h("strong", {}, `${b.what || "This step"} needs about ${b.need_gb || 6} GB of free GPU memory.`),
      h("p", {}, "Something else is holding it. Free it yourself, then check again. This page never stops other programs."),
      gpuBlock(b.gpu, b.need_gb),
      h("div", { class: "row", style: { marginTop: "12px" } },
        h("button", { class: "btn primary", onclick: () => retryJob(job, false) }, "Check again"),
        h("button", { class: "btn", onclick: () => retryJob(job, true) }, "Start anyway"),
        h("button", { class: "btn quiet", onclick: () => cancelJob(job) }, "Cancel"))));
  }
  if (job.state === "failed") {
    kids.push(h("div", { class: "note bad" }, h("strong", {}, "The step stopped with an error."), h("p", {}, job.error || ""),
      job.blocked && job.blocked.gpu ? gpuBlock(job.blocked.gpu) : null,
      h("div", { class: "row", style: { marginTop: "12px" } },
        h("button", { class: "btn primary", onclick: () => retryJob(job, false) }, "Try again"),
        h("button", { class: "btn quiet", onclick: () => cancelJob(job) }, "Dismiss"))));
  }
  if (["running", "queued"].includes(job.state)) kids.push(h("div", { class: "row" }, h("button", { class: "btn", onclick: () => cancelJob(job) }, "Cancel")));
  if (job.state === "done") kids.push(h("div", { class: "row" }, h("a", { class: "btn primary", href: `#/r/${encodeURIComponent(S.rid)}/speakers` }, "Name the speakers")));
  const log = h("div", { class: "log" }, S.jobLines.map((l) => h("div", { class: l.kind }, (l.kind === "step" ? "» " : l.kind === "cmd" ? "$ " : "") + l.text)));
  const det = h("details", { class: "more", open: pane.querySelector("details[open]") ? true : null }, h("summary", {}, "Full log"), log);
  fill(pane, ...kids, det);
  log.scrollTop = log.scrollHeight;
  if (["running", "queued", "blocked"].includes(job.state) && S.tab === "transcribe") {
    jobPollTimer = setTimeout(() => S.tab === "transcribe" && S.rec && renderJob(job), 1500);
  }
}
async function retryJob(job, force) {
  try { await api.send("POST", `/api/job/${job.id}/retry`, { force_gpu: force }); await poll(); renderJob(job); }
  catch (e) { toast(e.message, true); }
}
async function cancelJob(job) {
  try { await api.send("POST", `/api/job/${job.id}/cancel`); await poll(); S.rec = await api.get(`/api/rec/${encodeURIComponent(S.rid)}`); renderRec(); }
  catch (e) { toast(e.message, true); }
}

// ------------------------------------------------------------------ step 3: speakers
async function loadDoc() {
  if (S.doc && S.doc._rid === S.rid) return S.doc;
  const d = await api.get(`/api/rec/${encodeURIComponent(S.rid)}/doc`);
  d._rid = S.rid;
  S.doc = d;
  return d;
}

async function renderSpeakers(body) {
  const pane = h("section", { class: "pane" });
  fill(body, pane);
  pane.append(h("p", { class: "muted" }, h("span", { class: "spinner" }), " Loading…"));
  let doc;
  try { doc = await loadDoc(); } catch (e) { fill(pane, h("div", { class: "note bad" }, e.message)); return; }
  if (S.tab !== "speakers") return;
  const names = S.state.settings.names || [];
  const cfg = doc.speakers_cfg;
  const sp = doc.speakers;
  const assign = {};
  for (const s of sp) assign[s.id] = cfg.speakers[s.id] || "";
  let single = cfg.default || "";
  fill(pane, );

  const save = async () => {
    try {
      const payload = doc.diarized && sp.length > 1 ? { names: assign } : sp.length === 1 && doc.diarized ? { names: { [sp[0].id]: single } } : { default: single };
      const d = await api.send("POST", `/api/rec/${encodeURIComponent(S.rid)}/speakers`, payload);
      S.doc.speakers_cfg = d.speakers_cfg;
      S.rec = await api.get(`/api/rec/${encodeURIComponent(S.rid)}`);
      poll();
      go(S.rid, "review");
    } catch (e) { toast(e.message, true); }
  };

  const namesEditor = (need) => {
    const input = h("input", { class: "inp", placeholder: "Add a name", size: 12 });
    const add = async () => {
      const v = input.value.trim();
      if (!v) return;
      try { S.state.settings = await api.send("PUT", "/api/settings", { names: [...names, v] }); renderSpeakers(body); }
      catch (e) { toast(e.message, true); }
    };
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") add(); });
    const short = names.length < need;
    return h("details", { class: "more", open: short ? true : null },
      h("summary", {}, short ? (names.length ? "Add another name" : "Add the names of the people in this recording") : "Edit the list of names"),
      h("div", { class: "names-edit" }, names.map((n) => h("span", { class: "chip" }, n,
        h("button", { "aria-label": `Remove ${n}`, onclick: async () => {
          S.state.settings = await api.send("PUT", "/api/settings", { names: names.filter((x) => x !== n) });
          renderSpeakers(body);
        } }, "×"))), input, h("button", { class: "btn sm", onclick: add }, "Add")),
      h("p", { class: "small muted" }, "Letters, digits, _ . - only, no spaces (the names end up in file names)."));
  };

  if (!doc.diarized || sp.length <= 1) {
    pane.append(h("h2", {}, "Who is speaking?"),
      h("p", { class: "lead" }, doc.diarized ? "Only one voice was found in this recording." : "Speakers were not detected for this recording, so the whole transcript gets one name."),
      h("div", { class: "voice", style: { marginBottom: "12px" } }, doc.pieces.slice(0, 2).map((p) => h("blockquote", {},
        h("button", { class: "bplay", style: { background: "var(--ink-2)" }, title: `Play from ${fmtT(p.s)}`, onclick: () => playRange(p.s, p.e) }, icon("play")),
        h("span", {}, p.text)))),
      h("div", { class: "first-q" }, names.map((n) => h("button", { class: "btn" + (single === n ? " primary" : ""), onclick: () => { single = n; save(); } }, n))),
      !doc.diarized ? h("p", { class: "small muted", style: { marginTop: "14px" } }, "Two people in it after all? ",
        h("button", { class: "btn sm", onclick: () => startJob({}) }, "Find speakers")) : null,
      namesEditor(1));
    return;
  }

  const main = sp.filter((s) => s.share >= 5);
  pane.append(h("h2", {}, "Who talks first?"),
    h("p", { class: "lead" }, `${sp.length} voices were found. The one heard first gets the name you pick, the next one the other name. Listen to a line to check.`));
  const firstQ = h("div", { class: "first-q" }, h("span", { class: "q" }, "First to talk:"),
    names.map((n) => h("button", { class: "btn", onclick: () => {
      if (main[0]) assign[main[0].id] = n;
      const others = names.filter((x) => x !== n);
      if (main[1] && others.length === 1) assign[main[1].id] = others[0];
      drawVoices();
    } }, n)));
  const need = Math.min(2, main.length || 1);
  // no names yet: asking for them comes first, the quick question needs them
  pane.append(names.length ? firstQ : namesEditor(need));
  const voices = h("div", { class: "voices" });
  pane.append(voices);
  const saveRow = h("div", { class: "row", style: { marginTop: "18px" } });
  pane.append(saveRow, names.length ? namesEditor(need) : null);

  function drawVoices() {
    // colours follow the voices (first heard = first colour), the same as in the review
    const order = [...new Set(sp.map((x) => assign[x.id]).filter(Boolean))];
    fill(voices, ...sp.map((s, i) => {
      const nm = assign[s.id];
      const cls = spkClass(nm ? order.indexOf(nm) : -1);
      const sel = h("select", { class: "sel", onchange: (e) => { assign[s.id] = e.target.value; drawVoices(); } },
        h("option", { value: "" }, "— leave unnamed —"),
        names.map((n) => h("option", { value: n, selected: nm === n }, n)));
      return h("div", { class: "voice " + cls, style: { borderLeftColor: "var(--spk)" } },
        h("div", { class: "top" },
          h("span", { class: "who" }, nm || `Voice ${i + 1}`, h("span", {}, `  first at ${fmtT(s.first)}, ${fmtDur(s.seconds)} in total (${Math.round(s.share)}%)`)),
          s.share < 5 ? h("span", { class: "tag warn" }, "short: may be a slice of another voice") : null,
          sel),
        s.samples.map((x) => h("blockquote", {},
          h("button", { class: "bplay", title: `Play from ${fmtT(x.s)}`, onclick: () => playRange(x.s, x.e) }, icon("play")),
          h("span", {}, x.text))));
    }));
    const missing = sp.filter((s) => !assign[s.id]);
    fill(saveRow, 
      h("button", { class: "btn primary", onclick: save }, "Save names and review"),
      missing.length ? h("span", { class: "small muted" }, `${missing.length} unnamed voice${missing.length > 1 ? "s keep their" : " keeps its"} SPEAKER_ label; you can still fix single lines while reviewing.`) : null);
  }
  drawVoices();
}

function spkClass(i) { return i === 0 ? "spk-a" : i === 1 ? "spk-b" : i === 2 ? "spk-c" : "spk-x"; }

// plays one range of the recording's audio.wav (speaker samples)
let rangeEnd = null;
function ensureAudio() {
  const src = `/api/rec/${encodeURIComponent(S.rid)}/audio`;
  if (!audio.src.endsWith(src)) audio.src = src;
}
function playRange(s, e) {
  ensureAudio();
  if (!preview.paused) preview.pause();
  rangeEnd = e;
  audio.currentTime = Math.max(0, s - 0.1);
  audio.play().catch((err) => toast(err.message, true));
}
audio.addEventListener("timeupdate", () => {
  if (rangeEnd != null && audio.currentTime >= rangeEnd + 0.2) { audio.pause(); rangeEnd = null; }
});

// ------------------------------------------------------------------ step 4: review
const RV = {
  on: false, pieces: [], byId: {}, order: {}, edits: {}, pauses: {}, heard: new Set(), cov: {},
  rev: 0, dirty: false, saveTimer: null, undo: [], sel: null, editing: null, tokCache: new Map(),
  follow: true, curPiece: null, curWord: null, curBubble: null, names: [], suggestions: {},
  staticTL: null, raf: null, lastT: null,
};

function leaveReview() {
  if (!RV.on) return;
  flushSave();
  RV.on = false;
  cancelAnimationFrame(RV.raf);
  $(".dock") && $(".dock").remove();
}

const docLang = () => (S.doc && S.doc.language) || undefined;
const norm = (w) => w.toLocaleLowerCase(docLang()).normalize("NFC").replace(/[^\p{L}\p{N}]/gu, "");

function pieceText(p) { const e = RV.edits[p.id]; return e && e.text != null ? e.text : p.text; }
function pieceName(p) {
  const e = RV.edits[p.id];
  const cfg = S.doc.speakers_cfg;
  return (e && e.spk) || cfg.speakers[p.spk || ""] || cfg.default || p.spk || "?";
}
function nameIndex(n) { const i = RV.names.indexOf(n); return i < 0 ? 9 : i; }

// Map the (edited) text back onto the original words, so every word keeps a time:
// matched tokens take the original word's timing, new ones sit between their neighbours.
function tokens(p) {
  const text = pieceText(p);
  const key = p.id + "\u0000" + text;
  const hit = RV.tokCache.get(p.id);
  if (hit && hit.key === key) return hit.toks;
  const parts = text.split(/(\n)| +/).filter((x) => x);
  const words = p.words;
  const a = parts.map((x) => (x === "\n" ? "\n" : norm(x)));
  const b = words.map((w) => norm(w.w));
  const n = a.length, m = b.length;
  const dp = new Uint16Array((n + 1) * (m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) {
    dp[i * (m + 1) + j] = a[i] && a[i] === b[j] ? dp[(i + 1) * (m + 1) + j + 1] + 1
      : Math.max(dp[(i + 1) * (m + 1) + j], dp[i * (m + 1) + j + 1]);
  }
  const map = new Array(n).fill(null);
  for (let i = 0, j = 0; i < n && j < m;) {
    if (a[i] && a[i] === b[j]) { map[i] = j; i++; j++; }
    else if (dp[(i + 1) * (m + 1) + j] >= dp[i * (m + 1) + j + 1]) i++;
    else j++;
  }
  const edited = text !== p.text;
  const toks = parts.map((t, i) => {
    const o = map[i];
    if (o != null) {
      const w = words[o];
      return { t, s: w.s, e: w.e, orig: o, conf: w.c, changed: edited && t !== w.w };
    }
    return { t, s: null, e: null, orig: null, conf: null, changed: edited && t !== "\n", nl: t === "\n" };
  });
  // times for new tokens: between the matched neighbours
  for (let i = 0; i < toks.length; i++) {
    if (toks[i].s != null) continue;
    let k = i; while (k < toks.length && toks[k].s == null) k++;
    const from = i > 0 ? toks[i - 1].e : p.s;
    const to = k < toks.length ? toks[k].s : p.e;
    const span = Math.max(0, to - from) / (k - i);
    for (let x = i; x < k; x++) { toks[x].s = from + span * (x - i); toks[x].e = from + span * (x - i + 1); }
    i = k - 1;
  }
  RV.tokCache.set(p.id, { key, toks });
  return toks;
}

async function renderReview(body) {
  fill(body, h("div", { class: "pane" }, h("p", { class: "muted" }, h("span", { class: "spinner" }), " Loading the transcript…")));
  let doc;
  try { doc = await loadDoc(); } catch (e) { fill(body, h("div", { class: "pane" }, h("div", { class: "note bad" }, e.message))); return; }
  if (S.tab !== "review") return;
  const rv = doc.review || {};
  Object.assign(RV, {
    on: true, pieces: doc.pieces, byId: {}, order: {}, edits: rv.pieces || {}, pauses: rv.pauses || {},
    heard: new Set(rv.heard || []), cov: {}, rev: rv.rev || 0, dirty: false, undo: [], redo: [], sel: null, editing: null,
    tokCache: new Map(), follow: true, curPiece: null, curWord: null, curBubble: null, suggestions: {},
  });
  doc.pieces.forEach((p, i) => { RV.byId[p.id] = p; RV.order[p.id] = i; });
  computeNames();
  ensureAudio();

  const deck = h("section", { class: "deck", id: "deck" });
  const thread = h("div", { class: "thread", id: "thread" });
  const rail = h("aside", { class: "rail", id: "rail" });
  fill(body, deck, h("div", { class: "review-layout" }, thread, rail));
  buildDeck(deck);
  renderThread();
  renderRail();
  document.body.append(buildDock());
  renderDock();
  cancelAnimationFrame(RV.raf);
  RV.raf = requestAnimationFrame(tick);
  if (rv.fingerprint && rv.fingerprint !== doc.fingerprint) {
    thread.prepend(h("div", { class: "note warn banner" }, h("strong", {}, "The transcript changed after these edits were made."),
      h("p", {}, "Some corrections may sit on the wrong line. Check them, or start over from the Transcribe tab.")));
  }
  if (!doc.scenes_ready && doc.has_video) waitForScenes();
}

async function waitForScenes() {
  // scene detection runs in the background; pull the pause candidates in when it lands
  for (let i = 0; i < 120 && RV.on; i++) {
    await new Promise((r) => setTimeout(r, 3000));
    if (!RV.on) return;
    const d = await api.get(`/api/rec/${encodeURIComponent(S.rid)}/doc`).catch(() => null);
    if (d && d.scenes_ready) {
      S.doc.pauses = d.pauses; S.doc.scenes_ready = true;
      renderThread(); renderRail(); drawStaticTimeline();
      toast(`Video checked: ${d.pauses.length} possible pause${d.pauses.length === 1 ? "" : "s"}`);
      return;
    }
  }
}

function computeNames() {
  // order of first appearance decides left (first speaker) and right
  const seen = [];
  for (const p of RV.pieces) { const n = pieceName(p); if (!seen.includes(n)) seen.push(n); }
  const known = (S.state.settings.names || []);
  RV.names = seen.sort((x, y) => seen.indexOf(x) - seen.indexOf(y));
  for (const n of known) if (!RV.names.includes(n)) RV.names.push(n);
}

// ---------- deck: transport + timeline
function buildDeck(deck) {
  const playBtn = h("button", { class: "tbtn play", id: "playBtn", title: "Play / pause (Esc)", onclick: togglePlay }, icon("play"));
  const speed = h("button", { class: "tbtn", id: "speedBtn", title: "Speed (F3 slower, F4 faster)", onclick: () => stepSpeed(1) }, "1×");
  const follow = h("button", { class: "tbtn on", id: "followBtn", title: "Keep the playing line in view", onclick: () => setFollow(!RV.follow) }, "Follow");
  const clock = h("div", { class: "clock", id: "clock" });
  const tl = h("div", { class: "timeline", id: "timeline" }, h("canvas", { id: "tlCanvas" }), h("div", { class: "hover", hidden: true }));
  deck.append(
    h("div", { class: "transport" },
      h("button", { class: "tbtn", title: "Back 2 s (F1)", onclick: () => nudge(-2) }, icon("back")),
      playBtn,
      h("button", { class: "tbtn", title: "Forward 2 s (F2)", onclick: () => nudge(2) }, icon("fwd")),
      speed, clock, follow,
      h("div", { class: "deck-stats", id: "deckStats" })),
    tl);
  const seekAt = (ev) => {
    const r = tl.getBoundingClientRect();
    return Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width)) * S.doc.duration;
  };
  let dragging = false;
  tl.addEventListener("pointerdown", (ev) => { dragging = true; tl.setPointerCapture(ev.pointerId); seek(seekAt(ev)); });
  tl.addEventListener("pointermove", (ev) => {
    const hv = $(".hover", tl);
    const r = tl.getBoundingClientRect();
    hv.hidden = false; hv.style.left = (ev.clientX - r.left) + "px"; hv.textContent = fmtT(seekAt(ev));
    if (dragging) seek(seekAt(ev), false);
  });
  tl.addEventListener("pointerup", () => { dragging = false; });
  tl.addEventListener("pointerleave", () => { $(".hover", tl).hidden = true; });
  new ResizeObserver(() => drawStaticTimeline()).observe(tl);
  renderStats();
}

function speakerColor(name) {
  const cs = getComputedStyle(document.documentElement);
  const i = nameIndex(name);
  return cs.getPropertyValue(i === 0 ? "--a" : i === 1 ? "--b" : i === 2 ? "--c" : "--x").trim();
}

function drawStaticTimeline() {
  const c = $("#tlCanvas");
  if (!c || !S.doc) return;
  const dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = c.clientHeight;
  if (!W) return;
  const off = document.createElement("canvas");
  off.width = W * dpr; off.height = H * dpr;
  const g = off.getContext("2d");
  g.scale(dpr, dpr);
  const d = S.doc, dur = d.duration, pk = d.peaks, pps = d.peaks_per_s;
  const cs = getComputedStyle(document.documentElement);
  const splice = cs.getPropertyValue("--splice").trim();
  const muted = "rgba(154,161,178,.35)";
  // colour per column from the piece under it
  const mid = H / 2 + 4;
  let pi = 0;
  for (let x = 0; x < W; x++) {
    const t0 = x / W * dur, t1 = (x + 1) / W * dur;
    let v = 0;
    for (let k = Math.floor(t0 * pps); k < Math.min(pk.length, Math.ceil(t1 * pps)); k++) v = Math.max(v, pk[k]);
    while (pi < RV.pieces.length - 1 && RV.pieces[pi].e < t0) pi++;
    const p = RV.pieces[pi];
    let col = muted;
    if (p && p.s <= t1 && p.e >= t0) {
      const e = RV.edits[p.id];
      col = e && e.drop ? muted : speakerColor(pieceName(p));
    }
    const hgt = Math.max(1, v / 100 * (H - 14));
    g.fillStyle = col;
    g.fillRect(x, mid - hgt / 2, 1, hgt);
  }
  // pause markers
  for (const c2 of d.pauses) {
    const x = c2.t / dur * W;
    const decided = RV.pauses[c2.id];
    g.fillStyle = splice;
    g.globalAlpha = decided ? 0.35 : c2.confidence === "low" ? 0.6 : 1;
    g.beginPath(); g.moveTo(x - 5, 0); g.lineTo(x + 5, 0); g.lineTo(x, 7); g.closePath(); g.fill();
    g.fillRect(x - 0.5, 0, 1, H);
    g.globalAlpha = 1;
  }
  RV.staticTL = off;
  RV.heardDirty = true;
}

function drawTimeline(t) {
  const c = $("#tlCanvas");
  if (!c || !RV.staticTL) return;
  const dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = c.clientHeight;
  if (c.width !== W * dpr || c.height !== H * dpr) { c.width = W * dpr; c.height = H * dpr; }
  const g = c.getContext("2d");
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.clearRect(0, 0, c.width, c.height);
  g.drawImage(RV.staticTL, 0, 0);
  g.scale(dpr, dpr);
  // heard: a thin line along the bottom
  g.fillStyle = "rgba(94,194,122,.9)";
  for (const p of RV.pieces) if (RV.heard.has(p.id)) g.fillRect(p.s / S.doc.duration * W, H - 3, Math.max(1, (p.e - p.s) / S.doc.duration * W), 3);
  const x = t / S.doc.duration * W;
  g.fillStyle = "#fff";
  g.fillRect(x - 1, 0, 2, H);
}

function renderStats() {
  const el = $("#deckStats");
  if (!el || !S.doc) return;
  const total = RV.pieces.reduce((a, p) => a + (p.e - p.s), 0) || 1;
  const heard = RV.pieces.filter((p) => RV.heard.has(p.id)).reduce((a, p) => a + (p.e - p.s), 0);
  const edits = Object.values(RV.edits).filter((e) => e.text != null || e.drop || e.spk).length;
  const open = S.doc.pauses.filter((c) => !RV.pauses[c.id] && c.confidence !== "low").length;
  const unsure = RV.pieces.reduce((a, p) => a + (RV.edits[p.id] && RV.edits[p.id].text != null ? 0 : p.words.filter((w) => w.c != null && w.c < 0.5).length), 0);
  fill(el, 
    h("span", {}, "Heard ", h("b", {}, Math.round(heard / total * 100) + "%")),
    h("span", {}, h("b", {}, edits), edits === 1 ? " edit" : " edits"),
    S.doc.pauses.length ? h("span", {}, h("b", {}, open), open === 1 ? " pause to check" : " pauses to check") : null,
    unsure ? h("button", { title: "Jump to the next word Whisper was unsure about", onclick: nextUnsure }, `${unsure} unsure words: next`) : null,
    h("span", { class: "saved", id: "savedState" }, RV.dirty ? "Saving…" : RV.rev ? "Saved" : ""));
}

// ---------- the thread
function bubblesOf() {
  // consecutive pieces between two silences with one speaker form a bubble,
  // consecutive bubbles of one speaker form a turn
  const turns = [];
  let turn = null, bub = null;
  for (const p of RV.pieces) {
    const name = pieceName(p);
    if (!turn || turn.name !== name) { turn = { name, bubbles: [] }; turns.push(turn); bub = null; }
    if (!bub || bub.msg !== p.msg) { bub = { msg: p.msg, pieces: [], name }; turn.bubbles.push(bub); }
    bub.pieces.push(p);
  }
  for (const t of turns) for (const b of t.bubbles) {
    b.s = b.pieces[0].s; b.e = b.pieces[b.pieces.length - 1].e;
    const m = S.doc.messages.find((x) => x.i === b.msg);
    b.mixed = m && m.mixed;
  }
  return turns;
}

function renderThread() {
  const thread = $("#thread");
  if (!thread) return;
  const keepScroll = window.scrollY;
  const banner = $(".banner", thread);
  const turns = bubblesOf();
  const cards = [...S.doc.pauses].sort((a, b) => a.t - b.t);
  let ci = 0;
  const out = [];
  RV.bubbleEls = [];
  for (const t of turns) {
    while (ci < cards.length && cards[ci].t < t.bubbles[0].s + 0.3) out.push(pauseCard(cards[ci++]));
    const idx = nameIndex(t.name);
    const turnEl = h("div", { class: "turn " + (idx === 0 ? "" : "right") + " " + spkClass(idx) });
    const firstNonDropped = t.bubbles.flatMap((b) => b.pieces).find((p) => !(RV.edits[p.id] || {}).drop) || t.bubbles[0].pieces[0];
    turnEl.append(h("div", { class: "turn-head" },
      h("span", { class: "who" }, t.name),
      h("button", { class: "ts", title: "Play from here", onclick: () => playFrom(firstNonDropped.s) }, fmtT(firstNonDropped.s)),
      S.state.llm && S.state.llm.available ? h("button", { class: "ai", onclick: (e) => suggestTurn(t, e.currentTarget) }, "Suggest fixes") : null));
    for (const b of t.bubbles) {
      // pause cards that fall before this bubble
      while (ci < cards.length && cards[ci].t < b.s + 0.3) turnEl.append(pauseCard(cards[ci++]));
      const inner = [];  // pauses in the middle of this message go into its text, where they happen
      while (ci < cards.length && cards[ci].t < b.e - 0.3) inner.push(cards[ci++]);
      turnEl.append(bubbleEl(b, idx, inner));
    }
    out.push(turnEl);
  }
  while (ci < cards.length) { out.push(pauseCard(cards[ci])); ci++; }
  fill(thread, ...(banner ? [banner] : []), ...out.filter(Boolean));
  window.scrollTo({ top: keepScroll });
  RV.curBubble = null; RV.curWord = null; RV.curPiece = null;
  if (RV.sel) markSel(RV.sel);
}

function bubbleEl(b, idx, inner = []) {
  const allDropped = b.pieces.every((p) => (RV.edits[p.id] || {}).drop);
  const heard = b.pieces.every((p) => RV.heard.has(p.id));
  const canvas = h("canvas", { class: "bwave", title: "Click to play from there" });
  const el = h("div", { class: "bubble " + spkClass(idx) + (allDropped ? " all-dropped" : ""), "data-s": b.s, "data-e": b.e },
    h("div", { class: "bhead" },
      h("button", { class: "bplay", title: `Play from ${fmtT(b.s)}`,
        onclick: (ev) => { if (ev.currentTarget.closest(".bubble") === RV.curBubble && !audio.paused) audio.pause(); else playFrom(b.s); } }, icon("play")),
      canvas,
      h("span", { class: "btime" }, `${fmtT(b.s)}–${fmtT(b.e)}`, heard ? h("span", { class: "heard", title: "You have listened to all of it" }, "✓") : null),
      RV.names.length > 1 ? h("button", { class: "bswap", title: "Someone else says this part", onclick: () => swapBubble(b) },
        `→ ${nextName(b.name)}`) : null),
    b.mixed ? h("div", { class: "bnote" }, "Two voices were detected in this stretch; check who says what.") : null,
    textWithCards(b.pieces, inner));
  canvas.addEventListener("click", (ev) => {
    const r = canvas.getBoundingClientRect();
    playFrom(b.s + (ev.clientX - r.left) / r.width * (b.e - b.s));
  });
  el._b = b;
  RV.bubbleEls.push(el);
  requestAnimationFrame(() => drawBubbleWave(el, audio.currentTime));
  return el;
}

function textWithCards(pieces, cards) {
  const out = [];
  let cur = [], ci = 0;
  for (const p of pieces) {
    // a card goes before the first line it does not come after (word timings can spill
    // a line's start back into the silence before it, so compare with its middle)
    const due = () => ci < cards.length && cards[ci].t <= (p.s + p.e) / 2;
    if (due()) {
      if (cur.length) out.push(h("div", { class: "btext" }, cur));
      while (due()) out.push(pauseCard(cards[ci++]));
      cur = [];
    }
    cur.push(cur.length ? " " : "", pieceEl(p));
  }
  out.push(h("div", { class: "btext" }, cur));
  while (ci < cards.length) out.push(pauseCard(cards[ci++]));
  return out;
}

function nextName(n) {
  const i = RV.names.indexOf(n);
  return RV.names[(i + 1) % RV.names.length] === n ? RV.names[0] : RV.names[(i + 1) % RV.names.length];
}

function drawBubbleWave(el, t) {
  const c = el.querySelector(".bwave");
  if (!c) return;
  const b = el._b;
  const dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = c.clientHeight;
  if (!W) return;
  if (c.width !== W * dpr) { c.width = W * dpr; c.height = H * dpr; }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const col = getComputedStyle(el).getPropertyValue("--spk").trim();
  const pk = S.doc.peaks, pps = S.doc.peaks_per_s;
  const bars = Math.max(8, Math.floor(W / 5));
  const dur = b.e - b.s;
  for (let i = 0; i < bars; i++) {
    const t0 = b.s + i / bars * dur, t1 = b.s + (i + 1) / bars * dur;
    let v = 0;
    for (let k = Math.floor(t0 * pps); k < Math.min(pk.length, Math.ceil(t1 * pps)); k++) v = Math.max(v, pk[k]);
    const hh = Math.max(3, v / 100 * H);
    g.fillStyle = col;
    g.globalAlpha = t >= t1 ? 1 : t >= t0 ? 0.8 : 0.32;
    const x = i * 5;
    g.beginPath();
    if (g.roundRect) g.roundRect(x, (H - hh) / 2, 3, hh, 1.5); else g.rect(x, (H - hh) / 2, 3, hh);
    g.fill();
  }
  g.globalAlpha = 1;
}

function pieceEl(p) {
  const e = RV.edits[p.id] || {};
  const span = h("span", { class: "pc" + (e.drop ? " drop" : "") + (e.spk ? " other-spk" : ""), "data-id": p.id });
  if (RV.editing === p.id) {
    span.append(editorFor(p));
    return span;
  }
  const toks = tokens(p);
  toks.forEach((t, i) => {
    if (t.nl) { span.append(h("br")); return; }
    if (i && !toks[i - 1].nl) span.append(" ");
    const isMarker = /^\[.*\]$/.test(t.t);
    span.append(h("span", {
      class: "w" + (t.orig != null && !t.changed && t.conf != null && t.conf < 0.5 ? " unsure" : "") + (t.changed ? " changed" : "") + (isMarker ? " marker" : ""),
      "data-s": t.s, "data-i": i,
      title: t.orig != null && t.conf != null && t.conf < 0.5 ? `Whisper was unsure (${Math.round(t.conf * 100)}%)` : null,
    }, t.t));
  });
  const sug = RV.suggestions[p.id];
  if (sug) span.append(suggestionEl(p, sug));
  return span;
}

function rerenderPiece(id) {
  const old = $(`.pc[data-id="${CSS.escape(id)}"]`);
  const p = RV.byId[id];
  if (!old || !p) return renderThread();
  const nu = pieceEl(p);
  old.replaceWith(nu);
  if (RV.sel === id) nu.classList.add("sel");
  RV.curWord = null; RV.curPiece = null;
  const bub = nu.closest(".bubble");
  if (bub) bub.classList.toggle("all-dropped", bub._b.pieces.every((q) => (RV.edits[q.id] || {}).drop));
  return nu;
}

// clicks in the thread: a word plays from that word; double click edits the piece
document.addEventListener("click", (ev) => {
  if (!RV.on) return;
  const w = ev.target.closest(".thread .w");
  if (!w) return;
  const pc = w.closest(".pc");
  select(pc.dataset.id);
  playFrom(parseFloat(w.dataset.s) - 0.12);
});
document.addEventListener("dblclick", (ev) => {
  if (!RV.on) return;
  const w = ev.target.closest(".thread .w");
  if (!w) return;
  ev.preventDefault();
  const pc = w.closest(".pc");
  startEdit(pc.dataset.id, parseInt(w.dataset.i, 10));
});

function select(id) {
  RV.sel = id;
  markSel(id);
  renderDock();
}
function markSel(id) {
  $$(".pc.sel").forEach((x) => x.classList.remove("sel"));
  const el = id && $(`.pc[data-id="${CSS.escape(id)}"]`);
  if (el) el.classList.add("sel");
}

// ---------- editing
function editorFor(p) {
  const ta = h("textarea", { class: "pc-edit", rows: 1, spellcheck: true, lang: S.doc.language || null });
  ta.value = pieceText(p);
  const fit = () => { ta.style.height = "auto"; ta.style.height = ta.scrollHeight + "px"; };
  ta.addEventListener("input", fit);
  ta.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey && !ev.ctrlKey) { ev.preventDefault(); commitEdit(p.id, ta.value); }
    else if (ev.key === "Enter" && ev.ctrlKey) { ev.preventDefault(); playFrom(p.s - 0.1); }
    else if (ev.key === "Tab") {
      ev.preventDefault();
      const next = neighbour(p.id, ev.shiftKey ? -1 : 1);
      commitEdit(p.id, ta.value, next);
    }
  });
  ta.addEventListener("blur", () => { if (RV.editing === p.id) setTimeout(() => RV.editing === p.id && document.activeElement !== ta && commitEdit(p.id, ta.value), 120); });
  requestAnimationFrame(fit);
  return ta;
}

function neighbour(id, dir) {
  let i = RV.order[id] + dir;
  while (i >= 0 && i < RV.pieces.length && (RV.edits[RV.pieces[i].id] || {}).drop) i += dir;
  return i >= 0 && i < RV.pieces.length ? RV.pieces[i].id : null;
}

function startEdit(id, tokIndex) {
  if (RV.editing && RV.editing !== id) {
    const ta = $(".pc-edit");
    if (ta) commitEdit(RV.editing, ta.value, null, true);
  }
  RV.editing = id;
  select(id);
  const el = rerenderPiece(id);
  const ta = el && el.querySelector("textarea");
  if (!ta) return;
  ta.focus();
  // caret at the double-clicked word
  let pos = ta.value.length;
  if (tokIndex != null) {
    const toks = tokens(RV.byId[id]);
    let at = 0;
    for (let i = 0; i < toks.length && i < tokIndex; i++) at += toks[i].t.length + (toks[i].nl || (toks[i + 1] && toks[i + 1].nl) ? 0 : 1);
    pos = Math.min(ta.value.length, at);
  }
  ta.setSelectionRange(pos, pos);
}

function commitEdit(id, value, nextId, silent) {
  if (RV.editing !== id) return;
  RV.editing = null;
  const p = RV.byId[id];
  const clean = value.split("\n").map((l) => l.replace(/\s+/g, " ").trim()).join("\n").replace(/\n{2,}/g, "\n").trim();
  if (clean !== pieceText(p)) setEdit(id, { text: clean === p.text ? undefined : clean });
  else rerenderPiece(id);
  if (!silent && nextId) startEdit(nextId, 0);
  else if (!silent) renderDock();
}

function snapshot() { return JSON.stringify({ edits: RV.edits, pauses: RV.pauses }); }
function pushUndo() { RV.undo.push(snapshot()); if (RV.undo.length > 300) RV.undo.shift(); RV.redo = []; }
function undo() {
  if (!RV.undo.length) return toast("Nothing to undo");
  RV.redo.push(snapshot());
  const s = JSON.parse(RV.undo.pop());
  RV.edits = s.edits; RV.pauses = s.pauses;
  afterBigChange();
  toast("Undone");
}
function redoEdit() {
  if (!RV.redo.length) return;
  RV.undo.push(snapshot());
  const s = JSON.parse(RV.redo.pop());
  RV.edits = s.edits; RV.pauses = s.pauses;
  afterBigChange();
}
function afterBigChange() {
  RV.tokCache.clear();
  computeNames();
  renderThread(); renderRail(); drawStaticTimeline(); renderDock(); renderStats();
  scheduleSave();
}

// patch: {text, drop, spk}; undefined removes the key
function setEdit(id, patch, { quiet = false, noUndo = false } = {}) {
  if (!noUndo) pushUndo();
  const p = RV.byId[id];
  const e = { ...(RV.edits[id] || {}) };
  for (const [k, v] of Object.entries(patch)) {
    if (v === undefined || v === false || v === null || (k === "text" && v === p.text)) delete e[k]; else e[k] = v;
  }
  if (Object.keys(e).length) RV.edits[id] = e; else delete RV.edits[id];
  if (!quiet) {
    if ("spk" in patch) { computeNames(); renderThread(); drawStaticTimeline(); }
    else { rerenderPiece(id); if ("drop" in patch) drawStaticTimeline(); }
    renderDock(); renderStats();
  }
  scheduleSave();
}

function swapBubble(b) {
  const to = nextName(b.name);
  pushUndo();
  const cfgName = (p) => S.doc.speakers_cfg.speakers[p.spk || ""] || S.doc.speakers_cfg.default || p.spk;
  for (const p of b.pieces) setEdit(p.id, { spk: cfgName(p) === to ? undefined : to }, { quiet: true, noUndo: true });
  computeNames(); renderThread(); drawStaticTimeline(); renderRail(); renderStats();
  toast(`The part at ${fmtT(b.s)} is now ${to}'s`);
}

// ---------- pause cards
const KIND_TEXT = {
  jump: (s) => `The picture jumps at ${fmtTd(s.t)}${s.score >= 0.1 ? " (a big change)" : ""}.`,
  repeat: (s) => [`Words from ${fmtT(s.earlier[0])}–${fmtT(s.earlier[1])} are said again at ${fmtT(s.later[0])}–${fmtT(s.later[1])}: `, h("q", {}, s.text + "…")],
  midstart: (s) => [`At ${fmtT(s.t)} speech resumes mid-sentence after a silence: `, h("q", {}, s.text + "…"),
    s.weak ? " The sentence before it doesn't end, so it may simply go on." : ""],
};

function pauseCard(c) {
  const decided = RV.pauses[c.id];
  const repeat = c.signals.find((s) => s.kind === "repeat");
  const mid = c.signals.find((s) => s.kind === "midstart");
  const jump = c.signals.find((s) => s.kind === "jump");
  const markerFix = (mid && mid.fix) || (jump && jump.fix);
  const title = c.confidence === "high" ? "The recording was probably paused here" : "Possibly paused here";
  const decisionText = { later: "Kept the later take", earlier: "Kept the earlier take", marker: "Marked with […]", dismissed: "Not a pause" };
  const wrap = h("div", { class: "splice" + (decided ? " decided collapsed" : c.confidence === "low" ? " collapsed" : ""), id: "card-" + c.id });

  const acts = h("div", { class: "acts" },
    repeat ? [
      h("button", { class: "btn sm primary", onclick: () => applyRepeat(c, repeat, "later"), title: "Remove the repeated words from the first take" }, "Keep the later take"),
      h("button", { class: "btn sm", onclick: () => applyRepeat(c, repeat, "earlier"), title: "Remove the repeated words from the second take" }, "Keep the earlier take"),
      h("button", { class: "btn sm", onclick: () => playRange(repeat.earlier[0] - 0.3, repeat.earlier[1]) }, icon("play"), "First take"),
      h("button", { class: "btn sm", onclick: () => playRange(repeat.later[0] - 0.3, repeat.later[1]) }, icon("play"), "Second take"),
    ] : null,
    markerFix ? h("button", { class: "btn sm" + (repeat ? "" : " primary"), onclick: () => applyMarker(c, markerFix) }, "Insert […]") : null,
    !repeat ? h("button", { class: "btn sm", onclick: () => playFrom(Math.max(0, c.t - 4)) }, icon("play"), "Listen around it") : null,
    h("button", { class: "btn sm quiet", onclick: () => decide(c, "dismissed") }, "Not a pause"));

  if (wrap.classList.contains("collapsed")) {
    const summary = decided ? decisionText[decided] : [...new Set(c.signals.map((s) => s.kind === "jump" ? "the picture jumps" : s.kind === "repeat" ? "words repeat" : "starts mid-sentence"))].join(", ");
    wrap.append(h("div", { class: "card" },
      h("span", { class: "what" }, h("b", {}, decided ? "Pause " : "Possible pause "), h("span", { class: "num muted" }, fmtT(c.t) + "  "), summary),
      decided ? h("button", { class: "btn sm quiet", onclick: () => undecide(c) }, "Undo") :
        h("button", { class: "btn sm quiet", onclick: () => { wrap.classList.remove("collapsed"); wrap.replaceWith(expandedCard(c, title, acts)); } }, "Look")));
    return wrap;
  }
  return expandedCard(c, title, acts, wrap);
}

function expandedCard(c, title, acts, wrap) {
  wrap = wrap || h("div", { class: "splice", id: "card-" + c.id });
  wrap.classList.remove("collapsed");
  const ft = c.frame_t;
  const frames = S.doc.has_video ? h("div", { class: "frames" },
    [[ft - 0.25, "just before"], [ft + 0.25, "just after"]].map(([t, cap]) => h("figure", { onclick: () => lightbox(ft) },
      h("img", { loading: "lazy", alt: `Video frame at ${fmtTd(t)}`, src: `/api/rec/${encodeURIComponent(S.rid)}/frame?t=${Math.max(0, t).toFixed(2)}` }),
      h("figcaption", {}, `${fmtTd(t)} ${cap}`)))) : null;
  fill(wrap, h("div", { class: "card" },
    h("h4", {}, title, h("span", { class: "t" }, fmtT(c.t) + (c.t_end - c.t > 1 ? `–${fmtT(c.t_end)}` : "")),
      h("span", { class: "conf" }, { high: "likely", medium: "maybe", low: "unlikely" }[c.confidence])),
    h("ul", {}, c.signals.map((s) => h("li", {}, KIND_TEXT[s.kind](s)))),
    frames, acts));
  return wrap;
}

function lightbox(t) {
  const lb = $("#lightbox");
  fill(lb, h("div", { class: "pair" }, [[t - 0.25, "before"], [t + 0.25, "after"]].map(([x, cap]) =>
    h("figure", {}, h("img", { src: `/api/rec/${encodeURIComponent(S.rid)}/frame?t=${Math.max(0, x).toFixed(2)}`, alt: "" }),
      h("figcaption", {}, `${fmtTd(x)} ${cap}`)))));
  lb.hidden = false;
  lb.onclick = () => { lb.hidden = true; };
}

function decide(c, how) {
  pushUndo();
  RV.pauses[c.id] = how;
  afterBigChange();
}
function undecide(c) {
  // the fix itself is undone through the undo stack; this only reopens the card
  pushUndo();
  delete RV.pauses[c.id];
  afterBigChange();
  toast("Card reopened. Ctrl+Z undoes the text change too.");
}

function removeWords(pid, set) {
  const p = RV.byId[pid];
  if (!p) return;
  const toks = tokens(p);
  const kept = toks.filter((t) => t.orig == null || !set.has(t.orig));
  const text = kept.map((t) => t.t).join(" ").replace(/ ?\n ?/g, "\n").trim();
  if (!text || kept.every((t) => /^\[.*\]$/.test(t.t))) setEdit(pid, { drop: true }, { quiet: true, noUndo: true });
  else setEdit(pid, { text }, { quiet: true, noUndo: true });
}

function applyRepeat(c, sig, keep) {
  pushUndo();
  const refs = keep === "later" ? sig.fix.earlier : sig.fix.later;
  for (const r of refs) removeWords(r.piece, new Set(r.words));
  // "But I said," + "If only…": the sentence now runs on, so no capital
  const lastRef = refs[refs.length - 1];
  const before = RV.byId[refs[0].piece] && pieceText(RV.byId[refs[0].piece]);
  const nextId = lastRef && neighbour(lastRef.piece, 1);
  const prevKept = [...refs].reverse().map((r) => RV.byId[r.piece]).find((p) => p && !(RV.edits[p.id] || {}).drop);
  if (before != null && prevKept && nextId && /,$/.test(pieceText(prevKept))) {
    const nx = RV.byId[nextId];
    const t = pieceText(nx);
    if (/^\p{Lu}\p{Ll}/u.test(t)) setEdit(nextId, { text: t[0].toLocaleLowerCase(docLang()) + t.slice(1) }, { quiet: true, noUndo: true });
  }
  RV.pauses[c.id] = keep;
  afterBigChange();
  toast(keep === "later" ? "Removed the repeated words from the first take" : "Removed the repeated words from the second take");
}

function applyMarker(c, fix) {
  pushUndo();
  const p = RV.byId[fix.piece];
  if (p) {
    const toks = tokens(p);
    let at = toks.findIndex((t) => t.orig != null && t.orig >= fix.word);
    if (at < 0) at = 0;
    if (!(at > 0 && toks[at - 1].t === "[…]") && toks[at].t !== "[…]") {
      const parts = toks.map((t) => t.t);
      parts.splice(at, 0, "[…]");
      setEdit(p.id, { text: parts.join(" ").replace(/ ?\n ?/g, "\n") }, { quiet: true, noUndo: true });
    }
  }
  RV.pauses[c.id] = "marker";
  afterBigChange();
}

function renderRail() {
  const rail = $("#rail");
  if (!rail || !S.doc) return;
  const cards = S.doc.pauses;
  fill(rail, 
    h("h3", {}, "Possible pauses"),
    cards.length ? h("ol", {}, cards.map((c) => {
      const d = RV.pauses[c.id];
      return h("li", { class: d ? "decided" : c.confidence === "low" ? "" : "open" },
        h("button", { onclick: () => { const el = $("#card-" + c.id); if (el) el.scrollIntoView({ block: "center", behavior: "smooth" }); } },
          h("span", { class: "t" }, fmtT(c.t)),
          h("span", {}, d ? { later: "kept later take", earlier: "kept earlier take", marker: "marked […]", dismissed: "not a pause" }[d]
            : { high: "likely a pause", medium: "maybe a pause", low: "unlikely" }[c.confidence])));
    })) : h("p", { class: "small muted" }, S.doc.scenes_ready || !S.doc.has_video ? "None found." : "Checking the video…"),
    h("h3", {}, "Keys"),
    h("div", { class: "keys" },
      h("kbd", {}, "Esc"), h("span", {}, "play / pause, also while typing"),
      h("kbd", {}, "F1 F2"), h("span", {}, "back / forward 2 s"),
      h("kbd", {}, "F3 F4"), h("span", {}, "slower / faster"),
      h("span", {}, "click"), h("span", {}, "play from that word"),
      h("span", {}, "double-click"), h("span", {}, "edit that line"),
      h("kbd", {}, "Enter"), h("span", {}, "done editing (Shift+Enter: new line)"),
      h("kbd", {}, "Tab"), h("span", {}, "save, edit the next line"),
      h("kbd", {}, "Ctrl+Enter"), h("span", {}, "replay the line being edited"),
      h("kbd", {}, "Ctrl+Z"), h("span", {}, "undo")));
}

// ---------- dock
function buildDock() { return h("div", { class: "dock", id: "dock", role: "toolbar", "aria-label": "Selected line" }); }
function renderDock() {
  const dock = $("#dock");
  if (!dock) return;
  const p = RV.sel && RV.byId[RV.sel];
  if (!p) {
    fill(dock, h("span", { class: "lbl hint" }, "Click a word to play from it, double-click to correct it"));
    return;
  }
  const e = RV.edits[p.id] || {};
  const name = pieceName(p);
  const sel = h("select", { title: "Who says this line", onchange: (ev) => {
    const v = ev.target.value;
    const base = S.doc.speakers_cfg.speakers[p.spk || ""] || S.doc.speakers_cfg.default || p.spk;
    setEdit(p.id, { spk: v === base ? undefined : v });
  } }, RV.names.map((n) => h("option", { value: n, selected: n === name }, n)));
  fill(dock, 
    h("span", { class: "lbl" }, fmtT(p.s)),
    h("button", { class: "tbtn", onclick: () => playFrom(p.s - 0.1) }, icon("play"), "Play line"),
    h("button", { class: "tbtn", onclick: () => RV.editing === p.id ? null : startEdit(p.id) }, "Edit"),
    h("button", { class: "tbtn", onclick: () => setEdit(p.id, { drop: !e.drop }) }, e.drop ? "Keep line" : "Leave out"),
    sel,
    e.text != null ? h("button", { class: "tbtn", title: p.text, onclick: () => setEdit(p.id, { text: undefined }) }, "Back to Whisper's text") : null,
    RV.undo.length ? h("button", { class: "tbtn", onclick: undo }, "Undo") : null);
}

// ---------- local LLM suggestions
async function suggestTurn(t, btn) {
  const pieces = t.bubbles.flatMap((b) => b.pieces).filter((p) => !(RV.edits[p.id] || {}).drop);
  const prevIdx = RV.order[pieces[0].id];
  const context = RV.pieces.slice(Math.max(0, prevIdx - 3), prevIdx).map(pieceText).join(" ");
  btn.textContent = "Asking the local model…"; btn.disabled = true;
  try {
    const d = await api.send("POST", `/api/rec/${encodeURIComponent(S.rid)}/suggest`,
      { items: pieces.map((p) => ({ id: p.id, text: pieceText(p) })), context });
    for (const s of d.suggestions) RV.suggestions[s.id] = s.text;
    for (const s of d.suggestions) rerenderPiece(s.id);
    toast(d.suggestions.length ? `${d.suggestions.length} suggestion${d.suggestions.length > 1 ? "s" : ""}; accept or skip each one` : "No changes suggested");
  } catch (e) { toast(e.message, true); }
  btn.textContent = "Suggest fixes"; btn.disabled = false;
}

function diffWords(a, b) {
  const x = a.split(/\s+/).filter(Boolean), y = b.split(/\s+/).filter(Boolean);
  const n = x.length, m = y.length;
  const dp = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) dp[i][j] = x[i] === y[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const out = [];
  let i = 0, j = 0;
  while (i < n || j < m) {
    if (i < n && j < m && x[i] === y[j]) { out.push(x[i]); i++; j++; }
    else if (i < n && (j >= m || dp[i + 1][j] >= dp[i][j + 1])) { out.push(h("del", {}, x[i])); i++; }
    else { out.push(h("ins", {}, y[j])); j++; }
  }
  return out.flatMap((w, k) => (k ? [" ", w] : [w]));
}

function suggestionEl(p, text) {
  return h("span", { class: "suggest", style: { display: "block" } },
    h("span", { class: "diff", style: { display: "block" } }, diffWords(pieceText(p), text)),
    h("button", { class: "btn sm primary", onclick: () => { delete RV.suggestions[p.id]; setEdit(p.id, { text }); } }, "Accept"),
    " ", h("button", { class: "btn sm quiet", onclick: () => { delete RV.suggestions[p.id]; rerenderPiece(p.id); } }, "Skip"));
}

// ---------- playback
function playFrom(t) {
  ensureAudio();
  rangeEnd = null;
  if (!preview.paused) preview.pause();
  audio.currentTime = Math.max(0, t);
  RV.lastT = null;
  setFollow(true);
  audio.play().catch((e) => toast(e.message, true));
}
function seek(t, resumeFollow = true) {
  ensureAudio();
  audio.currentTime = Math.max(0, Math.min(t, S.doc ? S.doc.duration : t));
  RV.lastT = null;
  if (resumeFollow) setFollow(true);
}
function togglePlay() {
  ensureAudio();
  rangeEnd = null;
  if (audio.paused) audio.play().catch((e) => toast(e.message, true)); else audio.pause();
}
function nudge(d) { seek(audio.currentTime + d, false); }
const SPEEDS = [0.6, 0.75, 0.9, 1, 1.15, 1.3, 1.5, 1.75];
function stepSpeed(dir) {
  let i = SPEEDS.indexOf(audio.playbackRate);
  if (i < 0) i = 3;
  i = dir > 0 ? (i + 1) % SPEEDS.length : Math.max(0, i - 1);
  audio.playbackRate = SPEEDS[i];
  const b = $("#speedBtn"); if (b) b.textContent = SPEEDS[i] + "×";
}
function setFollow(on) {
  RV.follow = on;
  const b = $("#followBtn"); if (b) b.classList.toggle("on", on);
}
window.addEventListener("wheel", () => { if (RV.on && !audio.paused) setFollow(false); }, { passive: true });
window.addEventListener("touchmove", () => { if (RV.on && !audio.paused) setFollow(false); }, { passive: true });
audio.addEventListener("play", () => { const b = $("#playBtn"); if (b) fill(b, icon("pause")); });
audio.addEventListener("pause", () => { const b = $("#playBtn"); if (b) fill(b, icon("play")); scheduleSave(); });

function nextUnsure() {
  const t = audio.currentTime + 0.3;
  for (const p of RV.pieces) {
    if (p.e < t || (RV.edits[p.id] || {}).text != null || (RV.edits[p.id] || {}).drop) continue;
    const w = p.words.find((x) => x.c != null && x.c < 0.5 && x.s > t);
    if (w) {
      select(p.id);
      const el = $(`.pc[data-id="${CSS.escape(p.id)}"]`);
      if (el) el.scrollIntoView({ block: "center", behavior: "smooth" });
      playFrom(w.s - 1.2);
      return;
    }
  }
  toast("No more unsure words after this point");
}

function pieceAt(t) {
  const ps = RV.pieces;
  let lo = 0, hi = ps.length - 1, best = -1;
  while (lo <= hi) { const mid = (lo + hi) >> 1; if (ps[mid].s <= t) { best = mid; lo = mid + 1; } else hi = mid - 1; }
  if (best >= 0 && t <= ps[best].e + 0.25) return ps[best];
  return null;
}

function tick() {
  if (!RV.on) return;
  RV.raf = requestAnimationFrame(tick);
  RV.lastFrame = performance.now();
  update();
}
// some embedded or background views throttle animation frames; keep the page in step anyway
setInterval(() => { if (RV.on && performance.now() - (RV.lastFrame || 0) > 300) update(); }, 250);

function update() {
  const t = audio.currentTime || 0;
  const clock = $("#clock");
  if (clock) clock.innerHTML = `${fmtTd(t)} <span>/ ${fmtT(S.doc.duration)}</span>`;
  drawTimeline(t);
  // heard coverage
  if (!audio.paused && RV.lastT != null && t > RV.lastT && t - RV.lastT < 1.5) {
    for (const p of RV.pieces) {
      if (p.e < RV.lastT || p.s > t) continue;
      const add = Math.min(p.e, t) - Math.max(p.s, RV.lastT);
      if (add <= 0) continue;
      RV.cov[p.id] = (RV.cov[p.id] || 0) + add;
      if (!RV.heard.has(p.id) && RV.cov[p.id] >= 0.75 * (p.e - p.s)) {
        RV.heard.add(p.id);
        RV.heardChanged = true;
      }
    }
  }
  RV.lastT = audio.paused ? null : t;
  if (RV.heardChanged && audio.paused) { RV.heardChanged = false; renderStats(); scheduleSave(); }
  // current word + bubble
  const p = pieceAt(t);
  const pid = p ? p.id : null;
  let wEl = null;
  if (p && RV.editing !== pid) {
    const pc = $(`.pc[data-id="${CSS.escape(pid)}"]`);
    if (pc) {
      const ws = pc.querySelectorAll(".w");
      for (let i = ws.length - 1; i >= 0; i--) if (parseFloat(ws[i].dataset.s) <= t + 0.02) { wEl = ws[i]; break; }
    }
  }
  if (wEl !== RV.curWord) {
    if (RV.curWord) RV.curWord.classList.remove("now");
    if (wEl && !audio.paused) wEl.classList.add("now");
    RV.curWord = wEl;
  }
  if (audio.paused && RV.curWord) { RV.curWord.classList.remove("now"); RV.curWord = null; }
  let bEl = null;
  for (const b of RV.bubbleEls || []) if (b._b.s - 0.2 <= t && t <= b._b.e + 0.3) { bEl = b; break; }
  const playing = !audio.paused;
  if (bEl !== RV.curBubble || playing !== RV.curPlaying) {
    if (RV.curBubble && RV.curBubble !== bEl) {
      RV.curBubble.classList.remove("active"); drawBubbleWave(RV.curBubble, t);
      fill(RV.curBubble.querySelector(".bplay"), icon("play"));
    }
    if (bEl) { bEl.classList.add("active"); fill(bEl.querySelector(".bplay"), icon(playing ? "pause" : "play")); }
    RV.curBubble = bEl; RV.curPlaying = playing;
  }
  if (bEl && !audio.paused) drawBubbleWave(bEl, t);
  if (RV.follow && !audio.paused && (wEl || bEl) && !RV.editing) {
    const target = wEl || bEl;
    const r = target.getBoundingClientRect();
    const top = 190, bottom = window.innerHeight - 110;
    if (r.top < top || r.bottom > bottom) window.scrollBy({ top: r.top - (top + bottom) / 2, behavior: "smooth" });
  }
  if (!audio.paused && Math.random() < 0.02) renderStats();
}

// ---------- saving
function scheduleSave() {
  if (!RV.on) return;
  RV.dirty = true;
  const s = $("#savedState"); if (s) s.textContent = "Saving…";
  clearTimeout(RV.saveTimer);
  RV.saveTimer = setTimeout(flushSave, 700);
}
async function flushSave() {
  clearTimeout(RV.saveTimer);
  if (!RV.dirty || !S.doc) return;
  RV.dirty = false;
  const rid = S.doc._rid;
  try {
    // keepalive: the save still lands when the tab is being closed
    const d = await jfetch(`/api/rec/${encodeURIComponent(rid)}/review`, {
      method: "PUT", headers: { "Content-Type": "application/json" }, keepalive: true,
      body: JSON.stringify({ prev_rev: RV.rev,
        review: { pieces: RV.edits, pauses: RV.pauses, heard: [...RV.heard], fingerprint: S.doc.fingerprint } }),
    });
    RV.rev = d.rev;
    const s = $("#savedState"); if (s) s.textContent = "Saved";
  } catch (e) {
    RV.dirty = true;
    toast(e.status === 409 ? e.message : "Could not save: " + e.message, true);
  }
}
window.addEventListener("pagehide", () => { if (RV.on && RV.dirty) flushSave(); });

// ---------- keys
document.addEventListener("keydown", (ev) => {
  const inText = ev.target.matches("input, textarea, select");
  if (ev.key === "Escape" && !$("#lightbox").hidden) { $("#lightbox").hidden = true; return; }
  if (!RV.on) {
    if (ev.key === "Escape" && !audio.paused) audio.pause();
    return;
  }
  if (ev.key === "Escape") { ev.preventDefault(); togglePlay(); return; }
  if (ev.key === "F1") { ev.preventDefault(); nudge(-2); return; }
  if (ev.key === "F2") { ev.preventDefault(); nudge(2); return; }
  if (ev.key === "F3") { ev.preventDefault(); stepSpeed(-1); return; }
  if (ev.key === "F4") { ev.preventDefault(); stepSpeed(1); return; }
  if (inText) return;
  if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "z") { ev.preventDefault(); ev.shiftKey ? redoEdit() : undo(); return; }
  if (ev.key === "Enter" && RV.sel) { ev.preventDefault(); startEdit(RV.sel); return; }
  if (ev.key === " " && !ev.target.closest("button")) { ev.preventDefault(); togglePlay(); }
});

// ------------------------------------------------------------------ step 5: export
async function renderExport(body) {
  const pane = h("section", { class: "pane" });
  fill(body, pane);
  pane.append(h("p", { class: "muted" }, h("span", { class: "spinner" }), " Putting the transcript together…"));
  let d, doc;
  try {
    doc = await loadDoc();
    d = await api.get(`/api/rec/${encodeURIComponent(S.rid)}/clean`);
  } catch (e) { fill(pane, h("div", { class: "note bad" }, e.message)); return; }
  if (S.tab !== "export") return;
  const rv = doc.review || {};
  const edits = rv.pieces || {};
  const decided = rv.pauses || {};
  const heard = new Set(rv.heard || []);
  const total = doc.pieces.reduce((a, p) => a + (p.e - p.s), 0) || 1;
  const heardPct = Math.round(doc.pieces.filter((p) => heard.has(p.id)).reduce((a, p) => a + (p.e - p.s), 0) / total * 100);
  const openPauses = doc.pauses.filter((c) => !decided[c.id] && c.confidence !== "low");
  const nEdits = Object.keys(edits).length;
  const checks = [
    [d.unnamed.length === 0, d.unnamed.length ? `Unnamed speakers: ${d.unnamed.join(", ")}` : "Every turn has a name"],
    [openPauses.length === 0, openPauses.length ? `${openPauses.length} possible pause${openPauses.length > 1 ? "s" : ""} not decided yet` : "Pauses checked"],
    [heardPct >= 95, `Listened to ${heardPct}% of the speech`],
    [true, `${nEdits} line${nEdits === 1 ? "" : "s"} corrected, ${d.turns} speaker turn${d.turns === 1 ? "" : "s"}`],
  ];
  const stateNote = {
    current: h("div", { class: "note good" }, h("strong", {}, "transcript_clean.txt is saved and matches your edits.")),
    stale: h("div", { class: "note warn" }, h("strong", {}, "You have edited since the last save.")),
    other: h("div", { class: "note warn" }, h("strong", {}, "There is already a transcript_clean.txt that was not made here."),
      h("p", {}, "Saving moves it to previous/ inside the work folder, so it is kept.")),
  }[d.state] || null;

  const view = { which: "new" };
  const paper = h("div", { class: "paper" });
  const drawPaper = () => {
    const txt = view.which === "new" ? d.text : d.existing || "";
    fill(paper, ...txt.split("\n").map((line, i) => {
      const m = line.match(/^(\[\d[\d:]*\] [^:]+:)$/);
      return [i ? "\n" : "", m ? h("span", { class: "th" }, line) : line];
    }));
  };
  const tabs = d.existing != null && d.existing !== d.text ? h("div", { class: "tabs2" },
    h("button", { "aria-pressed": "true", onclick: (e) => { view.which = "new"; setPressed(e); drawPaper(); } }, "From your edits"),
    h("button", { "aria-pressed": "false", onclick: (e) => { view.which = "old"; setPressed(e); drawPaper(); } }, "Current file")) : null;
  function setPressed(e) { $$("button", e.target.parentNode).forEach((b) => b.setAttribute("aria-pressed", b === e.target)); }
  drawPaper();

  const saveBtn = h("button", { class: "btn primary", onclick: async () => {
    saveBtn.disabled = true;
    try {
      const r = await api.send("POST", `/api/rec/${encodeURIComponent(S.rid)}/clean`);
      toast(r.moved_to ? `Saved. The old file is in ${r.moved_to}/` : "Saved transcript_clean.txt");
      S.doc = null;
      S.rec = await api.get(`/api/rec/${encodeURIComponent(S.rid)}`);
      poll(); renderSteps(); renderExport(body);
    } catch (e) { toast(e.message, true); saveBtn.disabled = false; }
  } }, d.state === "current" ? "Save again" : "Save transcript_clean.txt");

  fill(pane, 
    h("h2", {}, "Export"),
    h("p", { class: "lead" }, "One paragraph per speaker turn, each with the time it starts. Slang and swearing stay as spoken; left-out lines are not in it."),
    stateNote,
    h("ul", { class: "checklist" }, checks.map(([ok, txt]) => h("li", { class: ok ? "" : "todo" }, txt))),
    h("div", { class: "row", style: { margin: "18px 0" } }, saveBtn,
      h("button", { class: "btn", onclick: () => copyText(d.text) }, "Copy text"),
      h("span", { class: "small muted copyline" }, h("code", {}, d.path.replace(/^\/home\/[^/]+/, "~")),
        h("button", { class: "btn sm quiet", onclick: () => copyText(d.path) }, "Copy path"))),
    tabs, paper);
}

// ------------------------------------------------------------------ boot
poll().then(route);
