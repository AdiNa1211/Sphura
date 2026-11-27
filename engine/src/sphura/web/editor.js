// SPHURA timeline editor (loaded before the main script; shares its globals)
// ---------- Timeline editor ----------
function edOpen() { return !!(ed && $("#edtl")); }
function edLeaveOk() {
  // Edits autosave; leaving sends any pending save. Only a failed save asks first
  if (!edOpen()) return true;
  edCommit();
  if (ed.conflict || ed.saveError) {
    if (!confirm("Your latest changes to this Short couldn't be saved.\n\nLeave anyway and lose them?")) return false;
    ed.dirty = false; return true;
  }
  if (ed.dirty) edSave(true);
  return true;
}
window.addEventListener("beforeunload", e => {
  if (!edOpen()) return;
  edCommit();
  if (ed.dirty && !ed.conflict) edSave(true);
  if (ed.conflict || ed.saveError) { e.preventDefault(); e.returnValue = ""; }
});

// ---------- History (undo/redo) and autosave ----------
// Every change is detected by comparing the editable state after a pointer release, a field
// change or a typing pause, so no edit can skip the history. Sites only name the step.
const ED_KEYS = ["segments", "overlays", "fx_off", "caption_bar", "speed", "seg_audio", "markers"];
function edSnap() {
  const s = ed.st;
  return JSON.stringify({ segments: s.segments, overlays: s.overlays, fx_off: s.fx_off, caption_bar: !!s.caption_bar,
    speed: s.speed, seg_audio: s.seg_audio, markers: s.markers });
}
function edDefaults(st) { st.speed ??= []; st.seg_audio ??= st.segments.map(() => ({ gain_db: 0, mute: false })); st.markers ??= []; st.teaser_s ??= st.teaser_out; return st; }
function edKeys() { for (const o of [...ed.st.overlays, ...ed.st.speed, ...ed.st.markers]) if (o._k == null) o._k = ++ed.k; }
function edFlags() {
  // What the preview shows vs what's edited: unrendered cuts play live from the recording
  const r = ed.rendered && JSON.parse(ed.rendered), s = ed.st, same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  const bare = l => (l || []).map(p => [p.start_s, p.end_s, p.factor]);
  ed.dirtyCuts = !r || !same(r.segments, s.segments);
  ed.dirtyFx = !!r && !same([...r.fx_off].sort(), [...s.fx_off].sort());
  ed.dirtyBar = !!r && r.caption_bar !== !!s.caption_bar;
  ed.dirtySpeed = !!r && !same(bare(r.speed), bare(s.speed));
  ed.dirtyAudio = !!r && !same(r.seg_audio, s.seg_audio);
}
function edCommit(name) {
  if (!ed || ed.restoring) return;
  let now = edSnap(); if (now === ed.base) return;
  // Text and speed changed by hand become yours: re-planning never touches them
  const was = JSON.parse(ed.base), plain = o => JSON.stringify({ ...o, owner: undefined, src_s: undefined });
  const before = new Map([...was.overlays, ...was.speed].map(o => [o._k, plain(o)]));
  for (const o of ed.st.overlays) if (before.get(o._k) !== plain(o)) o.owner = "user";
  for (const p of ed.st.speed) if (before.get(p._k) !== plain(p)) { p.owner = "user"; p.src_s = edPreToSrc(p.start_s); }
  now = edSnap();
  ed.undo.push({ name: name || ed.pending || "Edit", state: ed.base });
  if (ed.undo.length > 100) ed.undo.shift();
  ed.redo = []; ed.base = now; ed.pending = null;
  edFlags(); edHistUI(); edQueueSave(); edNote();
}
function edRestore(state) {
  const v = JSON.parse(state);
  for (const k of ED_KEYS) ed.st[k] = v[k];
  ed.base = state;
  if (!edSelValid(ed.sel)) ed.sel = null;
  ed.multi = ed.multi.filter(edSelValid);
  edFlags(); edDraw(); edInspector(); edDrawText(edNow()); edHistUI(); edQueueSave(); edNote();
}
function edUndo() {
  edCommit(); const step = ed.undo.pop(); if (!step) return;
  ed.redo.push({ name: step.name, state: edSnap() }); edRestore(step.state); toast(`Undone: ${step.name}`);
}
function edRedo() {
  edCommit(); const step = ed.redo.pop(); if (!step) return;
  ed.undo.push({ name: step.name, state: edSnap() }); edRestore(step.state); toast(`Redone: ${step.name}`);
}
function edHistUI() {
  const u = $("#edundo"), r = $("#edredo"); if (!u) return;
  const last = ed.undo[ed.undo.length - 1], next = ed.redo[ed.redo.length - 1];
  u.disabled = !last; u.title = last ? `Undo: ${last.name} (Ctrl+Z)` : "Nothing to undo";
  r.disabled = !next; r.title = next ? `Redo: ${next.name} (Ctrl+Shift+Z)` : "Nothing to redo";
}
function edSaveUI(text, cls = "") { const el = $("#edsave"); if (el) { el.className = "small " + cls; el.innerHTML = text; } }
function edQueueSave() {
  if (ed.conflict) return;
  ed.dirty = true; edSaveUI("Unsaved changes", "muted");
  clearTimeout(ed.saveT); ed.saveT = setTimeout(() => edSave(), 1500);
}
function edBody(v) {
  // The editable state in the API's field names
  return { segments: v.segments, overlays: v.overlays, fx_off: v.fx_off, caption_bar: v.caption_bar,
    speed: v.speed, segment_audio: v.seg_audio, markers: v.markers };
}
async function edSave(leaving) {
  // Stores the edit list only (no render); the effects re-plan at the next preview or export
  clearTimeout(ed.saveT);
  if (!ed.dirty || ed.conflict) return;
  if (ed.busy && !leaving) { ed.saveT = setTimeout(() => edSave(), 1500); return; }
  if (ed.saving && !leaving) { ed.saveAgain = true; return; }
  const snap = edSnap(), now = edBody(JSON.parse(snap)), was = edBody(JSON.parse(ed.saved)), body = { render: "none", base_version: ed.st.version };
  for (const k of Object.keys(now)) if (JSON.stringify(now[k]) !== JSON.stringify(was[k])) body[k] = now[k];
  if (Object.keys(body).length === 2) { ed.dirty = false; edSaveUI(`${ic("check")} Saved`, "muted"); return; }
  const id = ed.st.id, mine = ed;
  ed.saving = true; edSaveUI(`<span class="spin"></span> Saving…`, "muted");
  try {
    const r = await fetch(`/v1/clips/${id}/editor`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), keepalive: !!leaving });
    if (mine !== ed) return; // the editor was closed or reopened meanwhile
    if (r.status === 409) {
      ed.conflict = true;
      edSaveUI(`<span class="err">Changed in another tab or window.</span> <button class="small" onclick="openEditor('${id}')">Reload</button>`);
      return;
    }
    if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch {} throw new Error(m); }
    ed.st.version = (await r.json()).version; ed.saved = snap; ed.saveError = null;
    ed.dirty = edSnap() !== snap;
    if (ed.dirty || ed.saveAgain) { ed.saveAgain = false; ed.saveT = setTimeout(() => edSave(), 300); }
    else edSaveUI(`${ic("check")} Saved`, "muted");
  } catch (e) {
    if (mine !== ed) return;
    ed.saveError = e.message; edSaveUI(`<span class="err">Not saved: ${esc(e.message)}</span>`);
    ed.saveT = setTimeout(() => edSave(), 5000);
  } finally { if (mine === ed) ed.saving = false; }
}
window.addEventListener("pointerdown", e => {
  // Working on the timeline or preview leaves a text field: keys go to the editor again
  const f = document.activeElement;
  if (edOpen() && e.target.closest("#edtl, #edframe, #edbar") && f && ["INPUT", "TEXTAREA"].includes(f.tagName)) f.blur();
  if ($("#edmenu") && !e.target.closest("#edmenu")) edMenuClose(false);
}, true);
window.addEventListener("pointerup", () => { if (edOpen()) setTimeout(() => edCommit(), 0); }, true);
document.addEventListener("click", () => { if (edOpen()) setTimeout(() => edCommit(), 0); });
document.addEventListener("change", e => { if (edOpen() && (e.target.closest("#edinspector") || !e.target.isConnected)) setTimeout(() => edCommit(), 0); });
document.addEventListener("input", e => {
  if (!edOpen() || !e.target.closest("#edinspector")) return;
  clearTimeout(ed.inT); ed.inT = setTimeout(() => edCommit(), 800); // one step per typing pause
});
document.addEventListener("wheel", e => {
  if (!edOpen() || !e.target.closest("#edtext")) return;
  clearTimeout(ed.inT); ed.inT = setTimeout(() => edCommit(), 500);
}, { passive: true });
const ED_STYLES = { plain: "Subtitle", hook: "Pop-in hook", caps: "Caps opener", bar: "Meme caption bar" };
const FX_NAMES = { cold_open: "Cold open (climax first)", zoom: "Zoom punches", shake: "Camera shake", slowmo: "Slow motion",
  flash: "Flashes & glitches", rgb: "RGB split", trail: "Motion trail", vignette: "Vignette", letterbox: "Letterbox bars", sfx: "Sound effects" };
const ED_TRACKS = [["video", "Video", 48], ["audio", "Game audio", 34], ["music", "Music", 34], ["text", "Text", 0]];
const ED_SPEEDS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2];
const ED_FRAME = 1 / 30;
let ed = null;

async function openEditor(id) {
  nav("editor");
  rlStop();
  stopPolling(); leaveDownloads();
  history.replaceState(null, "", "#edit/" + id);
  $("#main").innerHTML = `<div class="card"><span class="spin"></span> Opening the editor: loading the Short and its recording…</div>`;
  try {
    const st = edDefaults(await api(`/v1/clips/${id}/editor`));
    let th = {}; try { th = JSON.parse(localStorage.getItem("sphura_tracks")) || {}; } catch {}
    ed = { st, sel: null, multi: [], dirty: false, pps: 40, busy: false, audio: null, audioKey: null, undo: [], redo: [], k: 0,
      th: { closed: {}, ...th }, snap: localStorage.getItem("sphura_snap") !== "off", follow: true, loop: null, mips: new WeakMap() };
    edKeys(); ed.base = ed.saved = edSnap(); ed.rendered = st.draft_fresh ? ed.base : null; edFlags();
    renderEditor();
    if (!st.draft_fresh && !st.source_missing) edRender("draft", true);
  } catch (e) { $("#main").innerHTML = `<div class="empty"><b>Couldn't open the editor</b><span class="err">${esc(e.message)}</span><div class="row"><button onclick="openEditor('${id}')">${ic("refresh")} Try again</button></div></div>`; }
}

// ---------- Time on the Short ----------
// Speed pieces sit on the pre-speed timeline (cold open included). Unrendered cuts are drawn 1:1,
// because the speed ramps are re-planned for the new cut at the next preview.
function edLead() { return ed.dirtyCuts ? ed.st.teaser_out : ed.st.teaser_s; }
function edOut(p) {
  if (ed.dirtyCuts) return p;
  let o = p;
  for (const s of ed.st.speed) if (p > s.start_s) o += (Math.min(p, s.end_s) - s.start_s) * (1 / s.factor - 1);
  return o;
}
function edPre(o) {
  if (ed.dirtyCuts) return o;
  let pre = 0, out = 0;
  for (const s of [...ed.st.speed].sort((a, b) => a.start_s - b.start_s)) {
    if (o <= out + (s.start_s - pre)) return pre + (o - out);
    out += s.start_s - pre; pre = s.start_s;
    const len = (s.end_s - s.start_s) / s.factor;
    if (o <= out + len) return pre + (o - out) * s.factor;
    out += len; pre = s.end_s;
  }
  return pre + (o - out);
}
function edPreTotal() { return edLead() + ed.st.segments.reduce((a, [x, y]) => a + (y - x), 0); }
function edTotal() { return edOut(edPreTotal()); }
function edSegOut() {
  let t = edLead();
  return ed.st.segments.map(([a, b]) => { const r = [edOut(t), edOut(t + b - a)]; t += b - a; return r; });
}
function edPreToSrc(p) {
  // Recording time at a point of the pre-speed timeline (null inside the cold open)
  let r = p - edLead(); if (r < -1e-6) return null;
  for (const [a, b] of ed.st.segments) { if (r <= b - a + 1e-6) return +(a + Math.max(0, r)).toFixed(3); r -= b - a; }
  const l = ed.st.segments[ed.st.segments.length - 1]; return l ? l[1] : null;
}
function edOutToSrc(o) {
  const p = edPre(o);
  if (p < edLead()) { const ts = ed.st.teaser_src; return ts ? +(ts[0] + p).toFixed(3) : null; }
  return edPreToSrc(p);
}
function edSrcToOut(src) {
  // Where a moment of the recording plays in the Short (its first appearance), or null if cut out
  let t = edLead();
  for (const [a, b] of ed.st.segments) { if (src >= a - 1e-6 && src <= b + 1e-6) return edOut(t + src - a); t += b - a; }
  return null;
}
function edFx2(f) { return +(+f).toFixed(2); }
function edSpdLabel(p) { return p.freeze ? "❚❚" : edFx2(p.factor) + "×"; }
function edSpeedOut(p) { return [edOut(p.start_s), edOut(p.end_s)]; }

// ---------- Layout ----------
function renderEditor() {
  const s = ed.st;
  const back = s.project_id ? `state.tab='shorts';openProject('${s.project_id}')` : "showNew()";
  $("#main").innerHTML = `
    <div class="edhead"><button class="ghost icon" onclick="${back}" data-tip="Back to Shorts" aria-label="Back to Shorts">${ic("back")}</button>
      <h1 class="grow">${esc(s.title)}</h1>
      <button class="ghost icon" id="edundo" onclick="edUndo()" aria-label="Undo" disabled>↶</button><button class="ghost icon" id="edredo" onclick="edRedo()" aria-label="Redo" disabled>↷</button>
      <span class="small muted" id="edsave">${ed.dirty ? "Unsaved changes" : `${ic("check")} Saved`}</span>
      <span class="small muted" id="edstatus"></span>
      <select id="edpq" style="width:auto" aria-label="Preview quality" title="Fast: half size, 30 fps, quick to make. Full: exactly like the export." onchange="localStorage.setItem('sphura_preview', this.value)">
        <option value="fast">Preview: fast</option><option value="full" ${localStorage.getItem("sphura_preview") === "full" ? "selected" : ""}>Preview: full quality</option></select>
      <button onclick="edRender('draft')" id="edprev" title="Render your changes so you can watch them">${ic("eye")} Update preview</button>
      <button class="primary" onclick="edRender('final')" title="Render the final Short and save it">${ic("export")} Export</button></div>
    ${s.source_missing ? `<div class="card" id="edrelink" style="border-color:var(--accent2)">
      <div><b>Recording not found</b> <span class="small muted">SPHURA can't open the recording this Short was cut from:</span></div>
      <div class="small faint" style="margin:4px 0 8px;word-break:break-all">${esc(s.source_path)}</div>
      <div class="small muted" style="margin-bottom:8px">It was moved, renamed or its drive isn't connected. Reconnect the drive and reopen the Short, or point SPHURA at the file's new place. Only the same recording is accepted.</div>
      <div class="row"><input type="text" id="edrelinkpath" class="grow" aria-label="New location of the recording" placeholder="New location, e.g. E:\\Game clips\\recording.mp4">
        <button onclick="edBrowseSource()">Browse…</button><button class="primary" onclick="edRelink()">Use this file</button></div></div>` : ""}
    <div class="edgrid">
      <div class="card edstage">
        <div class="edmodes"><button id="edm-short" class="on" onclick="edMode(false)" title="Watch and edit the Short">${ic("video")} Short</button>
          <button id="edm-rec" onclick="edMode(true)" title="See the whole recording and add more footage to the Short">${ic("reel")} Full recording</button></div>
        <div id="edframe" class="edframe" style="aspect-ratio:${s.out_w}/${s.out_h}">
          <video id="edvid" playsinline preload="auto"></video><div id="edtext" class="edtext"></div><div id="edlive" class="edlive" hidden>LIVE · straight from the recording</div>
          <video id="edtrimv" class="edtrimv" muted playsinline preload="auto" hidden></video><div id="edtrimlab" class="edtrimlab" hidden></div></div>
        <div class="edbar" id="edbar"></div>
      </div>
      <div class="card" id="edinspector"></div>
    </div>
    <div class="card"><div class="row small muted" style="margin-bottom:6px"><b style="color:var(--text)">Timeline</b><div class="grow"></div>
      <span id="edtlhint"></span>
      <button class="small ghost ${ed.snap ? "on" : ""}" id="edsnapbtn" aria-pressed="${ed.snap}" onclick="edSnapToggle()" title="Snap edges to parts, text, beats, markers and the playhead. Hold Alt while dragging to switch it off for a moment.">⌁ Snap</button>
      <button class="small ghost" onclick="edFitZoom()" title="Fit the whole Short (Z)">Fit</button>
      <span aria-hidden="true">−</span><input type="range" id="edzoom" min="10" max="320" value="${ed.pps}" style="width:120px" oninput="edZoomSlider(+this.value)" aria-label="Zoom" title="Zoom (Ctrl + scroll, or pinch)"><span aria-hidden="true">+</span>
      <button class="small ghost" onclick="edHelp()" aria-label="Keyboard shortcuts" title="Keyboard shortcuts (?)">?</button></div>
      <div id="edtl" class="edtl" tabindex="0" aria-label="Timeline. Shift+F10 opens the menu for the selection."></div></div>`;
  const tl = $("#edtl");
  tl.addEventListener("pointerdown", edDown);
  tl.addEventListener("contextmenu", edContext);
  tl.addEventListener("scroll", edScrolled, { passive: true });
  tl.addEventListener("pointermove", edHover);
  tl.addEventListener("pointerleave", () => { const h = $("#edhov"); if (h) h.hidden = true; });
  tl.addEventListener("wheel", edWheel, { passive: false });
  tl.addEventListener("dblclick", e => { const d = e.target.closest(".edsplit"); if (d) { delete ed.th[d.dataset.tr]; edSaveTracks(); edDraw(); } });
  edSource(); edDraw(); edInspector(); edHistUI(); edNote();
}

function edMode(orig) {
  if (ed.orig === orig) return;
  ed.orig = orig; ed.sel = null; ed.multi = []; edSetMarks(null, null); edSource(); edInspector();
}
function edSource() {
  const v = $("#edvid"), orig = !!ed.orig;
  ed.live = false; $("#edframe")?.classList.remove("live"); if ($("#edlive")) $("#edlive").hidden = true;
  $("#edm-short")?.classList.toggle("on", !orig); $("#edm-rec")?.classList.toggle("on", orig);
  v.src = orig ? `/v1/clips/${ed.st.id}/source` : (ed.st.has_draft ? `/v1/clips/${ed.st.id}/draft.mp4?v=${Date.now()}` : `/v1/clips/${ed.st.id}/video`);
  $("#edframe").style.aspectRatio = orig ? "16/9" : `${ed.st.out_w}/${ed.st.out_h}`;
  v.ontimeupdate = v.onseeked = edTick; v.onplay = v.onpause = () => { const b = $("#edplaybtn"); if (b) { b.innerHTML = v.paused ? ic("play") : ic("pause"); b.setAttribute("aria-label", v.paused ? "Play" : "Pause"); } };
  if (orig) { const seg = ed.st.segments[ed.sel?.kind === "seg" ? ed.sel.i : 0]; v.onloadedmetadata = () => v.currentTime = seg[0]; }
  else v.onloadedmetadata = null;
  if (!orig && ed.dirtyCuts) { v.removeAttribute("src"); edLiveSync(); }
  if ($("#edtl")) edSourceView();
}
async function edBrowseSource() {
  try { const r = await api("/v1/library/choose-file", { method: "POST" }); if (r.path) $("#edrelinkpath").value = r.path; } catch (e) { toast(e.message); }
}
async function edRelink() {
  const path = $("#edrelinkpath").value.trim(); if (!path) return toast("Pick the recording's new location first");
  try {
    await api(`/v1/assets/${ed.st.asset_id}/relink`, { method: "POST", body: { path } });
    toast("Found it. Every Short from this recording uses the new location."); openEditor(ed.st.id);
  } catch (e) { toast(e.message); }
}
function edBtn(label, on, why, fn, cls = "", aria = "") {
  return `<button class="${cls}" ${on ? "" : "disabled"} title="${esc(on ? "" : why)}" ${aria ? `aria-label="${esc(aria)}"` : ""} onclick="${fn}">${label}</button>`;
}
function edSplitOk() {
  const t = edNow(), so = edSegOut();
  return !ed.orig && so.some(([a, b]) => t > a + 0.3 && t < b - 0.3);
}
function edToolbar() {
  // One row of plain-labelled actions; greyed-out buttons say why on hover
  const bar = $("#edbar"); if (!bar) return;
  const v = $("#edvid"), paused = !v || v.paused;
  const play = `<button onclick="edPlay()" id="edplaybtn" aria-label="${paused ? "Play" : "Pause"}">${paused ? ic("play") : ic("pause")}</button><span class="small muted" id="edtime"></span>`
    + `<button class="${ed.loop ? "on" : ""}" aria-pressed="${!!ed.loop}" onclick="edLoopToggle()" title="${ed.loop ? "Stop looping" : "Loop the selection, or the whole Short"}">⟲ Loop</button><div class="grow"></div>`;
  if (!ed.orig) {
    const items = edSelAll(), sel = ed.sel, canDel = items.some(x => x.kind === "txt" || x.kind === "spd" || x.kind === "mk") || (items.some(x => x.kind === "seg") && ed.st.segments.length > items.filter(x => x.kind === "seg").length);
    const what = items.length > 1 ? ` ${items.length} items` : sel?.kind === "txt" ? " text" : sel?.kind === "seg" ? ` part ${sel.i + 1}` : sel?.kind === "spd" ? " speed change" : sel?.kind === "mk" ? " marker" : "";
    bar.innerHTML = play
      + `<span id="edsplitwrap">${edBtn(ic("scissors") + " Split<kbd>S</kbd>", edSplitOk(), "Move the playhead inside a part (not at its edge) to split it there", "edSplit()")}</span>`
      + edBtn(`${ic("trash")} Delete${what}<kbd>Del</kbd>`, canDel, sel?.kind === "seg" ? "A Short needs at least one part" : "Click a part, a text, a speed badge or a marker first", "edDelete()")
      + edBtn(ic("text") + " Text", true, "", "edAddText()")
      + edBtn("⚑ Marker<kbd>B</kbd>", true, "", "edAddMarker()")
      + `<i class="sep"></i>` + edBtn(ic("plus") + " Add footage", true, "", "edMode(true)");
  } else {
    const p = ed.pick, overlap = p && ed.st.segments.some(([a, b]) => p[0] < b && p[1] > a);
    const note = p ? `Marked ${hms(p[0])}–${hms(p[1])} (${(p[1] - p[0]).toFixed(1)} s)`
      : ed.mA != null ? `Start marked at ${hms(ed.mA)}. Now mark the end.` : ed.mB != null ? `End marked at ${hms(ed.mB)}. Now mark the start.` : "Mark a start and an end, drag across the recording, or click a yellow moment";
    bar.innerHTML = play
      + edBtn("⟦ Mark start<kbd>I</kbd>", true, "", "edMark(0)") + edBtn("Mark end ⟧<kbd>O</kbd>", true, "", "edMark(1)")
      + edBtn(ic("plus") + " Add to Short", !!p, "Mark a start and an end first", "edAddPick()", "primary")
      + edBtn("− Remove from Short", !!overlap, p ? "The marked stretch isn't in your Short" : "Mark the stretch to remove first", "edRemovePick()")
      + `<i class="sep"></i>` + edBtn(ic("back") + " Back to Short", true, "", "edMode(false)")
      + `<div style="flex-basis:100%"></div><span class="note">${esc(note)}</span>`;
  }
  edTick();
}
function edNote() {
  // Says which edits the preview doesn't show yet
  const st = $("#edstatus"); if (!st || ed.busy || ed.live) return;
  const todo = [ed.dirtyFx && "effects", ed.dirtyBar && "layout", ed.dirtySpeed && "speed", ed.dirtyAudio && "volume"].filter(Boolean);
  if (todo.length) st.textContent = `Not in the preview yet: ${todo.join(", ")}. Press Update preview.`;
  else if (/^Not in the preview/.test(st.textContent)) st.textContent = "";
}
function edFit() { return Math.max(0.02, (($("#edtl")?.clientWidth || 1000) - 40) / ed.st.source_duration); }
function edFitShort() { return Math.max(2, (($("#edtl")?.clientWidth || 1000) - 40) / Math.max(1, edTotal())); }
function edZoomAt(p, clientX, anchorT) {
  // Zoom keeping the time under the pointer (or the view's centre) in place
  const box = $("#edtl"); if (!box) return;
  const b = box.getBoundingClientRect(), x = clientX == null ? box.clientWidth / 2 : clientX - b.left, z = $("#edzoom");
  if (ed.orig) {
    const fit = edFit(), span = Math.log(Math.max(1.01, 60 / fit));
    const t = anchorT ?? (box.scrollLeft + x) / ed.spps; ed.spps = Math.min(60, Math.max(fit, p)); edDraw();
    edScrollTo(t * ed.spps - x); if (z) z.value = Math.round(100 * Math.log(ed.spps / fit) / span);
    return;
  }
  const t = anchorT ?? (box.scrollLeft + x) / ed.pps;
  ed.pps = Math.min(320, Math.max(Math.min(10, edFitShort()), p));
  edLayout(); edScrollTo(t * ed.pps - x); edWaves(); edFab();
  if (z) { z.min = Math.floor(Math.min(10, edFitShort())); z.value = Math.round(ed.pps); }
  clearTimeout(ed.zoomT); ed.zoomT = setTimeout(() => { if (ed && !ed.orig) edDraw(); }, 200); // sharp filmstrips once it settles
}
function edZoomSlider(v) {
  if (!ed.orig) return edZoomAt(v);
  const fit = edFit(); edZoomAt(fit * Math.exp(Math.log(Math.max(1.01, 60 / fit)) * v / 100));
}
function edScrollTo(x) { const box = $("#edtl"); ed.autoScroll = true; box.scrollLeft = x; requestAnimationFrame(() => { if (ed) ed.autoScroll = false; }); }
function edFitZoom() { if (ed.orig) return edZoomAt(edFit()); $("#edtl").scrollLeft = 0; edZoomAt(edFitShort(), $("#edtl").getBoundingClientRect().left); }
function edWheel(e) {
  // Ctrl + scroll or a trackpad pinch zooms around the pointer; Shift + scroll moves sideways
  if (!ed) return;
  const box = $("#edtl");
  if (e.ctrlKey || e.metaKey) {
    e.preventDefault();
    const cur = ed.orig ? ed.spps : ed.pps, now = performance.now();
    // One gesture keeps one anchor, so the point stays put even while the Short still fits the view
    if (!ed.zA || now - ed.zA.at > 300 || Math.abs(ed.zA.x - e.clientX) > 4) ed.zA = { t: (box.scrollLeft + e.clientX - box.getBoundingClientRect().left) / cur, x: e.clientX };
    ed.zA.at = now;
    ed.zq = { p: (ed.zq?.p ?? cur) * Math.exp(-e.deltaY * (e.deltaMode ? 0.05 : 0.0025)), x: e.clientX };
    if (!ed.zraf) ed.zraf = requestAnimationFrame(() => { ed.zraf = 0; const q = ed.zq; ed.zq = null; edZoomAt(q.p, q.x, ed.zA?.t); });
  } else if (e.shiftKey && !e.deltaX) { e.preventDefault(); box.scrollLeft += e.deltaY; }
}
function edScrolled() {
  if (!ed) return;
  if (!ed.autoScroll && !$("#edvid")?.paused) ed.follow = false; // you scrolled by hand: stop following
  edWaves(); edFab();
}
function edHover(e) {
  // Time under the pointer, shown on the ruler
  const h = $("#edhov"), inner = $("#edtl .edinner"); if (!h || !inner || document.body.classList.contains("edscrub")) return;
  const x = e.clientX - inner.getBoundingClientRect().left;
  if (x < 0) { h.hidden = true; return; }
  const t = x / (ed.orig ? ed.spps : ed.pps);
  h.hidden = false; h.style.transform = `translateX(${x}px)`; h.textContent = ed.orig ? hms(t) : `${t.toFixed(2)} s`;
}
async function edSourceView() {
  const z = $("#edzoom"), box = $("#edtl"), segs = ed.st.segments;
  $("#edtlhint").innerHTML = ed.orig ? `<b style="color:var(--text)">Whole recording</b>: highlighted parts are in your Short. Drag on the recording or click a moment to add it.&nbsp;&nbsp;` : "";
  if (!ed.orig) { z.min = Math.floor(Math.min(10, edFitShort())); z.max = 320; z.value = ed.pps; edDraw(); return; }
  const fit = edFit(), lo = Math.min(...segs.map(x => x[0])), hi = Math.max(...segs.map(x => x[1]));
  if (!ed.spps) ed.spps = Math.min(60, Math.max(fit, (box.clientWidth - 40) / (hi - lo + 240))); // the Short ±2 min
  z.min = 0; z.max = 100; z.value = Math.round(100 * Math.log(ed.spps / fit) / Math.log(Math.max(1.01, 60 / fit)));
  edDraw(); box.scrollLeft = Math.max(0, (lo - 60) * ed.spps);
  if (!ed.moments && ed.st.project_id) {
    try { ed.moments = await api(`/v1/projects/${ed.st.project_id}/highlights?limit=100`); } catch { ed.moments = []; }
    if (ed.orig) edDraw();
  }
}

// ---------- Playback ----------
function edPlay() {
  const v = $("#edvid"); edRev(false);
  if (v.paused) {
    ed.follow = true; v.playbackRate = 1;
    if (ed.loop && !ed.orig) { const t = edNow(); if (t < ed.loop[0] || t >= ed.loop[1] - 0.05) edGoto(ed.loop[0]); }
    if (ed.live) { const segs = ed.st.segments, n = segs.length - 1; if (v.currentTime >= segs[n][1] - 0.1 && ed.liveSeg >= n) edGoto(0); else if (ed.liveSeg == null) ed.liveSeg = Math.max(0, edLivePart(v.currentTime)); }
  }
  v.paused ? v.play() : v.pause();
}
function edJKL(k) {
  // K stops, L plays (again: faster), J plays backwards by stepping (browsers can't play video in reverse)
  const v = $("#edvid");
  if (k === "k") { edRev(false); v.pause(); v.playbackRate = 1; return; }
  if (k === "l") { edRev(false); if (v.paused) edPlay(); else v.playbackRate = Math.min(4, v.playbackRate * 2); return; }
  v.pause(); ed.follow = true; ed.revRate = ed.rev ? Math.min(4, ed.revRate * 2) : 1; edRev(true);
}
function edRev(on) {
  cancelAnimationFrame(ed.rev || 0); ed.rev = 0; if (!on) return;
  let last = performance.now();
  const step = now => {
    const v = $("#edvid"); if (!v || !ed.rev) return;
    if (!v.seeking) {
      const t = (ed.orig ? v.currentTime : edNow()) - (now - last) / 1000 * ed.revRate; last = now;
      if (t <= 0) { ed.orig ? v.currentTime = 0 : edGoto(0); ed.rev = 0; return; }
      ed.orig ? v.currentTime = t : edGoto(t); edTick();
    } else last = now;
    ed.rev = requestAnimationFrame(step);
  };
  ed.rev = requestAnimationFrame(step);
}
function edStep(d) {
  const v = $("#edvid"); edRev(false); v.pause();
  if (ed.orig) v.currentTime = Math.max(0, v.currentTime + d); else edGoto(Math.min(edTotal() - 0.02, Math.max(0, edNow() + d)));
  setTimeout(edTick, 60);
}
function edLoopToggle() {
  if (ed.loop) ed.loop = null;
  else {
    const s = ed.sel, so = edSegOut();
    ed.loop = s?.kind === "seg" ? so[s.i] : s?.kind === "txt" ? [ed.st.overlays[s.i].start_s, ed.st.overlays[s.i].end_s]
      : s?.kind === "spd" ? edSpeedOut(ed.st.speed[s.i]) : [0, edTotal()];
  }
  edDraw(); edToolbar();
}
function edLiveSync() {
  // Changed cuts play straight from the recording, part after part, until the next preview render
  const want = !ed.orig && !!ed.dirtyCuts, v = $("#edvid"); if (!v || want === !!ed.live) return;
  const t = edNow(); ed.live = want;
  $("#edframe").classList.toggle("live", want); $("#edlive").hidden = !want;
  if (!want) { v.volume = 1; return; }
  v.pause(); v.src = `/v1/clips/${ed.st.id}/source`;
  v.onloadedmetadata = () => edGoto(t);
  $("#edstatus").textContent = "Live preview: no effects or music yet. Press Update preview to see the finished Short.";
}
function edLivePart(src) {
  const segs = ed.st.segments, i = segs.findIndex(([a, b], k) => k >= (ed.liveSeg || 0) && src >= a - 0.05 && src < b);
  return i >= 0 ? i : segs.findIndex(([a, b]) => src >= a - 0.05 && src < b);
}
function edGoto(t) {
  // Move to time t of the Short (seconds), whichever video is showing
  const v = $("#edvid");
  if (ed.live) {
    const so = edSegOut(), segs = ed.st.segments;
    let i = so.findIndex(([a, b]) => t >= a && t < b); if (i < 0) i = t < (so[0]?.[0] || 0) ? 0 : segs.length - 1;
    ed.liveSeg = i; v.currentTime = Math.min(segs[i][1] - 0.02, segs[i][0] + Math.max(0, t - so[i][0]));
  } else if (ed.orig) {
    const so = edSegOut(), i = so.findIndex(([a, b]) => t >= a && t <= b); if (i >= 0) v.currentTime = ed.st.segments[i][0] + (t - so[i][0]);
  } else v.currentTime = t;
}
function edNow() {
  const v = $("#edvid");
  if (ed.live) { const i = ed.liveSeg != null && ed.st.segments[ed.liveSeg] ? ed.liveSeg : edLivePart(v.currentTime || 0), so = edSegOut(); return i < 0 || !so[i] ? (ed.liveT || 0) : (ed.liveT = so[i][0] + (v.currentTime - ed.st.segments[i][0])); }
  if (!ed.orig) return v.currentTime || 0;
  // Original footage: map the source time back onto the Short's timeline when inside a part
  const so = edSegOut(); const i = ed.st.segments.findIndex(([a, b]) => v.currentTime >= a && v.currentTime <= b);
  return i < 0 ? 0 : so[i][0] + (v.currentTime - ed.st.segments[i][0]);
}
function edSrcLabel(t) {
  const i = ed.st.segments.findIndex(([a, b]) => t >= a && t <= b);
  return `${hms(t)} in the recording${i >= 0 ? ` · part ${i + 1}` : ""}`;
}
function edTick() {
  const v = $("#edvid"); if (!v || !ed) return;
  if (ed.live && !v.paused) { // jump over the gaps between parts, in the Short's order
    const segs = ed.st.segments, i = ed.liveSeg || 0;
    if (v.currentTime >= segs[i][1] - 0.04 || v.currentTime < segs[i][0] - 0.3) {
      if (i + 1 < segs.length) { ed.liveSeg = i + 1; v.currentTime = segs[i + 1][0]; } else if (ed.loop) edGoto(ed.loop[0]); else v.pause();
    }
  }
  if (ed.live) { const a = ed.st.seg_audio[ed.liveSeg || 0]; v.volume = !a || a.mute ? (a ? 0 : 1) : Math.min(1, 10 ** (a.gain_db / 20)); }
  const t = edNow(), src = v.currentTime || 0;
  if (ed.loop && !ed.orig && !v.paused && t >= ed.loop[1] - 0.03) edGoto(ed.loop[0]);
  const tl = $("#edtime"); if (tl) tl.textContent = ed.orig ? edSrcLabel(src) : `${t.toFixed(1)} s / ${edTotal().toFixed(1)} s`;
  const sp = $("#edsplitwrap button"); if (sp) sp.disabled = !edSplitOk();
  const x = ed.orig ? src * ed.spps : t * ed.pps, ph = $("#edph"); if (ph) ph.style.transform = `translateX(${x}px)`;
  const box = $("#edtl");
  if (box && (!v.paused || ed.rev) && ed.follow && (x < box.scrollLeft + 20 || x > box.scrollLeft + box.clientWidth - 40)) {
    ed.autoScroll = true; box.scrollLeft = x - box.clientWidth * 0.2; requestAnimationFrame(() => { if (ed) ed.autoScroll = false; });
  }
  if (!ed.orig && $("#edfab") && !$("#edfab").hidden) edFab();
  edDrawText(t);
  if (!v.paused) requestAnimationFrame(edTick);
}

// ---------- Text on the preview ----------
function edStyleBox(o) {
  const s = ed.st, w = s.out_w, h = s.out_h, base = Math.min(w, h) * 0.085;
  const size = ({ caps: base * 0.9, plain: base * 0.7, bar: w * 0.064 }[o.style] || base) * (o.size || 1);
  const marginV = (h > w * 1.5 ? 0.2 : 0.1) * h;
  let y = o.y;
  if (y == null) y = o.style === "bar" ? (s.bar_h / 2) / h : o.style === "plain" ? 1 - (marginV + size / 2) / h : (marginV + size / 2) / h;
  if (o.style === "bar") y = (s.bar_h / 2) / h;
  return { size, y };
}
function edDrawText(t) {
  const box = $("#edtext"); if (!box) return;
  if (ed.orig) { box.innerHTML = ""; return; }
  const scale = $("#edframe").clientWidth / ed.st.out_w;
  box.innerHTML = ed.st.overlays.map((o, i) => {
    if (t < o.start_s || t > o.end_s) return "";
    const { size, y } = edStyleBox(o);
    const upper = o.style === "hook" || o.style === "caps";
    const color = o.color || (o.style === "bar" ? "#000" : "#fff");
    const on = edIsSel("txt", i);
    return `<div class="edov ed-${o.style} ${on ? "on" : ""}" data-i="${i}"
      style="top:${y * 100}%;font-size:${size * scale}px;color:${color};${upper ? "text-transform:uppercase;" : ""}">${esc(o.text)}${on ? `<i class="edsz" title="Drag to resize"></i>` : ""}</div>`;
  }).join("");
  box.querySelectorAll(".edov").forEach(el => {
    const i = +el.dataset.i;
    el.onpointerdown = ev => ev.target.classList.contains("edsz") ? edDragSize(ev, i) : edDragY(ev, i);
    el.onwheel = ev => { ev.preventDefault(); edSize(i, ev.deltaY < 0 ? 0.05 : -0.05); };
  });
}
function edSize(i, step, value, fromSlider) {
  const o = ed.st.overlays[i]; if (!o) return;
  const v = value != null ? value : (o.size || 1) + step;
  o.size = Math.round(Math.min(3, Math.max(0.4, v)) * 100) / 100; ed.pending = "Resize text";
  edDrawText(edNow());
  if (!fromSlider) edInspector(); // keep the slider under the pointer while dragging it
  else { const n = document.querySelector('#edinspector input[type=number][step="5"]'); n && (n.value = Math.round(o.size * 100)); }
}
function edDragSize(ev, i) {
  ev.preventDefault(); ev.stopPropagation(); edSelect("txt", i);
  const o = ed.st.overlays[i], y0 = ev.clientY, s0 = o.size || 1;
  const move = e => edSize(i, null, s0 * Math.pow(1.006, e.clientY - y0), true);
  const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); edInspector(); };
  window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
}
function edDragY(ev, i) {
  ev.preventDefault(); edSelect("txt", i);
  const fr = $("#edframe").getBoundingClientRect(); const o = ed.st.overlays[i]; if (o.style === "bar") return;
  ed.pending = "Move text";
  const move = e => { o.y = Math.min(0.97, Math.max(0.03, (e.clientY - fr.top) / fr.height)); edDrawText(edNow()); };
  const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); edInspector(); };
  window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
}

// ---------- Waveforms (canvas, visible range only) ----------
function edLoadAudio() {
  // Saved cuts come back mapped through slow-mo; unsaved ones 1:1, like the timeline draws them
  const key = ed.dirtyCuts ? JSON.stringify(ed.st.segments) : "saved";
  if (key === ed.audioKey) return;
  ed.audioKey = key; clearTimeout(ed.audioTimer);
  ed.audioTimer = setTimeout(async () => {
    try {
      const a = await api(`/v1/clips/${ed.st.id}/editor/audio`, { method: "POST", body: { segments: ed.dirtyCuts ? ed.st.segments : null } });
      if (ed && ed.audioKey === key) { ed.audio = a; edDraw(); edInspector(); }
    } catch (e) { /* waveforms are optional */ }
  }, ed.audio ? 350 : 0);
}
function edMip(env, spp) {
  // Peak levels at halving resolutions; pick the one with about one sample per pixel
  let lv = ed.mips.get(env);
  if (!lv) {
    lv = [env];
    while (lv[lv.length - 1].length > 32) { const p = lv[lv.length - 1], q = []; for (let i = 0; i < p.length; i += 2) q.push(Math.max(p[i], p[i + 1] ?? 0)); lv.push(q); }
    ed.mips.set(env, lv);
  }
  let k = 0; while (k + 1 < lv.length && spp / 2 ** (k + 1) >= 1) k++;
  return lv[k];
}
function edEnv(g, env, x0, w, h, gain, viewW) {
  if (!env || env.length < 2 || w < 1) return;
  const lv = edMip(env, env.length / w), n = lv.length;
  const a = Math.max(0, Math.floor(-x0)), b = Math.min(Math.ceil(w), Math.ceil(viewW - x0));
  if (b <= a) return;
  const mid = h / 2, vals = [];
  for (let px = a; px <= b; px++) {
    const i0 = Math.floor(px / w * n), i1 = Math.max(i0 + 1, Math.floor((px + 1) / w * n));
    let m = 0; for (let k = i0; k < i1 && k < n; k++) m = Math.max(m, lv[k]);
    vals.push(Math.min(1, m * gain));
  }
  g.beginPath(); g.moveTo(x0 + a, mid);
  vals.forEach((v, k) => g.lineTo(x0 + a + k, mid - Math.max(0.02, v) * h * 0.47));
  for (let k = vals.length - 1; k >= 0; k--) g.lineTo(x0 + a + k, mid + Math.max(0.02, vals[k]) * h * 0.47);
  g.closePath(); g.fill();
}
function edCanvas(id) {
  const c = $("#" + id), box = $("#edtl"); if (!c || !box) return null;
  const dpr = window.devicePixelRatio || 1, w = box.clientWidth, h = c.parentElement.clientHeight - 8;
  if (h <= 2) { c.width = 0; return null; }
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); c.style.width = w + "px"; c.style.height = h + "px"; }
  c.style.transform = `translateX(${box.scrollLeft}px)`;
  const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  return { g, w, h, sl: box.scrollLeft };
}
function edWaves() {
  if (!ed || ed.orig || !$("#edcv-audio")) return;
  const au = ed.audio, pps = ed.pps, so = edSegOut(), s = ed.st;
  const ca = edCanvas("edcv-audio");
  if (ca && au) {
    ca.g.fillStyle = "rgba(61,220,151,.55)";
    if (au.teaser?.length && s.teaser_out > 0.05) edEnv(ca.g, au.teaser, -ca.sl, edLead() * pps, ca.h, 1, ca.w);
    if (au.parts.length === so.length) so.forEach(([a, b], i) => {
      const p = s.seg_audio[i] || {}, gain = p.mute ? 0 : Math.min(2, 10 ** ((p.gain_db || 0) / 20));
      edEnv(ca.g, au.parts[i], a * pps - ca.sl, Math.max(6, (b - a) * pps), ca.h, gain, ca.w);
    });
  }
  const cm = edCanvas("edcv-music"), m = au?.music;
  if (cm && m) {
    const total = edTotal(), x0 = m.in_s * pps - cm.sl, w = Math.max(6, (total - m.in_s) * pps);
    cm.g.fillStyle = "rgba(167,139,250,.5)"; edEnv(cm.g, m.env, x0, w, cm.h, 1, cm.w);
    if (m.dip) { cm.g.fillStyle = "rgba(0,0,0,.45)"; cm.g.fillRect(m.dip[0] * pps - cm.sl, 0, Math.max(2, (m.dip[1] - m.dip[0]) * pps), cm.h); }
    cm.g.fillStyle = "rgba(255,255,255,.45)";
    for (const t of m.beats) { const x = t * pps - cm.sl; if (t >= m.in_s && t <= total && x >= 0 && x <= cm.w) cm.g.fillRect(Math.round(x), cm.h - 7, 1, 7); }
  }
}

// ---------- Full recording view ----------
function edFilm(a, b, w) {
  // Frames of the recording laid on a fixed time grid, so trimming only adds or drops tiles at the ends
  if (w < 8 || b <= a) return "";
  const pxs = w / (b - a), tile = 64, step = Math.max(0.5, Math.round(tile / pxs * 2) / 2);
  let html = "";
  for (let t = Math.floor(a / step) * step; t < b; t += step) {
    const at = Math.min(ed.st.source_duration - 0.5, t + step / 2);
    html += `<img loading="lazy" src="/v1/clips/${ed.st.id}/frame.jpg?t=${at.toFixed(1)}" style="left:${(t - a) * pxs}px;width:${step * pxs + 1}px" alt="">`;
  }
  return `<div class="edfilm" data-a0="${a}" data-pxs="${pxs}">${html}</div>`;
}
function edDrawSource() {
  const s = ed.st, src = s.source_duration, pps = ed.spps, W = Math.ceil(src * pps) + 40;
  const step = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600].find(x => x * pps >= 70) || 3600;
  let ruler = "";
  for (let t = 0; t <= src; t += step) ruler += `<span style="left:${t * pps}px">${hms(t)}</span>`;
  const mks = s.markers.map((m, k) => `<i class="edmk ${edIsSel("mk", k) ? "on" : ""}" data-mk="${k}" style="transform:translateX(${m.src_s * pps}px)" title="${esc(m.label || "Marker")}: ${hms(m.src_s)}"><b>${esc(m.label || "⚑")}</b></i>`).join("");
  const parts = s.segments.map(([a, b], i) => `<div class="edblk seg src ${edIsSel("seg", i) ? "on" : ""}" data-i="${i}" style="left:${a * pps}px;width:${Math.max(4, (b - a) * pps)}px" title="Part ${i + 1} of your Short: ${hms(a)} to ${hms(b)}">
      ${edFilm(a, b, Math.max(4, (b - a) * pps))}<i class="h l" title="Drag left to add earlier footage"></i><span>${i + 1}</span><i class="h r" title="Drag right to add later footage"></i></div>`).join("");
  const marks = (ed.mA != null && !ed.pick ? `<i class="edmark" data-l="start" style="left:${ed.mA * pps}px"></i>` : "")
    + (ed.mB != null && !ed.pick ? `<i class="edmark" data-l="end" style="left:${ed.mB * pps}px"></i>` : "");
  const pick = ed.pick ? `<div class="edblk pick" id="edpick" style="left:${ed.pick[0] * pps}px;width:${Math.max(2, (ed.pick[1] - ed.pick[0]) * pps)}px"></div>` : `<div class="edblk pick" id="edpick" hidden></div>`;
  const lanes = [];
  const moms = (ed.moments || []).map((m, k) => {
    let lane = lanes.findIndex(end => end <= m.start_s); if (lane < 0) { lane = lanes.length; lanes.push(0); } lanes[lane] = m.end_s + 2 / pps;
    const inShort = s.segments.some(([a, b]) => m.start_s < b && m.end_s > a);
    const title = m.ai?.title || `Moment at ${hms(m.peak_s)}`;
    return `<div class="edblk mom ${inShort ? "in" : ""} ${ed.sel?.kind === "mom" && ed.sel.i === k ? "on" : ""}" data-k="${k}" style="left:${m.start_s * pps}px;width:${Math.max(4, (m.end_s - m.start_s) * pps)}px;top:${4 + lane * 26}px"
      title="#${m.rank} ${esc(title)}${m.tier ? " (" + m.tier + ")" : ""}${inShort ? " · already in this Short" : ""}"><span>#${m.rank} ${esc(title)}</span></div>`;
  }).join("");
  $("#edtl").innerHTML = `<div class="edinner" style="width:${W}px">
    <div class="edruler">${ruler}${mks}<span id="edhov" class="edhov" hidden></span></div>
    <div class="edtrack" id="edrec" style="height:48px"><b class="edlab">Full recording: orange = already in your Short</b>${parts}${pick}${marks}</div>
    <div class="edtrack" style="height:${Math.max(1, lanes.length) * 26 + 8}px"><b class="edlab">Moments SPHURA found (click one to mark it)</b>${ed.moments ? moms || `<span class="ednone">No other moments</span>` : `<span class="ednone">Loading…</span>`}</div>
    <div id="edph" class="edph" style="transform:translateX(${($("#edvid")?.currentTime || 0) * pps}px)"><i class="edphh" title="Drag to move through the recording"></i></div></div>`;
}
function edLayoutSource() {
  const pps = ed.spps;
  $("#edtl").querySelectorAll(".edblk.seg.src").forEach(el => {
    const i = +el.dataset.i, sg = ed.st.segments[i]; if (!sg) return;
    el.style.left = sg[0] * pps + "px"; el.style.width = Math.max(4, (sg[1] - sg[0]) * pps) + "px";
    const f = el.querySelector(".edfilm"); if (f) f.style.transform = `translateX(${(+f.dataset.a0 - sg[0]) * +f.dataset.pxs}px)`;
  });
  $("#edtl").querySelectorAll("[data-mk]").forEach(el => { const m = ed.st.markers[+el.dataset.mk]; if (m) el.style.transform = `translateX(${m.src_s * pps}px)`; });
}
function edSrcAt(clientX) {
  const r = $("#edtl .edinner").getBoundingClientRect();
  return Math.min(ed.st.source_duration, Math.max(0, (clientX - r.left) / ed.spps));
}
function edSelectRange(ev) {
  // Drag across the recording to mark a stretch to add; a plain click just moves there
  ev.preventDefault();
  const v = $("#edvid"), t0 = edSrcAt(ev.clientX);
  v.pause(); document.body.classList.add("edscrub"); edSeekTo(ev.clientX);
  let t1 = t0;
  edDrag(ev, (d, e) => {
    t1 = edSrcAt(e.clientX); edSeekTo(e.clientX);
    const el = $("#edpick"), a = Math.min(t0, t1), w = Math.abs(t1 - t0);
    el.hidden = false; el.style.left = a * ed.spps + "px"; el.style.width = Math.max(2, w * ed.spps) + "px";
  }, () => {
    document.body.classList.remove("edscrub");
    if (Math.abs(t1 - t0) >= 0.3) { edSetMarks(Math.min(t0, t1), Math.max(t0, t1)); ed.sel = { kind: "pick", i: 0 }; }
    else if (ed.sel?.kind === "pick") { edSetMarks(null, null); ed.sel = null; }
    edDraw(); edInspector();
  });
}
function edPickMoment(k) {
  const m = ed.moments[k]; edSetMarks(m.start_s, m.end_s);
  $("#edvid").currentTime = m.start_s; ed.sel = { kind: "mom", i: k }; ed.multi = []; edDraw(); edInspector(); setTimeout(edTick, 80);
}
function edSetMarks(a, b) {
  ed.mA = a; ed.mB = b;
  ed.pick = a != null && b != null && b - a >= 0.3 ? [a, b] : null;
}
function edMark(j) {
  // Mark start (I) / Mark end (O) at the playhead; a mark on the wrong side of the other clears it
  if (!ed.orig) return;
  const t = $("#edvid").currentTime || 0;
  let a = ed.mA ?? null, b = ed.mB ?? null;
  if (j === 0) { a = t; if (b != null && b - a < 0.3) b = null; } else { b = t; if (a != null && b - a < 0.3) a = null; }
  edSetMarks(a, b);
  ed.sel = ed.pick ? { kind: "pick", i: 0 } : null; edDraw(); edInspector();
}
function edAudioOf(i) { return { ...(ed.st.seg_audio[i] || { gain_db: 0, mute: false }) }; }
function edRemovePick() {
  // Cut the marked stretch out of every part it overlaps
  const [a, b] = ed.pick, out = [], au = [];
  ed.st.segments.forEach(([x, y], i) => {
    if (b <= x || a >= y) { out.push([x, y]); au.push(edAudioOf(i)); return; }
    if (a - x >= 0.3) { out.push([x, a]); au.push(edAudioOf(i)); }
    if (y - b >= 0.3) { out.push([b, y]); au.push(edAudioOf(i)); }
  });
  if (!out.length) return toast("That would remove everything. A Short needs at least one part.");
  ed.st.segments = out; ed.st.seg_audio = au; ed.dirtyCuts = true; edSetMarks(null, null); ed.sel = null; edCommit("Remove footage");
  edDraw(); edInspector(); toast("Removed from your Short. Press Update preview to see it.");
}
function edSorted(segs) { return segs.every((x, i) => !i || segs[i - 1][0] <= x[0]); }
function edAddPick() {
  // In recording order, overlapping or touching parts merge; a reordered Short gets the new part where it happens
  const [a, b] = ed.pick, s = ed.st;
  let segs, au;
  if (edSorted(s.segments)) {
    const tagged = [...s.segments.map((x, i) => [...x, i]), [a, b, -1]].sort((x, y) => x[0] - y[0]);
    segs = []; au = [];
    for (const [x, y, i] of tagged) {
      const last = segs[segs.length - 1];
      if (last && x <= last[1] + 0.05) last[1] = Math.max(last[1], y);
      else { segs.push([x, y]); au.push(i >= 0 ? edAudioOf(i) : { gain_db: 0, mute: false }); }
    }
  } else {
    let at = s.segments.findIndex(x => x[0] > a); if (at < 0) at = s.segments.length;
    segs = [...s.segments.slice(0, at), [a, b], ...s.segments.slice(at)];
    au = [...s.seg_audio.slice(0, at), { gain_db: 0, mute: false }, ...s.seg_audio.slice(at)];
  }
  const total = segs.reduce((t, [x, y]) => t + y - x, 0);
  if (total > 180) return toast(`That would make the Short ${Math.round(total)} s long; the most is 180 s. Trim it first.`);
  s.segments = segs; s.seg_audio = au; ed.dirtyCuts = true; edSetMarks(null, null); edCommit("Add footage");
  const i = segs.findIndex(([x, y]) => a >= x - 0.01 && a <= y);
  edSelect("seg", i < 0 ? 0 : i);
  toast("Added to your Short. Press Update preview to see it.");
}

// ---------- The Short's timeline ----------
function edTrackH(id, def) { return ed.th.closed[id] ? 12 : ed.th[id] || def; }
function edSaveTracks() { localStorage.setItem("sphura_tracks", JSON.stringify(ed.th)); }
function edTrack(id, label, h, body, extra = "") {
  const closed = !!ed.th.closed[id];
  return `<div class="edtrack ${closed ? "closed" : ""}" data-tr="${id}" style="height:${h}px">
      <b class="edlab"><button class="edcol" onclick="edCollapse('${id}')" aria-expanded="${!closed}" aria-label="${closed ? "Show" : "Hide"} the ${label} track">${closed ? "▸" : "▾"}</button>${label}${extra}</b>${closed ? "" : body}</div>
    <div class="edsplit" data-tr="${id}" title="Drag to resize the ${label} track; double-click to reset" role="separator" aria-orientation="horizontal"></div>`;
}
function edCollapse(id) { ed.th.closed[id] = !ed.th.closed[id]; edSaveTracks(); edDraw(); }
function edRuler(total, pps) {
  const step = [0.5, 1, 2, 5, 10, 15, 30, 60].find(x => x * pps >= 48) || 60;
  let html = "";
  for (let t = 0; t <= total + 1e-6; t += step) html += `<span style="left:${t * pps}px">${t >= 60 ? hms(t) : +t.toFixed(1) + "s"}</span>`;
  return html;
}
function edDraw() {
  if (!ed || !$("#edtl")) return;
  if (ed.orig) return edDrawSource();
  $("#edtl").classList.remove("edcutting");
  edLiveSync();
  const s = ed.st, total = Math.max(edTotal(), 1), pps = ed.pps, W = Math.ceil(total * pps) + 40, so = edSegOut();
  const teaser = s.teaser_out > 0.05 ? `<div class="edblk teaser" style="transform:translateX(0);width:${edOut(edLead()) * pps}px" title="Cold open (the climax first)">Cold open</div>` : "";
  const segs = so.map(([a, b], i) => {
    const w = Math.max(6, (b - a) * pps);
    return `<div class="edblk seg ${edIsSel("seg", i) ? "on" : ""}" data-seg="${i}" style="transform:translateX(${a * pps}px);width:${w}px">
      ${edFilm(s.segments[i][0], s.segments[i][1], w)}<i class="h l" title="Drag left to add earlier footage, right to cut the start"></i><span>Part ${i + 1} · ${(s.segments[i][1] - s.segments[i][0]).toFixed(1)}s</span><i class="h r" title="Drag right to add later footage, left to cut the end"></i></div>`;
  }).join("");
  const spd = ed.dirtyCuts ? "" : s.speed.map((p, k) => {
    const [a, b] = edSpeedOut(p);
    return `<button class="edspd ${edIsSel("spd", k) ? "on" : ""} ${p.owner === "user" ? "mine" : ""}" data-spd="${k}" style="transform:translateX(${a * pps}px);width:${Math.max(26, (b - a) * pps)}px"
      title="${p.freeze ? "Freeze frame" : p.factor < 1 ? "Slow motion" : p.factor > 1 ? "Sped up" : "Normal speed"} ${edFx2(p.factor)}× · ${(b - a).toFixed(2)} s${p.owner === "user" ? " · your change" : " · planned by SPHURA"}. Click to change, drag to move.">${edSpdLabel(p)}</button>`;
  }).join("");
  const lanes = []; // stack overlapping text blocks in lanes
  const txt = s.overlays.map((o, i) => {
    let lane = lanes.findIndex(end => end <= o.start_s + 0.01); if (lane < 0) { lane = lanes.length; lanes.push(0); } lanes[lane] = o.end_s;
    return `<div class="edblk txt ${edIsSel("txt", i) ? "on" : ""}" data-txt="${i}" style="transform:translateX(${o.start_s * pps}px);width:${Math.max(8, (Math.min(o.end_s, total) - o.start_s) * pps)}px;top:${4 + lane * 28}px">
      <span>${o.owner === "user" ? "" : "✦ "}${esc(o.text)}</span><i class="h r" title="Drag to change how long it shows"></i></div>`; }).join("");
  edLoadAudio();
  const au = ed.audio;
  const tzA = s.teaser_out > 0.05 ? `<div class="edblk aud tz" style="transform:translateX(0);width:${edOut(edLead()) * pps}px" title="Game audio of the cold open"><span>Cold open</span></div>` : "";
  const aud = so.map(([a, b], i) => {
    const p = s.seg_audio[i] || {}, vol = p.mute ? "🔇 Muted" : Math.abs(p.gain_db || 0) >= 0.05 ? `${p.gain_db > 0 ? "+" : ""}${p.gain_db} dB` : "";
    return `<div class="edblk aud ${edIsSel("seg", i) ? "on" : ""} ${p.mute ? "muted" : ""}" data-seg="${i}" style="transform:translateX(${a * pps}px);width:${Math.max(6, (b - a) * pps)}px" title="Game audio of part ${i + 1}: trimmed together with its video${vol ? " · " + vol : ""}">
      <i class="h l"></i><span>Part ${i + 1}${vol ? " · " + vol : ""}</span><i class="h r"></i></div>`;
  }).join("");
  const m = au && au.music;
  const mus = m ? `<div class="edblk mus ${ed.sel?.kind === "music" ? "on" : ""}" style="transform:translateX(${m.in_s * pps}px);width:${Math.max(6, (total - m.in_s) * pps)}px" title="${esc(m.name)}">
      <span>♪ ${esc(m.name)}${m.bpm ? ` · ${m.bpm} BPM` : ""}</span></div>` : `<span class="ednone">${au ? "No music on this Short" : "Loading…"}</span>`;
  const mks = s.markers.map((mk, k) => { const x = edSrcToOut(mk.src_s); return x == null ? "" :
    `<i class="edmk ${edIsSel("mk", k) ? "on" : ""}" data-mk="${k}" style="transform:translateX(${x * pps}px)" title="${esc(mk.label || "Marker")} · drag to move"><b>${esc(mk.label || "⚑")}</b></i>`; }).join("");
  const loop = ed.loop ? `<i class="edloop" style="transform:translateX(${ed.loop[0] * pps}px);width:${(ed.loop[1] - ed.loop[0]) * pps}px" title="Looping"></i>` : "";
  const th = { video: edTrackH("video", 48), audio: edTrackH("audio", 34), music: edTrackH("music", 34), text: Math.max(ed.th.closed.text ? 12 : 0, edTrackH("text", 0), ed.th.closed.text ? 0 : Math.max(1, lanes.length) * 28 + 8) };
  $("#edtl").innerHTML = `<div class="edinner" style="width:${W}px">
    <div class="edruler"><div class="edticks">${edRuler(total, pps)}</div>${loop}${mks}<span id="edhov" class="edhov" hidden></span></div>
    ${edTrack("video", "Video", th.video, teaser + segs + spd, ed.dirtyCuts && s.speed.length ? ` <span class="faint">· speed shows after Update preview</span>` : "")}
    ${edTrack("audio", "Game audio", th.audio, tzA + aud + `<canvas id="edcv-audio" class="edcv"></canvas>`)}
    ${edTrack("music", "Music", th.music, mus + `<canvas id="edcv-music" class="edcv"></canvas>`)}
    ${edTrack("text", "Text", th.text, txt)}
    <div id="edph" class="edph" style="transform:translateX(${edNow() * pps}px)"><i class="edphh" title="Drag to move through the Short"></i></div>
    <i id="edguide" class="edguide" hidden></i><i id="edins" class="edins" hidden></i><div id="edbox" class="edbox" hidden></div>
    <div id="edfab" class="edfab" hidden role="toolbar" aria-label="Actions for the selection"></div></div>`;
  ed.rulerPps = pps;
  edWaves(); edFab();
}
function edPos(el, x, w) { el.style.transform = `translateX(${x}px)`; if (w != null) el.style.width = w + "px"; }
function edReq() { if (!ed.raf) ed.raf = requestAnimationFrame(edLayout); }
function edLayout() {
  // Move what's already drawn (no rebuild) while dragging, zooming or changing speed
  ed.raf = 0;
  const tl = $("#edtl"); if (!tl || !ed) return;
  if (ed.orig) return edLayoutSource();
  const s = ed.st, pps = ed.pps, so = edSegOut(), total = Math.max(edTotal(), 1);
  const inner = tl.querySelector(".edinner"); if (inner) inner.style.width = Math.ceil(total * pps) + 40 + "px";
  if (ed.rulerPps !== pps) { const r = tl.querySelector(".edticks"); if (r) r.innerHTML = edRuler(total, pps); ed.rulerPps = pps; }
  tl.querySelectorAll("[data-seg]").forEach(el => {
    const i = +el.dataset.seg, r = so[i]; if (!r) return;
    const w = Math.max(6, (r[1] - r[0]) * pps); edPos(el, r[0] * pps, w);
    const f = el.querySelector(".edfilm"), sg = s.segments[i];
    if (f && sg) { const pxs = w / Math.max(0.01, sg[1] - sg[0]); f.style.transformOrigin = "0 0"; f.style.transform = `translateX(${(+f.dataset.a0 - sg[0]) * pxs}px) scaleX(${pxs / +f.dataset.pxs})`; }
  });
  tl.querySelectorAll(".edblk.teaser, .edblk.aud.tz").forEach(el => el.style.width = edOut(edLead()) * pps + "px");
  tl.querySelectorAll("[data-txt]").forEach(el => { const o = s.overlays[+el.dataset.txt]; if (o) edPos(el, o.start_s * pps, Math.max(8, (Math.min(o.end_s, total) - o.start_s) * pps)); });
  tl.querySelectorAll("[data-spd]").forEach(el => { const p = s.speed[+el.dataset.spd]; if (!p) return; const [a, b] = edSpeedOut(p); edPos(el, a * pps, Math.max(26, (b - a) * pps)); el.textContent = edSpdLabel(p); });
  tl.querySelectorAll("[data-mk]").forEach(el => { const m = s.markers[+el.dataset.mk], x = m && edSrcToOut(m.src_s); if (x != null) edPos(el, x * pps); });
  const mus = tl.querySelector(".edblk.mus"), m = ed.audio?.music; if (mus && m) edPos(mus, m.in_s * pps, Math.max(6, (total - m.in_s) * pps));
  const lp = tl.querySelector(".edloop"); if (lp && ed.loop) edPos(lp, ed.loop[0] * pps, (ed.loop[1] - ed.loop[0]) * pps);
  const ph = $("#edph"); if (ph) ph.style.transform = `translateX(${edNow() * pps}px)`;
  edWaves(); edFab();
}

// ---------- Selection ----------
function edSelValid(x) {
  if (!x) return false;
  const s = ed.st;
  return x.kind === "seg" ? !!s.segments[x.i] : x.kind === "txt" ? !!s.overlays[x.i] : x.kind === "spd" ? !!s.speed[x.i] && !ed.dirtyCuts
    : x.kind === "mk" ? !!s.markers[x.i] : true;
}
function edSelAll() { return [ed.sel, ...ed.multi].filter(x => x && edSelValid(x)); }
function edIsSel(kind, i) { return edSelAll().some(x => x.kind === kind && x.i === i); }
function edSelect(kind, i) { ed.sel = kind ? { kind, i } : null; ed.multi = []; edMarkSel(); edInspector(); edDrawText(edNow()); }
function edToggleSel(kind, i) {
  // Shift+click adds or removes parts and texts from the selection
  if (ed.sel?.kind === kind && ed.sel.i === i) { ed.sel = ed.multi.shift() || null; }
  else if (ed.multi.some(x => x.kind === kind && x.i === i)) ed.multi = ed.multi.filter(x => !(x.kind === kind && x.i === i));
  else if (!ed.sel || !["seg", "txt"].includes(ed.sel.kind)) { ed.sel = { kind, i }; ed.multi = []; }
  else ed.multi.push({ kind, i });
  edMarkSel(); edInspector(); edDrawText(edNow());
}
function edMarkSel() {
  if (ed.orig) return edDraw();
  const tl = $("#edtl"); if (!tl) return;
  tl.querySelectorAll("[data-seg]").forEach(el => el.classList.toggle("on", edIsSel("seg", +el.dataset.seg)));
  tl.querySelectorAll("[data-txt]").forEach(el => el.classList.toggle("on", edIsSel("txt", +el.dataset.txt)));
  tl.querySelectorAll("[data-spd]").forEach(el => el.classList.toggle("on", edIsSel("spd", +el.dataset.spd)));
  tl.querySelectorAll("[data-mk]").forEach(el => el.classList.toggle("on", edIsSel("mk", +el.dataset.mk)));
  tl.querySelector(".edblk.mus")?.classList.toggle("on", ed.sel?.kind === "music");
  edFab();
}
function edSelRange(x) {
  // Where a selected item sits on the Short's timeline
  const s = ed.st;
  if (x.kind === "seg") return edSegOut()[x.i];
  if (x.kind === "txt") return [s.overlays[x.i].start_s, s.overlays[x.i].end_s];
  if (x.kind === "spd") return edSpeedOut(s.speed[x.i]);
  if (x.kind === "mk") { const t = edSrcToOut(s.markers[x.i].src_s); return t == null ? null : [t, t]; }
  return null;
}
function edFab() {
  // Floating actions next to the selection; moves aside so it never covers the playhead
  const fab = $("#edfab"); if (!fab) return;
  const x = ed.sel, r = x && !ed.multi.length && !ed.orig && edSelValid(x) && edSelRange(x);
  if (!r) { fab.hidden = true; fab.dataset.key = ""; return; }
  const key = JSON.stringify([x, ed.st.seg_audio[x.i]?.mute, ed.st.speed[x.i]?.factor, x.kind === "seg" && edSplitOk(), ed.dirtyCuts]);
  if (fab.dataset.key !== key) {
    fab.dataset.key = key;
    const mute = ed.st.seg_audio[x.i]?.mute;
    fab.innerHTML = x.kind === "seg" ? edBtn(ic("scissors"), edSplitOk(), "Move the playhead inside this part to split it", "edSplit()", "", "Split at the playhead")
        + edBtn("⏱", !ed.dirtyCuts, "Update the preview first: speed is re-planned after a cut change", "edFocusSpeed()", "", "Speed of this part")
        + edBtn(mute ? "🔈" : "🔇", true, "", "edMute()", "", mute ? "Unmute this part" : "Mute this part")
        + edBtn(ic("trash"), ed.st.segments.length > 1, "A Short needs at least one part", "edDelete()", "", "Delete this part")
      : x.kind === "txt" ? edBtn(ic("copy"), true, "", "edDuplicate()", "", "Duplicate the text") + edBtn("↦", true, "", `edJump(${ed.st.overlays[x.i].start_s})`, "", "Go to the text") + edBtn(ic("trash"), true, "", "edDelete()", "", "Delete the text")
      : x.kind === "spd" ? [0.5, 1, 2].map(f => edBtn(f + "×", true, "", `edSetFactor(${x.i},${f})`, "", `Make it ${f}×`)).join("") + edBtn(ic("trash"), true, "", "edDelete()", "", "Delete the speed change")
      : x.kind === "mk" ? edBtn("↦", true, "", `edJump(${r[0]})`, "", "Go to the marker") + edBtn(ic("trash"), true, "", "edDelete()", "", "Delete the marker") : "";
  }
  fab.hidden = false;
  const pps = ed.pps, w = fab.offsetWidth || 120, box = $("#edtl"), ph = edNow() * pps;
  const lo = box.scrollLeft + 4, hi = box.scrollLeft + box.clientWidth - w - 4;
  let left = Math.min(hi, Math.max(lo, (r[0] + r[1]) / 2 * pps - w / 2));
  if (ph > left - 6 && ph < left + w + 6) left = ph + w + 10 <= hi + w ? ph + 10 : ph - w - 10;
  fab.style.transform = `translateX(${left}px)`;
}

// ---------- Pointer input ----------
function edDown(ev) {
  if (ev.button !== 0 || !ed) return;
  const t = ev.target;
  if (t.closest("#edfab, .edcol")) return; // their buttons handle the click
  const split = t.closest(".edsplit"); if (split) return edResizeTrack(ev, split.dataset.tr);
  const mk = t.closest("[data-mk]"); if (mk) return edDragMarker(ev, +mk.dataset.mk);
  if (t.closest(".edruler") || t.closest(".edphh")) return edScrub(ev);
  if (ed.orig) {
    const sg = t.closest(".edblk.seg"); if (sg) return edDragSeg(ev, +sg.dataset.i);
    const m = t.closest(".edblk.mom"); if (m) { ev.preventDefault(); return edPickMoment(+m.dataset.k); }
    if (t.closest("#edrec")) return edSelectRange(ev);
    return edScrub(ev);
  }
  const sp = t.closest("[data-spd]"); if (sp) return edDragSpeed(ev, +sp.dataset.spd);
  const sg = t.closest("[data-seg]"); if (sg) return edDragSeg(ev, +sg.dataset.seg);
  const tx = t.closest("[data-txt]"); if (tx) return edDragTxt(ev, +tx.dataset.txt);
  if (t.closest(".edblk.mus")) { ev.preventDefault(); return edSelect("music", 0); }
  if (t.closest(".edblk.teaser, .edblk.tz")) return edScrub(ev);
  return edBoxOrSeek(ev);
}
function edDrag(ev, onMove, onUp) {
  // Drag in timeline seconds; scrolls when the pointer nears an edge and redraws once per frame
  ev.preventDefault(); ev.stopPropagation();
  const box = $("#edtl"), x0 = ev.clientX, sl0 = box.scrollLeft, scale = ed.orig ? ed.spps : ed.pps;
  let last = ev, roll = 0;
  try { box.setPointerCapture(ev.pointerId); } catch {}
  const apply = () => { onMove((last.clientX - x0 + box.scrollLeft - sl0) / scale, last); edReq(); };
  const edgeRoll = () => {
    const b = box.getBoundingClientRect(), dx = last.clientX > b.right - 30 ? 14 : last.clientX < b.left + 30 ? -14 : 0;
    if (dx && (dx < 0 ? box.scrollLeft > 0 : box.scrollLeft < box.scrollWidth - box.clientWidth)) { box.scrollLeft += dx; apply(); roll = requestAnimationFrame(edgeRoll); }
    else roll = 0;
  };
  const move = e => { last = e; apply(); if (!roll) roll = requestAnimationFrame(edgeRoll); };
  const up = e => {
    window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); window.removeEventListener("pointercancel", up);
    cancelAnimationFrame(roll); edGuide(null); onUp && onUp(e);
  };
  window.addEventListener("pointermove", move); window.addEventListener("pointerup", up); window.addEventListener("pointercancel", up);
}
function edTargets(skip = {}) {
  const s = ed.st, out = [0, edTotal()];
  if (!skip.ph) out.push(edNow());
  edSegOut().forEach(([a, b], i) => { if (skip.seg !== i) out.push(a, b); });
  s.overlays.forEach((o, i) => { if (skip.txt !== i) out.push(o.start_s, o.end_s); });
  s.markers.forEach((m, k) => { if (skip.mk !== k) { const x = edSrcToOut(m.src_s); if (x != null) out.push(x); } });
  if (!ed.dirtyCuts) { s.speed.forEach((p, k) => { if (skip.spd !== k) out.push(...edSpeedOut(p)); }); if (ed.audio?.music) out.push(...ed.audio.music.beats); }
  return out;
}
function edSnapShift(points, e, skip) {
  // How far to move so the closest of points lands on a snap target within 8 px (Alt: none)
  if (!ed.snap || e?.altKey || ed.orig) { edGuide(null); return 0; }
  let best = null, d = 8 / ed.pps;
  const targets = edTargets(skip);
  for (const p of points) for (const x of targets) if (Math.abs(x - p) < d) { d = Math.abs(x - p); best = [x, x - p]; }
  edGuide(best && best[0]); return best ? best[1] : 0;
}
function edGuide(t) { const g = $("#edguide"); if (!g) return; g.hidden = t == null; if (t != null) g.style.transform = `translateX(${t * ed.pps}px)`; }
function edSnapToggle() { ed.snap = !ed.snap; localStorage.setItem("sphura_snap", ed.snap ? "on" : "off"); const b = $("#edsnapbtn"); b.classList.toggle("on", ed.snap); b.setAttribute("aria-pressed", ed.snap); }
function edSeekTo(clientX, e) {
  if (ed.orig) {
    const t = edSrcAt(clientX); $("#edvid").currentTime = t;
    $("#edph").style.transform = `translateX(${t * ed.spps}px)`; $("#edtime") && ($("#edtime").textContent = edSrcLabel(t));
    return;
  }
  const r = $("#edtl .edinner").getBoundingClientRect(), total = edTotal();
  let t = Math.min(total, Math.max(0, (clientX - r.left) / ed.pps));
  if (e) t += edSnapShift([t], e, { ph: true });
  edGoto(t);
  // Move the playhead and text right away; the video frame follows when it has seeked
  $("#edph").style.transform = `translateX(${t * ed.pps}px)`;
  $("#edtime") && ($("#edtime").textContent = `${t.toFixed(1)} s / ${total.toFixed(1)} s`);
  const sp = $("#edsplitwrap button"); if (sp) sp.disabled = !edSplitOk();
  edDrawText(t); edFab();
}
function edScrub(ev) {
  // Click or drag on the ruler, the playhead or a cold-open block to move through the Short
  const v = $("#edvid"), playing = !v.paused;
  v.pause(); edRev(false); document.body.classList.add("edscrub"); edSeekTo(ev.clientX, ev);
  edDrag(ev, (d, e) => edSeekTo(e.clientX, e), () => {
    document.body.classList.remove("edscrub"); if (playing) v.play(); else setTimeout(edTick, 80);
  });
}
function edBoxOrSeek(ev) {
  // A click on empty track space moves the playhead; a drag draws a box that selects parts and texts
  const inner = $("#edtl .edinner"), r0 = inner.getBoundingClientRect(), x0 = ev.clientX - r0.left, y0 = ev.clientY - r0.top, box = $("#edbox");
  let moved = false, x1 = x0, y1 = y0;
  edDrag(ev, (d, e) => {
    const r = inner.getBoundingClientRect(); x1 = e.clientX - r.left; y1 = e.clientY - r.top;
    if (!moved && Math.hypot(x1 - x0, y1 - y0) < 5) return;
    moved = true; box.hidden = false;
    Object.assign(box.style, { left: Math.min(x0, x1) + "px", top: Math.min(y0, y1) + "px", width: Math.abs(x1 - x0) + "px", height: Math.abs(y1 - y0) + "px" });
  }, () => {
    box.hidden = true;
    if (!moved) { edSeekTo(ev.clientX); setTimeout(edTick, 80); if (!ev.shiftKey) edSelect(null); return; }
    const r = inner.getBoundingClientRect(), L = Math.min(x0, x1), R = Math.max(x0, x1), T = Math.min(y0, y1), B = Math.max(y0, y1), got = [];
    inner.querySelectorAll("[data-seg], [data-txt]").forEach(el => {
      const b = el.getBoundingClientRect(), bl = b.left - r.left, bt = b.top - r.top;
      if (bl < R && bl + b.width > L && bt < B && bt + b.height > T) {
        const item = el.dataset.seg != null ? { kind: "seg", i: +el.dataset.seg } : { kind: "txt", i: +el.dataset.txt };
        if (!got.some(x => x.kind === item.kind && x.i === item.i)) got.push(item);
      }
    });
    const keep = ev.shiftKey ? edSelAll().filter(x => ["seg", "txt"].includes(x.kind)) : [];
    const all = [...keep, ...got.filter(g => !keep.some(x => x.kind === g.kind && x.i === g.i))];
    ed.sel = all[0] || null; ed.multi = all.slice(1); edMarkSel(); edInspector(); edDrawText(edNow());
  });
}
function edResizeTrack(ev, id) {
  ev.preventDefault();
  const el = $(`#edtl .edtrack[data-tr="${id}"]`), y0 = ev.clientY, h0 = el.clientHeight;
  const move = e => {
    ed.th[id] = Math.round(Math.min(240, Math.max(20, h0 + e.clientY - y0))); delete ed.th.closed[id];
    el.style.height = ed.th[id] + "px"; edWaves();
  };
  const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); edSaveTracks(); edDraw(); };
  window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
}
function edDragSeg(ev, i) {
  const edge = ev.target.classList.contains("l") ? "l" : ev.target.classList.contains("r") ? "r" : null;
  if (!edge) {
    if (ev.shiftKey && !ed.orig) return edToggleSel("seg", i);
    if (!edIsSel("seg", i)) edSelect("seg", i);
    if (!ed.orig) edReorder(ev, i);
    return;
  }
  edSelect("seg", i);
  const segs = ed.st.segments, seg = segs[i], [a0, b0] = seg, src = ed.st.source_duration;
  // Grow up to the neighbouring part in the recording, so no footage is used twice
  const lo = Math.max(0, ...segs.filter((x, k) => k !== i && x[1] <= a0 + 0.01).map(x => x[1]));
  const hi = Math.min(src, ...segs.filter((x, k) => k !== i && x[0] >= b0 - 0.01).map(x => x[0]));
  const tv = ed.orig ? $("#edvid") : $("#edtrimv"), lab = $("#edtrimlab");
  if (!ed.orig && !tv.src) tv.src = `/v1/clips/${ed.st.id}/source`;
  if (!ed.orig) tv.hidden = false;
  let want = null, busy = false;
  const show = t => { // one seek at a time keeps the preview responsive
    want = t;
    if (tv.readyState < 1) { tv.onloadedmetadata = () => show(want); return; } // not loaded yet: seek once it is
    if (busy) return;
    busy = true; tv.currentTime = t;
    const done = () => { busy = false; if (Math.abs(want - tv.currentTime) > 0.05) show(want); };
    tv.onseeked = done; setTimeout(() => busy && done(), 400); // never get stuck on a lost event
  };
  const label = () => {
    const t = edge === "l" ? seg[0] : seg[1], grown = edge === "l" ? a0 - seg[0] : seg[1] - b0;
    lab.hidden = false;
    lab.textContent = `Part ${i + 1} ${edge === "l" ? "starts" : "ends"} at ${hms(t)} · ${(seg[1] - seg[0]).toFixed(1)} s`
      + (Math.abs(grown) >= 0.05 ? ` (${grown > 0 ? "+" : "−"}${Math.abs(grown).toFixed(1)} s)` : "")
      + ((edge === "l" ? seg[0] <= lo + 0.01 : seg[1] >= hi - 0.01) ? (edge === "l" ? (lo > 0 ? " · reached the previous part" : " · start of the recording") : (hi < src ? " · reached the next part" : " · end of the recording")) : "");
  };
  show(edge === "l" ? a0 : b0); label();
  ed.pending = `Trim part ${i + 1}`;
  edDrag(ev, (d, e) => {
    // First change: the timeline switches to the 1:1 layout of unrendered cuts (redrawn on release)
    if (!ed.dirtyCuts && !ed.orig) $("#edtl").classList.add("edcutting");
    ed.dirtyCuts = true;
    if (edge === "l") seg[0] = Math.min(Math.max(lo, a0 + d), seg[1] - 0.3);
    else {
      seg[1] = Math.max(Math.min(hi, b0 + d), seg[0] + 0.3);
      const so = edSegOut()[i], sh = edSnapShift([so[1]], e, { seg: i });
      if (sh) seg[1] = Math.max(Math.min(hi, seg[1] + sh), seg[0] + 0.3);
    }
    show(edge === "l" ? seg[0] : seg[1]); label();
  }, () => {
    if (!ed.orig) tv.hidden = true; lab.hidden = true;
    edDraw(); edInspector();
    setTimeout(() => { // land the playhead on the new edge so Play starts there
      if (ed.orig) return;
      edLiveSync(); const so = edSegOut()[i]; if (!so) return;
      const t = edge === "l" ? so[0] : Math.max(so[0], so[1] - 1);
      const v = $("#edvid"); if (v.readyState >= 1) edGoto(t); else v.onloadedmetadata = () => edGoto(t);
      setTimeout(edTick, 120);
    }, 0);
  });
}
function edReorder(ev, i) {
  // Drag a part's body to another place in the Short; a line shows where it will land
  const els = [...$("#edtl").querySelectorAll(`[data-seg="${i}"]`)], line = $("#edins"), so = edSegOut(), pps = ed.pps;
  const others = so.map((r, k) => [r, k]).filter(([, k]) => k !== i);
  let moved = false, to = i;
  const inner = $("#edtl .edinner");
  edDrag(ev, (d, e) => {
    if (!moved && Math.abs(d * pps) < 5) return;
    moved = true; els.forEach(el => { el.classList.add("ghost"); el.style.transform = `translateX(${(so[i][0] + d) * pps}px)`; });
    const at = (e.clientX - inner.getBoundingClientRect().left) / pps; // lands where the pointer is
    to = others.filter(([r]) => (r[0] + r[1]) / 2 < at).length;
    const x = to === 0 ? (others[0]?.[0][0] ?? 0) : others[to - 1][0][1];
    line.hidden = false; line.style.transform = `translateX(${x * pps}px)`;
  }, () => {
    line.hidden = true; els.forEach(el => el.classList.remove("ghost"));
    if (!moved || to === i) { edLayout(); return; }
    edMovePart(i, to);
  });
}
function edMovePart(i, to) {
  const s = ed.st; if (to < 0 || to >= s.segments.length || to === i) return;
  const [seg] = s.segments.splice(i, 1), [au] = s.seg_audio.splice(i, 1);
  s.segments.splice(to, 0, seg); s.seg_audio.splice(to, 0, au || { gain_db: 0, mute: false });
  ed.dirtyCuts = true; ed.sel = { kind: "seg", i: to }; ed.multi = [];
  edCommit(`Move part ${i + 1}`); edDraw(); edInspector();
}
function edRecordingOrder() {
  const s = ed.st, order = s.segments.map((x, i) => i).sort((a, b) => s.segments[a][0] - s.segments[b][0]);
  s.segments = order.map(i => s.segments[i]); s.seg_audio = order.map(i => s.seg_audio[i] || { gain_db: 0, mute: false });
  ed.dirtyCuts = true; ed.sel = null; ed.multi = []; edCommit("Put parts in recording order"); edDraw(); edInspector();
}
function edDragTxt(ev, i) {
  if (ev.shiftKey) return edToggleSel("txt", i);
  if (!edIsSel("txt", i)) edSelect("txt", i);
  const o = ed.st.overlays[i], s0 = o.start_s, e0 = o.end_s, total = edTotal(), resize = ev.target.classList.contains("r");
  ed.pending = resize ? "Change text length" : "Move text in time";
  edDrag(ev, (d, e) => {
    if (resize) { o.end_s = Math.max(o.start_s + 0.2, Math.min(total, e0 + d)); o.end_s = Math.max(o.start_s + 0.2, o.end_s + edSnapShift([o.end_s], e, { txt: i })); }
    else {
      const len = e0 - s0; o.start_s = Math.max(0, Math.min(total - len, s0 + d));
      o.start_s = Math.max(0, Math.min(total - len, o.start_s + edSnapShift([o.start_s, o.start_s + len], e, { txt: i }))); o.end_s = o.start_s + len;
    }
    edDrawText(edNow());
  }, () => { edDraw(); edInspector(); });
}
function edDragSpeed(ev, k) {
  edSelect("spd", k);
  const p = ed.st.speed[k], a0 = p.start_s, len = p.end_s - p.start_s, others = ed.st.speed.filter((x, j) => j !== k);
  const lo = Math.max(0, ...others.filter(x => x.end_s <= a0 + 1e-6).map(x => x.end_s));
  const hi = Math.min(edPreTotal(), ...others.filter(x => x.start_s >= p.end_s - 1e-6).map(x => x.start_s));
  ed.pending = "Move speed change";
  edDrag(ev, d => { p.start_s = +Math.min(hi - len, Math.max(lo, a0 + d)).toFixed(3); p.end_s = +(p.start_s + len).toFixed(3); },
    () => { edFlags(); edDraw(); edInspector(); edNote(); });
}
function edDragMarker(ev, k) {
  if (!edIsSel("mk", k)) edSelect("mk", k);
  const m = ed.st.markers[k], src0 = m.src_s, out0 = ed.orig ? null : edSrcToOut(src0);
  ed.pending = "Move marker";
  edDrag(ev, (d, e) => {
    if (ed.orig) { m.src_s = +Math.min(ed.st.source_duration, Math.max(0, src0 + d)).toFixed(3); return; }
    let t = Math.min(edTotal(), Math.max(0, out0 + d)); t += edSnapShift([t], e, { mk: k });
    const src = edOutToSrc(t); if (src != null) m.src_s = src;
  }, () => { edDraw(); edInspector(); });
}

// ---------- Context menu, help ----------
function edContext(ev) {
  ev.preventDefault(); if (!ed) return;
  const t = ev.target, inner = $("#edtl .edinner"), at = (ev.clientX - inner.getBoundingClientRect().left) / (ed.orig ? ed.spps : ed.pps);
  const pick = (kind, i) => { if (!edIsSel(kind, i)) edSelect(kind, i); };
  const mk = t.closest("[data-mk]"), sp = t.closest("[data-spd]"), sg = t.closest("[data-seg]"), tx = t.closest("[data-txt]");
  if (mk) pick("mk", +mk.dataset.mk); else if (sp) pick("spd", +sp.dataset.spd); else if (sg) pick("seg", +sg.dataset.seg); else if (tx) pick("txt", +tx.dataset.txt);
  else if (t.closest(".edblk.mus")) edSelect("music", 0); else if (!ed.orig) edSelect(null);
  edMenu(ev.clientX, ev.clientY, edMenuItems(mk || sp || sg || tx ? null : at));
}
function edMenuItems(at) {
  // The same actions as the toolbar and keys, for the selection (or the empty spot at time `at`)
  const s = ed.st, items = edSelAll(), x = ed.sel, why = "Update the preview first: speed is re-planned after a cut change";
  if (ed.orig) return [{ label: "Add a marker here", kbd: "B", fn: () => edAddMarker(at) }, { label: "Back to the Short", fn: () => edMode(false) }];
  if (items.length > 1) {
    const segs = items.filter(y => y.kind === "seg");
    return [segs.length ? { label: "Mute or unmute the selected parts", kbd: "M", fn: edMute } : null,
      { label: "Duplicate", kbd: "D", fn: edDuplicate },
      { label: `Delete ${items.length} items`, kbd: "Del", fn: edDelete, off: segs.length >= s.segments.length && "A Short needs at least one part" }].filter(Boolean);
  }
  if (x?.kind === "seg") {
    const i = x.i, mute = s.seg_audio[i]?.mute;
    return [{ label: "Split at the playhead", kbd: "S", fn: edSplit, off: !edSplitOk() && "Move the playhead inside this part first" },
      { label: "Duplicate", kbd: "D", fn: edDuplicate },
      { label: mute ? "Unmute" : "Mute", kbd: "M", fn: edMute },
      { label: "Slow motion (0.5×)", fn: () => edPartSpeed(i, 0.5), off: ed.dirtyCuts && why },
      { label: "Normal speed", fn: () => edPartSpeed(i, 1), off: ed.dirtyCuts && why }, "-",
      { label: "Move earlier", fn: () => edMovePart(i, i - 1), off: i === 0 && "It's already first" },
      { label: "Move later", fn: () => edMovePart(i, i + 1), off: i === s.segments.length - 1 && "It's already last" },
      { label: "Put all parts in recording order", fn: edRecordingOrder, off: edSorted(s.segments) && "They're already in recording order" },
      { label: "Loop this part", fn: () => { ed.loop = null; edLoopToggle(); } }, "-",
      { label: "Delete part", kbd: "Del", fn: edDelete, off: s.segments.length < 2 && "A Short needs at least one part" }];
  }
  if (x?.kind === "txt") return [{ label: "Duplicate", kbd: "D", fn: edDuplicate }, { label: "Go to", fn: () => edJump(s.overlays[x.i].start_s) },
    { label: "Loop this text", fn: () => { ed.loop = null; edLoopToggle(); } }, "-", { label: "Delete text", kbd: "Del", fn: edDelete }];
  if (x?.kind === "spd") return [...[0.25, 0.5, 0.75, 1.5, 2].map(f => ({ label: `Make it ${f}×`, fn: () => edSetFactor(x.i, f) })), "-", { label: "Delete speed change", kbd: "Del", fn: edDelete }];
  if (x?.kind === "mk") return [{ label: "Go to", fn: () => edJump(edSrcToOut(s.markers[x.i].src_s)) }, { label: "Delete marker", kbd: "Del", fn: edDelete }];
  if (x?.kind === "music") return [{ label: "Change the song on the Shorts page", fn: () => toast("Use Change under this Short on the Shorts page") }];
  return [{ label: "Add a marker here", kbd: "B", fn: () => edAddMarker(at) }, { label: "Add text here", fn: () => { if (at != null) edGoto(at); setTimeout(edAddText, 60); } },
    { label: ed.loop ? "Stop looping" : "Loop the whole Short", fn: () => { ed.sel = null; edLoopToggle(); } },
    { label: "Fit the Short in view", kbd: "Z", fn: edFitZoom }, { label: ed.snap ? "Turn snapping off" : "Turn snapping on", fn: edSnapToggle }];
}
function edMenu(x, y, items) {
  edMenuClose(false);
  const m = document.createElement("div"); m.id = "edmenu"; m.className = "edmenu"; m.setAttribute("role", "menu"); m.setAttribute("aria-label", "Timeline actions");
  m.innerHTML = items.map((it, k) => it === "-" ? `<hr role="separator">` : `<button role="menuitem" tabindex="-1" data-k="${k}" ${it.off ? `aria-disabled="true" title="${esc(it.off)}"` : ""}>${esc(it.label)}${it.kbd ? `<kbd>${it.kbd}</kbd>` : ""}</button>`).join("");
  document.body.appendChild(m);
  m.style.left = Math.min(x, innerWidth - m.offsetWidth - 8) + "px"; m.style.top = Math.min(y, innerHeight - m.offsetHeight - 8) + "px";
  const btns = [...m.querySelectorAll("button")], focus = k => { btns[(k + btns.length) % btns.length]?.focus(); };
  m.onclick = e => { const b = e.target.closest("button"); if (!b || b.getAttribute("aria-disabled")) return; const it = items[+b.dataset.k]; edMenuClose(); it.fn(); setTimeout(() => edCommit(), 0); };
  m.onkeydown = e => {
    const k = btns.indexOf(document.activeElement);
    if (e.key === "ArrowDown") { e.preventDefault(); focus(k + 1); } else if (e.key === "ArrowUp") { e.preventDefault(); focus(k - 1); }
    else if (e.key === "Home") { e.preventDefault(); focus(0); } else if (e.key === "End") { e.preventDefault(); focus(-1); }
    else if (e.key === "Escape" || e.key === "Tab") { e.preventDefault(); edMenuClose(); }
    e.stopPropagation();
  };
  focus(btns.findIndex(b => !b.getAttribute("aria-disabled")));
}
function edMenuClose(refocus = true) { const m = $("#edmenu"); if (!m) return; m.remove(); if (refocus) $("#edtl")?.focus({ preventScroll: true }); }
function edMenuKey() {
  // Shift+F10 or the Menu key: the menu for the selection, opened next to it
  const x = ed.sel, el = x && $(`#edtl [data-${{ seg: "seg", txt: "txt", spd: "spd", mk: "mk" }[x.kind]}="${x.i}"]`);
  const r = (el || $("#edph") || $("#edtl")).getBoundingClientRect();
  edMenu(r.left + Math.min(40, r.width / 2), r.bottom, edMenuItems(el ? null : edNow()));
}
const ED_HELP = [["Space", "Play / pause"], ["J · K · L", "Play backwards (stepped) · stop · play (again: faster)"], ["← →", "One frame back / forward"],
  ["Shift + ← →", "One second back / forward"], ["Home · End", "Start · end of the Short"], ["S", "Split at the playhead"], ["Del", "Delete the selection"],
  ["D", "Duplicate the selected part or text"], ["M", "Mute / unmute the selected parts"], ["B", "Add a marker at the playhead"],
  ["+ · − · Z", "Zoom in · out · fit the Short"], ["Ctrl + scroll / pinch", "Zoom around the pointer"], ["Shift + scroll", "Scroll sideways"],
  ["Shift + click, drag a box", "Select several parts and texts"], ["Alt while dragging", "No snapping"], ["Shift + F10, Menu key, right-click", "Actions for the selection"],
  ["Esc", "Clear the selection"], ["Ctrl + Z · Ctrl + Shift + Z", "Undo · redo"], ["I · O", "Mark start · end (full recording)"], ["?", "This list"]];
function edHelp() {
  if ($("#edhelpdlg")) return;
  const d = document.createElement("div"); d.id = "edhelpdlg"; d.className = "edhelpdlg"; d.setAttribute("role", "dialog"); d.setAttribute("aria-modal", "true"); d.setAttribute("aria-label", "Keyboard shortcuts");
  d.innerHTML = `<div class="card"><div class="row"><h2 class="grow" style="margin:0">Keyboard shortcuts</h2><button id="edhelpx" aria-label="Close">${ic("x")}</button></div>
    <table class="small">${ED_HELP.map(([k, v]) => `<tr><td><kbd>${esc(k)}</kbd></td><td>${esc(v)}</td></tr>`).join("")}</table></div>`;
  const close = () => { d.remove(); $("#edtl")?.focus({ preventScroll: true }); };
  d.onclick = e => { if (e.target === d || e.target.closest("#edhelpx")) close(); };
  d.onkeydown = e => { if (e.key === "Escape" || e.key === "?") { e.preventDefault(); close(); } e.stopPropagation(); };
  document.body.appendChild(d); $("#edhelpx").focus();
}

// ---------- Inspector ----------
function edInspector() {
  const s = ed.st, el = $("#edinspector"); if (!el) return;
  let html = "";
  const items = edSelAll();
  if (items.length > 1) {
    const segs = items.filter(x => x.kind === "seg"), txts = items.filter(x => x.kind === "txt");
    html = `<h2>${[segs.length && `${segs.length} part${segs.length > 1 ? "s" : ""}`, txts.length && `${txts.length} text${txts.length > 1 ? "s" : ""}`].filter(Boolean).join(", ")} selected</h2>
      <div class="small muted">Shift+click adds or removes one; drag a box over empty track space to pick several.</div>
      <div class="row" style="margin-top:10px">${segs.length ? `<button onclick="edMute()">🔇 Mute / unmute parts<kbd>M</kbd></button>` : ""}
        <button onclick="edDuplicate()">${ic("copy")} Duplicate<kbd>D</kbd></button>
        <button onclick="edDelete()" ${segs.length >= s.segments.length ? "disabled title='A Short needs at least one part'" : ""}>${ic("trash")} Delete<kbd>Del</kbd></button></div>`;
  } else if (ed.sel?.kind === "txt" && s.overlays[ed.sel.i]) {
    const o = s.overlays[ed.sel.i], i = ed.sel.i;
    html = `<h2>Text · ${(o.end_s - o.start_s).toFixed(1)} s ${o.owner === "user" ? `<span class="badge">yours</span>` : `<span class="badge" title="Written by SPHURA: it's re-planned when you change cuts, until you edit it">✦ AI</span>`}</h2>
      <textarea rows="2" style="width:100%" aria-label="Text" oninput="ed.st.overlays[${i}].text=this.value;ed.pending='Edit text';edDraw();edDrawText(edNow())">${esc(o.text)}</textarea>
      <div class="edrow"><label>Style</label><select aria-label="Text style" onchange="ed.st.overlays[${i}].style=this.value;ed.pending='Change text style';edDrawText(edNow())">${Object.entries(ED_STYLES).filter(([k]) => k !== "bar" || s.style === "meme" || s.caption_bar).map(([k, l]) => `<option value="${k}" ${k === o.style ? "selected" : ""}>${l}</option>`).join("")}</select></div>
      <div class="edrow"><label>Shows</label><input type="number" step="0.1" aria-label="Starts at" value="${o.start_s.toFixed(1)}" onchange="edSetTime(${i},'start_s',this.value)"> to <input type="number" step="0.1" aria-label="Ends at" value="${o.end_s.toFixed(1)}" onchange="edSetTime(${i},'end_s',this.value)"> s</div>
      <div class="edrow"><label>Size</label><button title="Smaller" aria-label="Smaller" onclick="edSize(${i},-0.1)">A−</button>
        <input type="number" min="40" max="300" step="5" style="width:64px" aria-label="Size in percent" value="${Math.round((o.size || 1) * 100)}" onchange="edSize(${i},null,this.value/100)"> %
        <button title="Bigger" aria-label="Bigger" onclick="edSize(${i},0.1)">A+</button></div>
      <div class="edrow"><label></label>${[["S", 0.7], ["M", 1], ["L", 1.4], ["XL", 1.9]].map(([l, v]) =>
        `<button class="${Math.abs((o.size || 1) - v) < 0.01 ? "primary" : ""}" onclick="edSize(${i},null,${v})">${l}</button>`).join("")}</div>
      <div class="edrow"><label></label><input type="range" min="0.4" max="3" step="0.05" aria-label="Size" value="${o.size || 1}" oninput="edSize(${i},null,+this.value,true)"></div>
      <div class="small muted">Tip: drag the corner handle of the text on the preview, or scroll over it.</div>
      <div class="edrow"><label>Colour</label><input type="color" aria-label="Colour" value="${o.color || "#ffffff"}" oninput="ed.st.overlays[${i}].color=this.value;ed.pending='Change text colour';edDrawText(edNow())">
        <button onclick="ed.st.overlays[${i}].color='';ed.pending='Reset text colour';edInspector();edDrawText(edNow())">Default</button></div>
      <div class="edrow"><label>Position</label>${o.style === "bar" ? `<span class="small muted">In the caption bar</span>` : `<span class="small muted">${o.y == null ? "Default" : Math.round(o.y * 100) + "% from top"} · drag it on the preview</span>
        <button onclick="ed.st.overlays[${i}].y=null;ed.pending='Reset text position';edInspector();edDrawText(edNow())">Reset</button>`}</div>
      <div class="row" style="margin-top:10px"><button onclick="edJump(${o.start_s})">Go to</button><button onclick="edDuplicate()">${ic("copy")} Duplicate<kbd>D</kbd></button><div class="grow"></div><button onclick="edDelete()">Delete text</button></div>`;
  } else if ((ed.sel?.kind === "pick" || ed.sel?.kind === "mom") && ed.pick) {
    const [a, b] = ed.pick, m = ed.sel.kind === "mom" ? ed.moments[ed.sel.i] : null;
    const total = s.segments.reduce((t, [x, y]) => t + y - x, 0);
    const overlap = s.segments.findIndex(([x, y]) => a < y && b > x);
    html = `<h2>${m ? `Moment #${m.rank}` : "Add to your Short"}</h2>
      ${m ? `<div><b>${esc(m.ai?.title || "Moment at " + hms(m.peak_s))}</b> ${m.tier ? `<span class="badge">${esc(m.tier)}</span>` : ""}</div>
        ${m.ai?.what_happens ? `<div class="small muted" style="margin-top:4px">${esc(m.ai.what_happens)}</div>` : ""}` : ""}
      <div class="edrow"><label>Start</label><input type="number" step="0.1" aria-label="Start" value="${a.toFixed(1)}" onchange="edSetMarks(Math.min(+this.value,ed.pick[1]-0.3),ed.pick[1]);edDraw();edInspector()"> <span class="small muted">${hms(a)}</span></div>
      <div class="edrow"><label>End</label><input type="number" step="0.1" aria-label="End" value="${b.toFixed(1)}" onchange="edSetMarks(ed.pick[0],Math.max(+this.value,ed.pick[0]+0.3));edDraw();edInspector()"> <span class="small muted">${hms(b)}</span></div>
      <div class="small muted" style="margin-top:6px">${(b - a).toFixed(1)} s · your Short would be about ${Math.round(total + b - a)} s before effects.
        ${overlap >= 0 && edSorted(s.segments) ? `<br>Overlaps part ${overlap + 1}: they'll be joined into one part.` : ""}
        ${edSorted(s.segments) ? "It goes in where it happens in the recording." : "It goes in before the first part that comes later in the recording; drag it anywhere afterwards."}</div>
      <div class="row" style="margin-top:10px"><button class="primary" onclick="edAddPick()">＋ Add to Short</button>
        <button onclick="$('#edvid').currentTime=ed.pick[0];$('#edvid').play()">▶ Play it</button><div class="grow"></div>
        <button onclick="edSetMarks(null,null);ed.sel=null;edDraw();edInspector()">Clear marks</button></div>`;
  } else if (ed.sel?.kind === "music" && ed.audio?.music) {
    const m = ed.audio.music, kind = { ai: "AI music", score: "SPHURA score", track: "Your track" }[m.kind];
    html = `<h2>Music</h2><div><b>${esc(m.name)}</b></div>
      <div class="small muted">${[kind, m.mood, m.bpm && m.bpm + " BPM"].filter(Boolean).map(esc).join(" · ")}</div>
      <div class="edrow"><label>Starts</label><span class="small">${m.in_s > 0.05 ? `at ${m.in_s.toFixed(1)} s, after the cold open` : "at the start"}</span></div>
      <div class="edrow"><label>Volume</label><span class="small">${m.gain_db != null ? `${m.gain_db} dB, ` : ""}${m.duck ? "backs off when the game gets loud" : "music-led: the game audio sits underneath"}</span></div>
      ${m.dip ? `<div class="edrow"><label>Drop</label><span class="small">Drops out at ${m.dip[0].toFixed(1)} s so the key hit lands clean</span></div>` : ""}
      ${m.credit ? `<div class="edrow"><label>Credit</label><span class="small">${esc(m.credit)}</span></div>` : ""}
      <div class="small muted" style="margin-top:8px">The ticks on the track are its beats; cuts and the key hit are lined up with them, and edges snap to them.
        ${ed.dirtyCuts ? "<br><b>Your cut changes re-sync the same song when you update the preview.</b>" : ""}</div>
      <div class="small muted" style="margin-top:8px">To change the song or its volume, use <b>Change</b> under this Short on the Shorts page.</div>`;
  } else if (ed.sel?.kind === "seg" && s.segments[ed.sel.i]) {
    const [a, b] = s.segments[ed.sel.i], i = ed.sel.i, au = s.seg_audio[i] || { gain_db: 0, mute: false }, f = edPartFactor(i);
    html = `<h2>Part ${i + 1} · ${(b - a).toFixed(1)} s</h2><div class="small muted">From the original recording: ${hms(a)} to ${hms(b)}. Its video and game audio are trimmed together.</div>
      <div class="edrow"><label>Starts</label><button onclick="edNudge(${i},0,-1)">−1s</button><button onclick="edNudge(${i},0,-0.1)">−0.1</button>
        <input type="number" step="0.1" aria-label="Starts at (recording seconds)" value="${a.toFixed(1)}" onchange="edSetSeg(${i},0,this.value)"><button onclick="edNudge(${i},0,0.1)">+0.1</button><button onclick="edNudge(${i},0,1)">+1s</button></div>
      <div class="edrow"><label>Ends</label><button onclick="edNudge(${i},1,-1)">−1s</button><button onclick="edNudge(${i},1,-0.1)">−0.1</button>
        <input type="number" step="0.1" aria-label="Ends at (recording seconds)" value="${b.toFixed(1)}" onchange="edSetSeg(${i},1,this.value)"><button onclick="edNudge(${i},1,0.1)">+0.1</button><button onclick="edNudge(${i},1,1)">+1s</button></div>
      <div class="small faint" style="margin:8px 0">Drag its edges in the timeline: outward adds footage, inward cuts. Drag the middle to move it.</div>
      <div class="sect"><h2>Sound</h2>
        <label class="small"><input type="checkbox" ${au.mute ? "checked" : ""} onchange="edSetAudio(${i},{mute:this.checked})"> Mute this part's game audio <kbd>M</kbd></label>
        <div class="edrow"><label>Volume</label><input type="range" min="-30" max="12" step="1" value="${au.gain_db}" aria-label="Volume in decibels" ${au.mute ? "disabled" : ""}
          oninput="edSetAudio(${i},{gain_db:+this.value},true)" onchange="edDraw()"><span class="small" id="edvol">${au.gain_db > 0 ? "+" : ""}${au.gain_db} dB</span>
          <button onclick="edSetAudio(${i},{gain_db:0})" ${au.gain_db ? "" : "disabled title='Already at the original volume'"}>Reset</button></div>
        <div class="small faint">${ed.live ? "Live playback goes up to the original volume; louder shows after Update preview." : "Heard after Update preview."}</div></div>
      <div class="sect"><h2>Speed</h2>${ed.dirtyCuts ? `<div class="small muted">Update the preview first: speed is re-planned for your new cut, then you can change it.</div>`
        : `<div class="row" id="edpartspeed" role="group" aria-label="Speed of this part">${ED_SPEEDS.map(v => `<button class="${f === v ? "primary" : ""}" onclick="edPartSpeed(${i},${v})">${v}×</button>`).join("")}</div>
        <div class="small faint" style="margin-top:4px">${f == null ? "Mixed: SPHURA ramps the speed inside this part. Pick one to use a single speed, or click a badge on the timeline to change one piece." : "Applies to the whole part. Speed badges on the video track show each change."}</div>`}</div>
      <div class="row" style="margin-top:10px"><button onclick="edMovePart(${i},${i - 1})" ${i ? "" : "disabled title='Already first'"} aria-label="Move earlier">← Earlier</button><button onclick="edMovePart(${i},${i + 1})" ${i < s.segments.length - 1 ? "" : "disabled title='Already last'"} aria-label="Move later">Later →</button>
        ${edSorted(s.segments) ? "" : `<button onclick="edRecordingOrder()" title="Put every part back in the order it happens in the recording">Recording order</button>`}<div class="grow"></div>
        <button ${s.segments.length < 2 ? "disabled" : ""} onclick="edDelete()">🗑 Delete part ${i + 1}</button></div>`;
  } else if (ed.sel?.kind === "spd" && s.speed[ed.sel.i] && !ed.dirtyCuts) {
    const k = ed.sel.i, p = s.speed[k], [a, b] = edSpeedOut(p);
    html = `<h2>${p.freeze ? "Freeze frame" : `Speed ${edFx2(p.factor)}×`} · ${(b - a).toFixed(2)} s ${p.owner === "user" ? `<span class="badge">yours</span>` : `<span class="badge" title="Planned by SPHURA around the key hit">✦ AI</span>`}</h2>
      <div class="small muted">${p.freeze ? "Freeze frame" : p.factor < 1 ? "Slow motion" : p.factor > 1 ? "Sped up" : "Normal speed"} from ${a.toFixed(2)} s to ${b.toFixed(2)} s of the Short (${(p.end_s - p.start_s).toFixed(2)} s of footage).</div>
      <div class="row" style="margin-top:8px" role="group" aria-label="Speed">${ED_SPEEDS.filter(v => v !== 1).map(v => `<button class="${p.factor === v ? "primary" : ""}" onclick="edSetFactor(${k},${v})">${v}×</button>`).join("")}</div>
      <div class="edrow"><label>Factor</label><input type="number" min="0.25" max="2" step="0.05" aria-label="Speed factor" value="${p.factor}" onchange="edSetFactor(${k},+this.value)"> ×</div>
      <div class="small faint" style="margin-top:6px">Drag the badge to move it. Your changes stay when SPHURA re-plans the effects after a cut. Shows after Update preview.</div>
      <div class="row" style="margin-top:10px"><button onclick="edJump(${a})">Go to</button><div class="grow"></div><button onclick="edDelete()">${ic("trash")} Delete</button></div>`;
  } else if (ed.sel?.kind === "mk" && s.markers[ed.sel.i]) {
    const k = ed.sel.i, m = s.markers[k], t = edSrcToOut(m.src_s);
    html = `<h2>Marker</h2><input type="text" style="width:100%" aria-label="Marker name" value="${esc(m.label)}" placeholder="e.g. the hit" oninput="ed.st.markers[${k}].label=this.value;ed.pending='Rename marker';edDraw()">
      <div class="small muted" style="margin-top:6px">At ${hms(m.src_s)} in the recording${t == null ? ": that moment isn't in the Short right now." : `, ${t.toFixed(2)} s into the Short.`} It stays with that moment when you change cuts, and edges snap to it.</div>
      <div class="row" style="margin-top:10px">${t == null ? "" : `<button onclick="edJump(${t})">Go to</button>`}<div class="grow"></div><button onclick="edDelete()">${ic("trash")} Delete</button></div>`;
  } else if (ed.orig) html = `<h2>Add footage</h2><div class="small muted">Click a yellow moment, or mark a start <kbd>I</kbd> and end <kbd>O</kbd>, then press <b>Add to Short</b>.</div>
      <details style="margin-top:8px" class="small"><summary>More tips</summary><ol class="edhelp">
      <li>Drag across the recording track to mark a stretch.</li>
      <li>To take something <b>out</b>, mark it and press <b>Remove from Short</b>.</li>
      <li>Go back to <b>Short</b>, then <b>Update preview</b> to watch the result.</li></ol></details>`;
  else html = `<h2>Selection</h2><div class="small muted">Click a part, a text, a speed badge, a marker or the music in the timeline to change it.</div>
      <details style="margin-top:8px" class="small"><summary>Editing tips</summary><ol class="edhelp">
      <li><b>Longer or shorter:</b> drag a part's edge outward to add footage, inward to cut.</li>
      <li><b>Reorder:</b> drag a part's middle left or right.</li>
      <li><b>Cut the middle:</b> split <kbd>S</kbd> at both ends, select the piece, delete <kbd>Del</kbd>.</li>
      <li><b>Text:</b> move the playhead and press <b>Text</b>. Drag it on the preview to move it.</li>
      <li><b>Right-click</b> anything for its actions; <kbd>?</kbd> lists every shortcut.</li>
      <li><kbd>Space</kbd> plays and pauses. Changes show after <b>Update preview</b>.</li></ol></details>
      ${s.markers.length ? `<div class="small muted" style="margin-top:8px">${s.markers.length} marker${s.markers.length > 1 ? "s" : ""}${s.markers.some(m => edSrcToOut(m.src_s) == null) ? ", some at moments that aren't in the Short now" : ""}.</div>` : ""}`;
  if (s.style !== "meme") html += `<div class="sect"><h2>Layout</h2>
    <label class="small" style="display:block"><input type="checkbox" ${s.caption_bar ? "checked" : ""} onchange="ed.st.caption_bar=this.checked;ed.dirtyBar=true;ed.pending=this.checked?'Turn on caption bar':'Turn off caption bar';edInspector()"> White caption bar on top (meme-style)</label>
    ${ed.dirtyBar ? `<div class="small" style="color:var(--accent2)">Click "Update preview" to see the new layout and its caption.</div>` : ""}</div>`;
  html += `<div class="sect"><h2>Effects</h2><div class="small" style="margin-bottom:6px">${esc(s.fx_label || "")}</div>
    ${s.toggles.map(k => `<label class="small" style="display:block"><input type="checkbox" ${s.fx_off.includes(k) ? "" : "checked"} onchange="edFx('${k}',this.checked)"> ${FX_NAMES[k] || k}</label>`).join("")}
    ${s.music ? `<div class="small muted" style="margin-top:8px">♪ ${esc(s.music)} (change it on the Shorts page)</div>` : ""}
    <div class="small faint" style="margin-top:8px">Applied when you update the preview or export.</div></div>`;
  el.innerHTML = html; edToolbar(); edFab();
}
function edJump(t) { if (ed.orig) edMode(false); edGoto(t + 0.05); setTimeout(edTick, 60); }
function edSetTime(i, k, v) { const o = ed.st.overlays[i]; o[k] = Math.max(0, +v); if (o.end_s <= o.start_s) o.end_s = o.start_s + 0.5; ed.pending = "Change text timing"; edDraw(); edDrawText(edNow()); }
function edSetSeg(i, j, v) { const seg = ed.st.segments[i]; seg[j] = Math.max(0, Math.min(ed.st.source_duration, +v)); if (seg[1] - seg[0] < 0.3) seg[j] = j ? seg[0] + 0.3 : seg[1] - 0.3; ed.dirtyCuts = true; ed.pending = `Trim part ${i + 1}`; edDraw(); edInspector(); if (ed.orig) $("#edvid").currentTime = seg[j]; }
function edNudge(i, j, d) { edSetSeg(i, j, ed.st.segments[i][j] + d); }
function edFx(k, on) { const off = new Set(ed.st.fx_off); on ? off.delete(k) : off.add(k); ed.st.fx_off = [...off]; ed.dirtyFx = true; edCommit(`${on ? "Turn on" : "Turn off"} ${FX_NAMES[k] || k}`); }
function edSetAudio(i, ch, typing) {
  const s = ed.st; while (s.seg_audio.length < s.segments.length) s.seg_audio.push({ gain_db: 0, mute: false });
  Object.assign(s.seg_audio[i], ch);
  ed.pending = "mute" in ch ? `${ch.mute ? "Mute" : "Unmute"} part ${i + 1}` : `Change part ${i + 1} volume`;
  if (typing) { const v = $("#edvol"); if (v) v.textContent = `${ch.gain_db > 0 ? "+" : ""}${ch.gain_db} dB`; edWaves(); return; }
  edDraw(); edInspector();
}
function edMute() {
  const segs = edSelAll().filter(x => x.kind === "seg"); if (!segs.length) return toast("Select a part first");
  const on = segs.some(x => !ed.st.seg_audio[x.i]?.mute);
  for (const x of segs) edSetAudio(x.i, { mute: on });
  edCommit(segs.length > 1 ? `${on ? "Mute" : "Unmute"} ${segs.length} parts` : `${on ? "Mute" : "Unmute"} part ${segs[0].i + 1}`);
}
function edPartPre(i) { const s = ed.st; let t = edLead(); for (let k = 0; k < i; k++) t += s.segments[k][1] - s.segments[k][0]; return [t, t + s.segments[i][1] - s.segments[i][0]]; }
function edPartFactor(i) {
  // The part's single speed, or null when pieces only cover some of it
  const [a, b] = edPartPre(i), on = ed.st.speed.filter(p => p.start_s < b - 1e-3 && p.end_s > a + 1e-3);
  if (!on.length) return 1;
  return on.length === 1 && on[0].start_s <= a + 0.01 && on[0].end_s >= b - 0.01 ? on[0].factor : null;
}
function edPartSpeed(i, f) {
  // One speed for the whole part: pieces inside it go, pieces across its edges are cut there
  if (ed.dirtyCuts) return toast("Update the preview first: speed is re-planned for your new cut");
  const [a, b] = edPartPre(i), out = [];
  for (const p of ed.st.speed) {
    if (p.end_s <= a + 1e-3 || p.start_s >= b - 1e-3) { out.push(p); continue; }
    if (p.start_s < a - 0.05) out.push({ ...p, end_s: +a.toFixed(3), _k: ++ed.k });
    if (p.end_s > b + 0.05) out.push({ ...p, start_s: +b.toFixed(3), _k: ++ed.k });
  }
  if (f !== 1) out.push({ type: "speed", start_s: +a.toFixed(3), end_s: +b.toFixed(3), factor: f, owner: "user", _k: ++ed.k });
  ed.st.speed = out.sort((x, y) => x.start_s - y.start_s);
  ed.sel = { kind: "seg", i }; edFlags(); edCommit(`Part ${i + 1} speed ${f}×`); edDraw(); edInspector();
}
function edFocusSpeed() { const b = $("#edpartspeed button.primary") || $("#edpartspeed button"); b?.focus(); b?.scrollIntoView({ block: "nearest" }); }
function edSetFactor(k, f) {
  const p = ed.st.speed[k]; if (!p || !(f > 0)) return;
  p.factor = Math.round(Math.min(2, Math.max(0.25, f)) * 100) / 100; delete p.freeze;
  edFlags(); edCommit(`Speed ${p.factor}×`); edDraw(); edInspector();
}
function edAddText() {
  const t = Math.min(edNow(), Math.max(0, edTotal() - 0.5));
  ed.st.overlays.push({ text: "Your text", start_s: +t.toFixed(2), end_s: +Math.min(edTotal(), t + 2.5).toFixed(2), style: "plain", y: 0.72, size: 1, color: "", owner: "user", _k: ++ed.k });
  edDraw(); edSelect("txt", ed.st.overlays.length - 1); edCommit("Add text");
  setTimeout(() => { const ta = document.querySelector("#edinspector textarea"); ta && (ta.focus(), ta.select()); }, 30);
}
function edAddMarker(at) {
  const t = at ?? (ed.orig ? null : edNow());
  const src = ed.orig ? (at ?? $("#edvid").currentTime) : edOutToSrc(t);
  if (src == null) return toast("Put the playhead on the Short first");
  ed.st.markers.push({ src_s: +src.toFixed(3), label: `Marker ${ed.st.markers.length + 1}`, _k: ++ed.k });
  edDraw(); edSelect("mk", ed.st.markers.length - 1); edCommit("Add marker");
}
function edSplit() {
  const t = edNow(), so = edSegOut(), i = so.findIndex(([a, b]) => t > a + 0.3 && t < b - 0.3);
  if (i < 0) return toast("Put the playhead inside a part (not at its edge) to split it");
  const [a, b] = ed.st.segments[i], cut = +(a + edPre(t) - edPre(so[i][0])).toFixed(3);
  ed.st.segments.splice(i, 1, [a, cut], [cut, b]); ed.st.seg_audio.splice(i, 1, edAudioOf(i), edAudioOf(i)); ed.dirtyCuts = true;
  edDraw(); edSelect("seg", i + 1); edCommit(`Split part ${i + 1}`);
}
function edDuplicate() {
  // A copy of each selected part right after it (same footage), or of each text right after it
  const items = edSelAll(), s = ed.st, segs = items.filter(x => x.kind === "seg").map(x => x.i).sort((a, b) => b - a), txts = items.filter(x => x.kind === "txt").map(x => x.i);
  if (!segs.length && !txts.length) return toast("Select a part or a text first");
  const total = s.segments.reduce((t, [x, y]) => t + y - x, 0) + segs.reduce((t, i) => t + s.segments[i][1] - s.segments[i][0], 0);
  if (total > 180) return toast(`That would make the Short ${Math.round(total)} s long; the most is 180 s.`);
  for (const i of segs) { s.segments.splice(i + 1, 0, [...s.segments[i]]); s.seg_audio.splice(i + 1, 0, edAudioOf(i)); }
  if (segs.length) ed.dirtyCuts = true;
  const end = edTotal();
  for (const i of txts) { const o = s.overlays[i], len = o.end_s - o.start_s, a = Math.min(o.end_s, Math.max(0, end - len)); s.overlays.push({ ...o, start_s: +a.toFixed(2), end_s: +(a + len).toFixed(2), owner: "user", _k: ++ed.k }); }
  ed.sel = null; ed.multi = []; edDraw(); edInspector();
  edCommit(segs.length + txts.length > 1 ? "Duplicate selection" : segs.length ? `Duplicate part ${segs[0] + 1}` : "Duplicate text");
}
function edDelete() {
  const items = edSelAll(), s = ed.st; if (!items.length) return;
  const by = k => items.filter(x => x.kind === k).map(x => x.i).sort((a, b) => b - a);
  const segs = by("seg");
  if (segs.length && segs.length >= s.segments.length) return toast("A Short needs at least one part");
  const name = items.length > 1 ? `Delete ${items.length} items` : { txt: "Delete text", seg: `Delete part ${segs[0] + 1}`, spd: "Delete speed change", mk: "Delete marker" }[items[0].kind];
  if (!name) return;
  for (const i of by("txt")) s.overlays.splice(i, 1);
  for (const i of segs) { s.segments.splice(i, 1); s.seg_audio.splice(i, 1); }
  for (const i of by("spd")) s.speed.splice(i, 1);
  for (const i of by("mk")) s.markers.splice(i, 1);
  if (segs.length) ed.dirtyCuts = true;
  ed.sel = null; ed.multi = []; edFlags(); edDraw(); edInspector(); edDrawText(edNow()); edCommit(name);
}

async function edRender(kind, quiet) {
  if (ed.busy) return;
  edCommit(); clearTimeout(ed.saveT); ed.busy = true;
  const startSnap = edSnap();
  const st = $("#edstatus"); st.innerHTML = `<span class="spin"></span> ${kind === "draft" ? "Rendering preview…" : "Saving and exporting…"}`;
  try {
    const full = localStorage.getItem("sphura_preview") === "full", body = { render: kind, full };
    if (!quiet) Object.assign(body, edBody(ed.st), { base_version: ed.st.version });
    const r = await api(`/v1/clips/${ed.st.id}/editor`, { method: "PUT", body });
    for (;;) {
      await new Promise(res => setTimeout(res, 1000));
      const j = await api(`/v1/jobs/${r.job_id}`);
      if (!$("#edstatus")) return;
      st.innerHTML = `<span class="spin"></span> ${kind === "draft" ? "Rendering preview" : "Exporting"} ${Math.round((j.progress || 0) * 100)}%`;
      if (j.status === "succeeded") break;
      if (j.status === "failed" || j.status === "cancelled") throw new Error(j.error || "render failed");
    }
    const keepSel = ed.sel, t = edNow(), sl = $("#edtl")?.scrollLeft || 0;
    edCommit(); // edits made while it rendered stay on top of the saved state
    const local = edSnap() !== startSnap ? JSON.parse(edSnap()) : null;
    ed.st = edDefaults(await api(`/v1/clips/${ed.st.id}/editor`)); ed.sel = keepSel; ed.multi = []; ed.dirty = false; ed.saveError = null;
    edKeys(); const server = edSnap();
    if (local) for (const k of ED_KEYS) ed.st[k] = local[k];
    ed.base = edSnap(); ed.saved = server; ed.rendered = ed.st.draft_fresh || kind === "final" ? server : null; edFlags();
    if (!edSelValid(ed.sel)) ed.sel = null;
    ed.audioKey = null; ed.busy = false; // cuts, speed and music may have changed
    if (kind === "final") toast("Saved and exported");
    renderEditor(); $("#edtl").scrollLeft = sl;
    $("#edstatus").textContent = kind === "draft" ? (full ? "Preview updated (full quality)" : "Preview updated (fast: half size, 30 fps)") : "Saved and exported to your Shorts folder";
    const v = $("#edvid"); v.onloadeddata = () => { v.currentTime = Math.min(t, ed.st.duration); edTick(); };
    if (local) { edQueueSave(); edNote(); }
  } catch (e) {
    if (/another tab/.test(e.message)) { ed.conflict = true; edSaveUI(`<span class="err">Changed in another tab or window.</span> <button class="small" onclick="openEditor('${ed.st.id}')">Reload</button>`); }
    st.innerHTML = `<span class="err">${esc(e.message)}</span>`;
  } finally { ed.busy = false; }
}

// ---------- Keyboard ----------
document.addEventListener("keydown", e => {
  if (!ed || !$("#edtl") || ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
  if (e.target.closest?.("#edmenu, #edhelpdlg")) return;
  const mod = e.ctrlKey || e.metaKey, k = e.key;
  if (mod && (k === "z" || k === "Z")) { e.preventDefault(); e.shiftKey ? edRedo() : edUndo(); return; }
  if (mod && (k === "y" || k === "Y")) { e.preventDefault(); edRedo(); return; }
  if ((e.shiftKey && k === "F10") || k === "ContextMenu") { e.preventDefault(); edMenuKey(); return; }
  if (mod || e.altKey) return;
  const act = {
    " ": edPlay, Delete: edDelete, Backspace: edDelete, s: edSplit, j: () => edJKL("j"), k: () => edJKL("k"), l: () => edJKL("l"),
    ArrowLeft: () => edStep(e.shiftKey ? -1 : -ED_FRAME), ArrowRight: () => edStep(e.shiftKey ? 1 : ED_FRAME),
    Home: () => { edRev(false); ed.orig ? $("#edvid").currentTime = 0 : edGoto(0); setTimeout(edTick, 60); },
    End: () => { edRev(false); ed.orig ? $("#edvid").currentTime = ed.st.source_duration - 0.1 : edGoto(edTotal() - 0.05); setTimeout(edTick, 60); },
    "+": () => edZoomAt((ed.orig ? ed.spps : ed.pps) * 1.25), "=": () => edZoomAt((ed.orig ? ed.spps : ed.pps) * 1.25), "-": () => edZoomAt((ed.orig ? ed.spps : ed.pps) / 1.25),
    z: edFitZoom, d: edDuplicate, m: edMute, b: () => edAddMarker(), "?": edHelp,
    Escape: () => { edSelect(null); if (ed.orig) { edSetMarks(null, null); edDraw(); edInspector(); } },
    i: () => edMark(0), o: () => edMark(1),
  }[k.length === 1 ? k.toLowerCase() : k] || (e.code === "Space" ? edPlay : null);
  if (!act || (ed.orig && "sdm".includes(k.toLowerCase()) && k.length === 1)) return;
  e.preventDefault(); act();
});
