from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
import warnings
from dataclasses import replace
from pathlib import Path

from sphura.config import Settings
from sphura.container import LocalEngine, build_local_engine
from sphura.domain.errors import SphuraError
from sphura.domain.events import JobProgressEvent
from sphura.domain.models import AspectRatio, EditingMode, Job, MediaInfo
from sphura.infrastructure.ai.accel import cuda_device_count
from sphura.infrastructure.ai.ollama import OllamaProvider
from sphura.pipeline.preferences import describe
from sphura.pipeline.report import render_html
from sphura.pipeline.signals import top_peaks


def _hms(seconds: float) -> str:
    s = round(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _print_media(info: MediaInfo) -> None:
    v = info.primary_video
    flags = [f for f, on in (("VFR", v.variable_frame_rate), ("HDR", v.hdr)) if on]
    if v.rotation:
        flags.append(f"rotated {v.rotation}°")
    print(f"duration {_hms(info.duration_s)}  ({info.format_name})")
    print(
        f"video    {v.codec} {v.display_width}x{v.display_height} @ {v.fps:g} fps"
        f"  {v.bit_depth}-bit {v.pix_fmt or ''}  {' '.join(flags)}".rstrip()
    )
    if not info.audio:
        print("audio    none")
    for a in info.audio:
        label = f'"{a.title}"' if a.title else ""
        print(
            f"audio    #{a.index} {a.codec} {a.channels}ch {a.sample_rate // 1000}kHz {label}"
            f"{'  default' if a.is_default and len(info.audio) > 1 else ''}".rstrip()
        )


def _cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    from sphura.api.app import create_app

    settings = replace(settings, host=args.host or settings.host, port=args.port or settings.port)
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info")
    return 0


def _cmd_ui(args: argparse.Namespace, settings: Settings) -> int:
    import threading
    import webbrowser

    import uvicorn

    from sphura.api.app import create_app

    port = args.port or settings.port
    host, token = "127.0.0.1", None
    if args.lan:  # other devices on the network, with a key; this PC as usual
        host, token = "0.0.0.0", _lan_key(settings.data_dir)
    settings = replace(settings, host=host, port=port, api_token=token)
    url = f"http://127.0.0.1:{port}/"
    if not args.no_browser:
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    print(f"SPHURA is running at {url}  (keep this window open; Ctrl+C to quit)")
    if token:
        print("On other devices on your network (phone, laptop), open:")
        for ip in _lan_ips():
            print(f"  http://{ip}:{port}/?key={token}")
        print("Anyone with that link can use SPHURA. Delete lan_key.txt in the data folder to"
              " make a new key.")  # fmt: skip
    uvicorn.run(create_app(settings), host=host, port=port, log_level="warning")
    return 0


def _lan_key(data_dir: Path) -> str:
    import secrets

    path = data_dir / "lan_key.txt"
    if path.is_file() and path.read_text(encoding="utf-8").strip():
        return path.read_text(encoding="utf-8").strip()
    data_dir.mkdir(parents=True, exist_ok=True)
    key = secrets.token_urlsafe(16)
    path.write_text(key, encoding="utf-8")
    return key


def _lan_ips() -> list[str]:
    import contextlib
    import socket

    ips: set[str] = set()
    with contextlib.suppress(OSError):
        ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
    # The address used to reach the network, even when the hostname lookup is thin
    with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
    return sorted(ip for ip in ips if not ip.startswith("127.")) or ["<this PC's IP>"]


def _cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    engine = build_local_engine(settings)
    engine.start()
    try:
        r = engine.ingest.ingest_local_file(
            args.path, project_name=args.name, mode=EditingMode(args.mode), reprobe=args.reprobe
        )
        a = r.asset
        print(f"project  {r.project.id}  {r.project.name}")
        print(f"asset    {a.id}  {a.container.value}  {a.size_bytes / 1e9:.2f} GB  {a.fingerprint}")
        if a.media_info:
            _print_media(a.media_info)
        if r.already_registered:
            print("already registered")
            return 0
        if not args.wait or r.checksum_job is None:
            return 0

        job_id = r.checksum_job.id
        done = threading.Event()

        def on_event(e: JobProgressEvent) -> None:
            if e.job_id != job_id:
                return
            print(f"\r{e.stage or e.status.value:10s} {e.progress * 100:5.1f}%", end="", flush=True)
            if e.status.is_terminal:
                done.set()

        engine.events.subscribe(on_event)
        done.wait()
        job = engine.queries.job(job_id)
        print(f"\n{job.status.value}  {job.result or job.error}")
        return 0 if job.status.value == "succeeded" else 1
    finally:
        engine.stop()


def _wait_for_job(engine: LocalEngine, job_id: str) -> Job:
    done = threading.Event()
    last = {"stage": ""}

    def on_event(e: JobProgressEvent) -> None:
        if e.job_id != job_id:
            return
        stage = e.stage or e.status.value
        if stage != last["stage"] and last["stage"]:
            print()
        last["stage"] = stage
        msg = f"  {e.message}" if e.message and not e.status.is_terminal else ""
        print(f"\r{stage:14s} {e.progress * 100:5.1f}%{msg:40s}", end="", flush=True)
        if e.status.is_terminal:
            done.set()

    unsubscribe = engine.events.subscribe(on_event)
    try:
        while not done.wait(0.5):
            if engine.queries.job(job_id).status.is_terminal:  # finished before we subscribed
                break
    finally:
        unsubscribe()
    print()
    return engine.queries.job(job_id)


def _cmd_analyze(args: argparse.Namespace, settings: Settings) -> int:
    if args.model:
        settings = replace(settings, whisper_model=args.model)
    if args.device:
        settings = replace(settings, whisper_device=args.device)
    if args.ai:
        settings = replace(settings, ai_provider=args.ai)
    engine = build_local_engine(settings)
    _announce_judge(engine, settings)
    if isinstance(engine.llm, OllamaProvider) and (problem := engine.llm.problem()):
        # Check before the long stages, not after them
        print(f"error: {problem}", file=sys.stderr)
        return 2
    engine.start()
    try:
        target = args.target
        if Path(target).expanduser().exists():
            r = engine.ingest.ingest_local_file(target)
            project_id = r.project.id
            print(f"project  {project_id}  {r.project.name}")
        else:
            project_id = target
        job = engine.analysis.start(project_id, args.language)
        print(f"job      {job.id}  (Ctrl+C to stop; re-run to resume)")
        t0 = time.monotonic()
        try:
            job = _wait_for_job(engine, job.id)
        except KeyboardInterrupt:
            engine.queue.cancel(job.id)
            print("\nstopped; finished chunks are saved")
            return 130
        took = time.monotonic() - t0
        if job.status.value != "succeeded":
            print(f"{job.status.value}: {job.error or ''}")
            return 1
        sig = (job.result or {}).get("signals")
        if sig:
            peaks = ", ".join(_hms(t) for t in sig["top_peaks"])
            print(f"signals     {sig['scene_cuts']} scene cuts; strongest moments at {peaks}")
        hl = (job.result or {}).get("highlights")
        if hl:
            line = f"highlights  {hl['candidates']} candidate moments"
            if hl["judge"]:
                line += f", AI-reviewed by {hl['judge']} ({hl['judged']} new, {hl['cached']} cached"
                line += f", {hl['failed']} failed)" if hl["failed"] else ")"
                if hl["input_tokens"]:
                    line += f"  tokens in/out {hl['input_tokens']}/{hl['output_tokens']}"
            if hl.get("comparisons"):
                line += f", {hl['comparisons']} side-by-side comparisons"
            print(line)
            print(f"            see them: sphura highlights {job.project_id}")
        tr = (job.result or {}).get("transcript")
        if tr:
            print(
                f"transcript  {tr['language']}  {tr['segments']} segments, {tr['words']} words"
                f"  (model {tr['model']}, audio stream #{tr['audio_stream_index']})"
            )
        else:
            print("no audio: transcript skipped")
        print(f"took {_hms(took)}")
        return 0
    finally:
        engine.stop()


def _srt_time(t: float) -> str:
    ms = round(t * 1000)
    h, m, s = ms // 3_600_000, ms % 3_600_000 // 60_000, ms % 60_000 // 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms % 1000:03d}"


def _cmd_transcript(args: argparse.Namespace, settings: Settings) -> int:
    engine = build_local_engine(settings)
    t = engine.queries.transcript(args.project_id)
    if args.srt:
        lines = []
        for i, s in enumerate(t.segments, 1):
            lines += [str(i), f"{_srt_time(s.start_s)} --> {_srt_time(s.end_s)}", s.text, ""]
        Path(args.srt).expanduser().write_text("\n".join(lines), encoding="utf-8")
        print(f"wrote {len(t.segments)} subtitles to {args.srt}")
        return 0
    print(f"# {t.language}  model {t.model}  {len(t.segments)} segments  {t.word_count} words")
    for s in t.segments[: args.limit] if args.limit else t.segments:
        print(f"[{_hms(s.start_s)}] {s.text}")
    return 0


def _announce_judge(engine: LocalEngine, settings: Settings) -> None:
    llm = engine.llm
    if llm is None:
        hint = (
            "set ANTHROPIC_API_KEY or run Ollama"
            if settings.ai_provider == "auto"
            else f"'{settings.ai_provider}' is not available"
        )
        print(f"AI judge    none ({hint}); moments will be ranked by signals only")
    elif llm.info.runs_locally:
        print(f"AI judge    {llm.info.name} (local: nothing leaves this computer)")
    else:
        print(
            f"AI judge    {llm.info.name} (cloud: sends 8 low-res frames + transcript lines "
            "per candidate moment; the video itself is never uploaded)"
        )


_CATEGORY_ICON = {
    "action": "⚔️",
    "clutch": "🔥",
    "funny": "😂",
    "cinematic": "🎬",
    "unexpected": "😱",
}
_TIER_LABEL = {"top": "TOP   ", "strong": "STRONG", "maybe": "maybe "}


def _cmd_highlights(args: argparse.Namespace, settings: Settings) -> int:
    engine = build_local_engine(settings)
    project = engine.queries.project(args.project_id)
    mode = EditingMode(args.mode) if args.mode else None
    ranked = engine.queries.highlights(args.project_id, mode, args.limit)
    reviewed = sum(1 for r in ranked if r.candidate.judgement)
    print(f"# {project.name}  mode: {(mode or project.mode).value}")
    if not reviewed:
        print("# not AI-reviewed: ordered by signals only (titles and tiers need an AI judge)")
    for r in ranked:
        c, j = r.candidate, r.candidate.judgement
        span = c.clip_range
        dur = f"{span.duration_s:.0f}s"
        at = f"{_hms(span.start_s)}-{_hms(span.end_s)}"
        if j is None:
            print(f"{r.rank:2d}. {at} {dur:>4}  {', '.join(c.drivers) or '-'}")
            continue
        icon = _CATEGORY_ICON.get(j.category.value if j.category else "", "·")
        tier = _TIER_LABEL[r.tier.value] if r.tier else "      "
        title = j.title or "(untitled)"
        tags = f"  [{' · '.join(j.tags)}]" if j.tags else ""
        print(f"{r.rank:2d}. {icon} {tier} {title} — {dur}  [{at}]{tags}")
        if args.details:
            if c.compare_score is not None:
                print(
                    f"      side-by-side: placed above {c.compare_score:.0%} "
                    "of the moments it was compared with"
                )
            print(f"      what: {j.what_happens}")
            print(f"      why:  {j.reason}")
            if j.hook:
                print(f'      hook: "{j.hook}"')
    return 0


def _cmd_render(args: argparse.Namespace, settings: Settings) -> int:
    engine = build_local_engine(settings)
    engine.start()
    try:
        mode = EditingMode(args.mode) if args.mode else None
        clips = engine.clips.generate(
            args.project_id,
            args.top,
            mode,
            args.include_maybe,
            AspectRatio(args.aspect),
            args.layout,
            not args.plain,
            args.style,
            args.game,
        )
        out_dir = Path(args.out).expanduser().resolve() if args.out else None
        if not clips:
            print("no moments rated TOP/STRONG; use --include-maybe to render the best anyway")
            return 0
        if len(clips) < args.top and not args.include_maybe:
            print(
                f"only {len(clips)} moment(s) are worth posting (TOP/STRONG); "
                "use --include-maybe to fill up to --top"
            )
        print(f"rendering {len(clips)} clip(s){f' -> {out_dir}' if out_dir else ''}")
        jobs = engine.clips.render(clips, out_dir)
        names = {j.id: f"#{c.rank} {c.title[:40]}" for j, c in zip(jobs, clips, strict=True)}
        failed = 0
        try:
            for job in jobs:
                print(f"{names[job.id]}  ({engine.queries.job(job.id).payload['clip_id'][:8]})")
                done = _wait_for_job(engine, job.id)
                if done.status.value == "succeeded" and done.result:
                    r = done.result
                    where = r["exported"] or r["path"]
                    shape = f"{r['aspect']} {r['layout']}" if r["aspect"] != "16:9" else "16:9"
                    print(f"   ok  {r['duration_s']:.0f}s {shape} via {r['encoder']}  {where}")
                else:
                    failed += 1
                    print(f"   {done.status.value}: {done.error}")
        except KeyboardInterrupt:
            for job in jobs:
                engine.queue.cancel(job.id)
            print("\nstopped; finished clips are kept")
            return 130
        return 1 if failed else 0
    finally:
        engine.stop()


def _cmd_clips(args: argparse.Namespace, settings: Settings) -> int:
    engine = build_local_engine(settings)
    clips = engine.queries.clips(args.project_id)
    if not clips:
        print("no clips yet; run: sphura render <project id>")
    for c, path in clips:
        cuts = f", {len(c.segments) - 1} jump cut(s)" if len(c.segments) > 1 else ""
        start = c.segments[0].start_s
        print(f"#{c.rank or '-':<2} {c.title[:50]:50s} {c.duration_s:5.1f}s at {_hms(start)}{cuts}")
        print(f"    {path or 'not rendered yet'}")
    return 0


def _cmd_feedback(args: argparse.Namespace, settings: Settings) -> int:
    engine = build_local_engine(settings)
    if not args.good and not args.bad:
        print("nothing to rate: use --good 1 2 and/or --bad 3 (clip numbers from file names)")
        return 2
    for line in engine.feedback.rate_clips(args.project_id, args.good, args.bad):
        print(line)
    _print_taste(engine)
    print("rankings now follow your taste; re-run `sphura render` to use it")
    return 0


def _cmd_taste(args: argparse.Namespace, settings: Settings) -> int:
    engine = build_local_engine(settings)
    for words, likes, verb in (
        (args.like, True, "like"),
        (args.dislike, False, "dislike"),
        (args.forget, None, "forget"),
    ):
        if words:
            done = engine.feedback.set_taste(words, likes)
            print(f"{verb}: {', '.join(done)}")
    _print_taste(engine)
    return 0


def _print_taste(engine: LocalEngine) -> None:
    model = engine.queries.taste_summary()
    print(f"taste: learned from {model.ratings} rating(s) ({model.good} good, {model.bad} bad)")
    if not model.weights:
        print("       rate at least one good and one bad clip (or use --like/--dislike)")
        return
    likes, dislikes = describe(model)
    print(f"       leaning toward:   {', '.join(likes) or '(nothing clear yet)'}")
    print(f"       leaning away:     {', '.join(dislikes) or '(nothing clear yet)'}")
    print("       wrong? correct it: sphura taste --like action  (or --dislike / --forget)")


def _cmd_signals(args: argparse.Namespace, settings: Settings) -> int:
    engine = build_local_engine(settings)
    project = engine.queries.project(args.project_id)
    tl = engine.queries.signals(args.project_id)
    peaks = top_peaks(tl, args.top)
    cuts = int(tl.features["scene_cut"].sum())
    print(f"# {project.name}  {_hms(tl.duration_s)}  {cuts} scene cuts  (preliminary ranking)")
    for i, p in enumerate(peaks, 1):
        print(f"{i:2d}. {_hms(p.t_s)}  score {p.score:5.2f}  {', '.join(p.drivers) or '-'}")
    if args.html:
        out = Path(args.html).expanduser()
        out.write_text(render_html(project.name, tl, peaks), encoding="utf-8")
        print(f"chart: {out.resolve()}")
    return 0


def _cmd_info(args: argparse.Namespace, settings: Settings) -> int:
    engine = build_local_engine(settings)
    _print_media(engine.probe.probe(str(Path(args.path).expanduser().resolve())))
    return 0


def _cmd_doctor(args: argparse.Namespace, settings: Settings) -> int:
    engine = build_local_engine(settings)
    p = engine.toolchain.profile(refresh=args.refresh)
    print(f"ffmpeg   {p.ffmpeg_version}  {p.ffmpeg_path}")
    print(f"ffprobe  {p.ffprobe_path}")
    print(f"license  {'GPL build' if p.gpl_build else 'LGPL build'}")
    print(f"decode   hwaccels: {', '.join(p.hwaccels) or 'none'}")
    print(f"h264     {', '.join(p.h264_encoders) or 'NONE WORKING'}")
    print(f"hevc     {', '.join(p.hevc_encoders) or 'none'}")
    best = p.best_h264_encoder
    gpu = "GPU" if p.has_hw_encoder else "CPU only"
    print(f"render   will use {best or '-'} ({gpu})")

    cuda = cuda_device_count()
    if cuda or settings.whisper_device not in ("auto", "cpu"):
        device = settings.whisper_device if settings.whisper_device != "auto" else "cuda"
        testing = f"speech   testing {device} (first time downloads a 75 MB test model)..."
        print(testing, end="\r", flush=True)
        check = engine.accel.check(device, refresh=args.refresh)
        status = "OK" if check.ok else f"FAILED - {check.detail}"
        print(f"speech   {device}: {status}".ljust(len(testing)))
    model, device, compute = engine.whisper.resolve()
    print(f"speech   will use Whisper {model} on {device} ({compute})")
    return 0 if best else 1


def _cmd_projects(_: argparse.Namespace, settings: Settings) -> int:
    engine = build_local_engine(settings)
    for p in engine.queries.list_projects():
        print(f"{p.id}  {p.status.value:9s}  {p.mode.value:9s}  {p.name}")
        for j in engine.queries.jobs(p.id):
            print(f"    job {j.id}  {j.type.value:8s} {j.status.value:9s} {j.progress * 100:5.1f}%")
    return 0


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Per-request HTTP logs from model downloads drown out the progress line
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING if verbose else logging.ERROR)
    warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sphura")
    parser.add_argument("-v", "--verbose", action="store_true", help="show detailed logs")
    parser.add_argument("--data-dir", type=Path, help="override SPHURA_DATA_DIR")
    sub = parser.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("ui", help="open SPHURA in your browser")
    u.add_argument("--port", type=int)
    u.add_argument("--no-browser", action="store_true")
    u.add_argument("--lan", action="store_true", help="also allow other devices on your network")

    s = sub.add_parser("serve", help="run the local engine API")
    s.add_argument("--host")
    s.add_argument("--port", type=int)

    i = sub.add_parser("ingest", help="register a local video")
    i.add_argument("path")
    i.add_argument("--name")
    i.add_argument("--mode", default=EditingMode.AUTO.value, choices=[m.value for m in EditingMode])
    i.add_argument("--wait", action="store_true", help="wait for the background checksum")
    i.add_argument(
        "--reprobe", action="store_true", help="re-read details of an already registered file"
    )

    sub.add_parser("projects", help="list projects and jobs")

    an = sub.add_parser("analyze", help="extract audio and transcribe (resumable)")
    an.add_argument("target", help="video path or project id")
    an.add_argument("--language", help="e.g. en, hi; default auto-detect")
    an.add_argument("--model", help="whisper model: tiny, base, small, medium, turbo ...")
    an.add_argument(
        "--ai",
        choices=["auto", "anthropic", "ollama", "none"],
        help="AI judge for ranking moments (default: auto)",
    )
    an.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], help="mps = Apple GPU (experimental)"
    )

    tr = sub.add_parser("transcript", help="print a project's transcript")
    tr.add_argument("project_id")
    tr.add_argument("--limit", type=int, default=0, help="only the first N lines")
    tr.add_argument("--srt", help="write subtitles to this .srt file instead")

    hl = sub.add_parser("highlights", help="list the best moments, AI-ranked")
    hl.add_argument("project_id")
    hl.add_argument("--mode", choices=[m.value for m in EditingMode])
    hl.add_argument("--limit", type=int, default=15)
    hl.add_argument("--details", action="store_true", help="show what happens, why, and hook")

    rd = sub.add_parser("render", help="cut the top moments into MP4 clips")
    rd.add_argument("project_id")
    rd.add_argument("--top", type=int, default=5, help="at most this many clips (default 5)")
    rd.add_argument("--aspect", choices=["9:16", "16:9", "1:1"], default="9:16")
    rd.add_argument(
        "--layout",
        choices=["auto", "crop", "fit"],
        help="crop = follow the action; fit = whole frame on a blurred background",
    )
    rd.add_argument(
        "--include-maybe", action="store_true", help="also render moments rated 'maybe'"
    )
    rd.add_argument(
        "--style",
        choices=["hype", "meme", "cinematic"],
        default="hype",
        help="hype: punchy 9:16 edit; meme: ~10 s with a caption bar (3:4); "
        "cinematic: ~20 s square, mono grade, glitch cuts, music-led",
    )
    rd.add_argument("--game", help='game name for titles and hashtags, e.g. "Phantom Blade Zero"')
    rd.add_argument(
        "--plain",
        action="store_true",
        help="no cold open, punch-ins or slow motion (hook text stays)",
    )
    rd.add_argument("--mode", choices=[m.value for m in EditingMode])
    rd.add_argument("--out", help="also copy finished clips to this folder")

    cl = sub.add_parser("clips", help="list a project's clips and where they are")
    cl.add_argument("project_id")

    fb = sub.add_parser("feedback", help="rate rendered clips so rankings learn your taste")
    fb.add_argument("project_id")
    fb.add_argument("--good", type=int, nargs="*", default=[], help="clip numbers you liked")
    fb.add_argument("--bad", type=int, nargs="*", default=[], help="clip numbers you didn't")

    ts = sub.add_parser("taste", help="show or correct what SPHURA has learned you like")
    ts.add_argument("--like", nargs="*", default=[], help='e.g. action cutscene "boss"')
    ts.add_argument("--dislike", nargs="*", default=[], help="e.g. routine")
    ts.add_argument("--forget", nargs="*", default=[], help="drop a stated like/dislike")

    sg = sub.add_parser("signals", help="show the moments with the strongest signals")
    sg.add_argument("project_id")
    sg.add_argument("--top", type=int, default=15)
    sg.add_argument("--html", help="also write an interactive chart to this .html file")

    inf = sub.add_parser("info", help="show video details without registering it")
    inf.add_argument("path")

    d = sub.add_parser("doctor", help="check FFmpeg and hardware encoders")
    d.add_argument(
        "--refresh", action="store_true", help="re-test encoders (e.g. after a driver update)"
    )

    args = parser.parse_args(argv)
    _setup_logging(args.verbose or args.cmd == "serve")
    settings = Settings.from_env()
    if args.data_dir:
        settings = replace(settings, data_dir=args.data_dir)

    handlers = {
        "serve": _cmd_serve,
        "ui": _cmd_ui,
        "ingest": _cmd_ingest,
        "projects": _cmd_projects,
        "info": _cmd_info,
        "analyze": _cmd_analyze,
        "transcript": _cmd_transcript,
        "signals": _cmd_signals,
        "highlights": _cmd_highlights,
        "render": _cmd_render,
        "clips": _cmd_clips,
        "feedback": _cmd_feedback,
        "taste": _cmd_taste,
        "doctor": _cmd_doctor,
    }
    try:
        return handlers[args.cmd](args, settings)
    except SphuraError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
