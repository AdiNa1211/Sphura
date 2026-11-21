from __future__ import annotations

import asyncio
import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated, Any

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect

from sphura import __version__
from sphura.api.schemas import (
    AnalyzeRequest,
    IngestRequest,
    IngestResponse,
    JobOut,
    MediaAssetOut,
    ProjectOut,
)
from sphura.api.ui import LOCAL_HOSTS, build_router
from sphura.config import Settings
from sphura.container import LocalEngine, build_local_engine
from sphura.domain.clip_codec import clip_to_dict
from sphura.domain.errors import (
    InvalidMediaError,
    NotFoundError,
    SphuraError,
    ToolchainError,
)
from sphura.domain.events import JobProgressEvent
from sphura.domain.models import AspectRatio, EditingMode
from sphura.pipeline.judge import judgement_to_dict
from sphura.pipeline.signals import top_peaks


def engine_of(request: Request) -> LocalEngine:
    engine: LocalEngine = request.app.state.engine
    return engine


Engine = Annotated[LocalEngine, Depends(engine_of)]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = build_local_engine(settings)
        engine.start()
        app.state.engine = engine
        try:
            yield
        finally:
            engine.stop()

    app = FastAPI(title="SPHURA Engine", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def local_only(request: Request, call_next: Any) -> Any:
        # Without a token the API is open to this machine only: refuse other Host names so
        # a web page can't reach it through DNS rebinding
        host = (request.headers.get("host") or "").rsplit(":", 1)[0]
        if settings.api_token is None and host not in LOCAL_HOSTS:
            return JSONResponse({"detail": "local access only"}, status_code=403)
        return await call_next(request)

    def _token_ok(token: str | None) -> bool:
        if settings.api_token is None:
            return True
        return token is not None and hmac.compare_digest(token, settings.api_token)

    def auth(x_sphura_token: Annotated[str | None, Header()] = None) -> None:
        if not _token_ok(x_sphura_token):
            raise HTTPException(status_code=401, detail="invalid token")

    @app.exception_handler(SphuraError)
    async def _domain_error(_: Request, exc: SphuraError) -> JSONResponse:
        codes = {NotFoundError: 404, InvalidMediaError: 422, ToolchainError: 503}
        code = next((c for t, c in codes.items() if isinstance(exc, t)), 400)
        return JSONResponse(
            status_code=code, content={"error": type(exc).__name__, "detail": str(exc)}
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    v1 = [Depends(auth)]
    app.include_router(build_router(v1))

    @app.post("/v1/ingest", response_model=IngestResponse, dependencies=v1)
    def ingest(body: IngestRequest, engine: Engine) -> IngestResponse:
        r = engine.ingest.ingest_local_file(
            body.path, body.project_id, body.project_name, body.mode
        )
        return IngestResponse(
            project=ProjectOut.of(r.project),
            asset=MediaAssetOut.of(r.asset),
            checksum_job=JobOut.of(r.checksum_job) if r.checksum_job else None,
            already_registered=r.already_registered,
        )

    @app.get("/v1/projects", response_model=list[ProjectOut], dependencies=v1)
    def projects(engine: Engine, limit: int = 100) -> list[ProjectOut]:
        return [ProjectOut.of(p) for p in engine.queries.list_projects(limit)]

    @app.get("/v1/projects/{project_id}", response_model=ProjectOut, dependencies=v1)
    def project(project_id: str, engine: Engine) -> ProjectOut:
        return ProjectOut.of(engine.queries.project(project_id))

    @app.get(
        "/v1/projects/{project_id}/assets", response_model=list[MediaAssetOut], dependencies=v1
    )
    def assets(project_id: str, engine: Engine) -> list[MediaAssetOut]:
        return [MediaAssetOut.of(a) for a in engine.queries.assets(project_id)]

    @app.get("/v1/projects/{project_id}/jobs", response_model=list[JobOut], dependencies=v1)
    def project_jobs(project_id: str, engine: Engine) -> list[JobOut]:
        return [JobOut.of(j) for j in engine.queries.jobs(project_id)]

    @app.post("/v1/projects/{project_id}/analyze", response_model=JobOut, dependencies=v1)
    def analyze(project_id: str, body: AnalyzeRequest, engine: Engine) -> JobOut:
        return JobOut.of(engine.analysis.start(project_id, body.language))

    @app.get("/v1/projects/{project_id}/transcript", dependencies=v1)
    def transcript(project_id: str, engine: Engine) -> dict[str, Any]:
        return engine.queries.transcript(project_id).to_dict()

    @app.get("/v1/projects/{project_id}/highlights", dependencies=v1)
    def highlights(
        project_id: str, engine: Engine, mode: EditingMode | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        out = []
        for r in engine.queries.highlights(project_id, mode, limit):
            c, j = r.candidate, r.candidate.judgement
            span = c.clip_range
            out.append(
                {
                    "rank": r.rank,
                    "id": c.id,
                    "start_s": span.start_s,
                    "end_s": span.end_s,
                    "duration_s": round(span.duration_s, 1),
                    "peak_s": c.peak_s,
                    "tier": r.tier.value if r.tier else None,
                    "signal_drivers": list(c.drivers),
                    "ai": judgement_to_dict(j) if j else None,
                }
            )
        return out

    @app.post("/v1/projects/{project_id}/clips", dependencies=v1)
    def generate_clips(
        project_id: str, engine: Engine, top: int = 5, mode: EditingMode | None = None
    ) -> list[dict[str, Any]]:
        return [clip_to_dict(c) for c in engine.clips.generate(project_id, top, mode)]

    @app.get("/v1/projects/{project_id}/clips", dependencies=v1)
    def list_clips(project_id: str, engine: Engine) -> list[dict[str, Any]]:
        return [
            {**clip_to_dict(c), "output_path": path} for c, path in engine.queries.clips(project_id)
        ]

    @app.post(
        "/v1/projects/{project_id}/clips/render", response_model=list[JobOut], dependencies=v1
    )
    def render_clips(
        project_id: str,
        engine: Engine,
        top: int = 5,
        aspect: str = "9:16",
        layout: str | None = None,
    ) -> list[JobOut]:
        clips = engine.clips.generate(project_id, top, aspect=AspectRatio(aspect), layout=layout)
        return [JobOut.of(j) for j in engine.clips.render(clips)]

    @app.get("/v1/projects/{project_id}/signals", dependencies=v1)
    def signals(
        project_id: str, engine: Engine, points: int = 2000, top: int = 20
    ) -> dict[str, Any]:
        tl = engine.queries.signals(project_id)
        step = max(1, tl.length // max(points, 1))
        keep = ("interest", "z_loud_rel", "z_motion", "z_flux", "speech_wps", "scene_cut")
        return {
            "hop_s": tl.hop_s * step,
            "duration_s": tl.duration_s,
            "curves": {
                k: [round(float(v), 3) for v in np.nan_to_num(tl.features[k][::step])]
                for k in keep
                if k in tl.features
            },
            "peaks": [
                {"t_s": p.t_s, "score": round(p.score, 3), "drivers": list(p.drivers)}
                for p in top_peaks(tl, top)
            ],
        }

    @app.get("/v1/jobs/{job_id}", response_model=JobOut, dependencies=v1)
    def job(job_id: str, engine: Engine) -> JobOut:
        return JobOut.of(engine.queries.job(job_id))

    @app.post("/v1/jobs/{job_id}/cancel", response_model=JobOut, dependencies=v1)
    def cancel(job_id: str, engine: Engine) -> JobOut:
        engine.queries.job(job_id)
        engine.queue.cancel(job_id)
        return JobOut.of(engine.queries.job(job_id))

    @app.get("/v1/system/hardware", dependencies=v1)
    def hardware(engine: Engine, refresh: bool = False) -> dict[str, Any]:
        p = engine.toolchain.profile(refresh=refresh)
        return {
            **asdict(p),
            "best_h264_encoder": p.best_h264_encoder,
            "has_hw_encoder": p.has_hw_encoder,
        }

    @app.websocket("/v1/events")
    async def events(
        ws: WebSocket, token: Annotated[str | None, Query()] = None, project_id: str | None = None
    ) -> None:
        if not _token_ok(token):
            await ws.close(code=4401)
            return
        await ws.accept()
        engine: LocalEngine = ws.app.state.engine
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[JobProgressEvent] = asyncio.Queue(maxsize=1000)

        def on_event(e: JobProgressEvent) -> None:
            # Called from worker threads; hop onto the event loop
            if project_id is None or e.project_id == project_id:
                loop.call_soon_threadsafe(_offer, e)

        def _offer(e: JobProgressEvent) -> None:
            if queue.full():
                queue.get_nowait()  # drop oldest; clients can re-sync via REST
            queue.put_nowait(e)

        unsubscribe = engine.events.subscribe(on_event)
        try:
            while True:
                await ws.send_json((await queue.get()).to_dict())
        except WebSocketDisconnect:
            pass
        finally:
            unsubscribe()

    return app
