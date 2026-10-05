"""HTTP routes. Thin layer: validate input, delegate to BlurService, map errors to status codes."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from blurring.config import Config
from blurring.serving.schemas import BlurParams, BlurResponse, ModelInfo
from blurring.serving.service import BlurService, InvalidImageError


def create_app(cfg: Config, service: BlurService | None = None) -> FastAPI:
    """Build the API. ``service`` can be injected (tests); otherwise the checkpoint is loaded at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or BlurService.from_config(cfg)
        yield

    app = FastAPI(title="Street View Blurring API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=cfg.serving.cors_origins, allow_methods=["GET", "POST"], allow_headers=["*"]
    )
    max_bytes = cfg.serving.max_upload_mb * 1024 * 1024

    def _service(request: Request) -> BlurService:
        return request.app.state.service

    @app.get("/api/health")
    def health(request: Request) -> dict:
        svc = _service(request)
        return {"status": "ok", "backend": svc.predictor.backend}

    @app.get("/api/model", response_model=ModelInfo)
    def model_info(request: Request) -> ModelInfo:
        return _service(request).model_info()

    @app.post("/api/blur", response_model=BlurResponse)
    def blur(
        request: Request,
        file: UploadFile = File(...),
        score_thresh: float | None = Form(None),
        nms_iou: float | None = Form(None),
        box_padding: float | None = Form(None),
        method: Literal["gaussian", "pixelate"] | None = Form(None),
    ) -> BlurResponse:
        svc = _service(request)
        data = file.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, f"Image larger than {cfg.serving.max_upload_mb} MB")
        requested = dict(score_thresh=score_thresh, nms_iou=nms_iou, box_padding=box_padding, method=method)
        overrides = {k: v for k, v in requested.items() if v is not None}
        try:
            params = svc.default_params().model_copy(update=overrides)
            params = BlurParams.model_validate(params.model_dump())
        except ValidationError as exc:
            raise HTTPException(422, exc.errors()) from exc
        try:
            return svc.blur(data, params)
        except InvalidImageError as exc:
            raise HTTPException(400, str(exc)) from exc

    # Serve the built React app from the same origin (Azure deployment). Mounted last so /api wins.
    static = cfg.resolve(cfg.serving.static_dir) if cfg.serving.static_dir else None
    if static and static.is_dir():
        app.mount("/", StaticFiles(directory=static, html=True), name="ui")

    return app
