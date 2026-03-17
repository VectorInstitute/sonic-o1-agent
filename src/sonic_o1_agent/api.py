"""FastAPI server for sonic-o1-agent demo.

Serves the demo UI and proxies inference to a vLLM server running
Qwen3-Omni.  The user is expected to start the vLLM server separately
(e.g. ``vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct ...``) and then
run ``sonic-o1-agent serve --vllm-url <URL>`` to launch this demo.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, Optional

import uvicorn
import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEMO_DIR = Path(__file__).parent / "demo"

# ---------------------------------------------------------------------------
# Environment / CLI configuration
# ---------------------------------------------------------------------------
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "")
MAX_VIDEO_DURATION = int(os.environ.get("MAX_VIDEO_DURATION", "1800"))  # 30 min default
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "500"))

# Upload directory — must be on a shared filesystem visible to BOTH
# the demo server and the vLLM server (they may be in separate containers).
# Falls back to a project-level path; override via SONIC_UPLOAD_DIR env var.
_default_upload_dir = os.environ.get(
    "SONIC_UPLOAD_DIR",
    str(Path.home() / ".sonic_o1_uploads"),
)
UPLOAD_DIR = Path(_default_upload_dir)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    model: str
    vllm_url: str


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise the SonicAgent on startup."""
    vllm_url = getattr(app.state, "vllm_base_url", VLLM_BASE_URL)
    config_path = getattr(app.state, "config_path", "configs/agent_config.yaml")

    # Load config
    if Path(config_path).exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        logger.warning("Config %s not found, using defaults", config_path)
        config = {}

    app.state.config = config
    app.state.vllm_base_url = vllm_url

    # Build the SonicAgent (lazy – model loads on first request via vLLM server)
    try:
        from sonic_o1_agent.agents.sonic_agent import SonicAgent

        # Inject vLLM server URL into the config so the model wrapper
        # uses the OpenAI-compatible HTTP endpoint instead of embedded vLLM.
        model_cfg = config.setdefault("model", {})
        model_cfg["vllm_base_url"] = vllm_url

        app.state.agent = SonicAgent(config)
        logger.info("SonicAgent initialised (vLLM @ %s)", vllm_url)
    except Exception as e:
        logger.warning("Could not initialise SonicAgent: %s", e)
        app.state.agent = None

    yield

    # Cleanup uploads
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    app.state.agent = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Sonic O1 Agent – Demo",
    description=(
        "Multi-agent video/audio understanding: planning, reasoning, "
        "reflection, and temporal grounding powered by Qwen3-Omni."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

if (DEMO_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=DEMO_DIR / "static"), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html_file = DEMO_DIR / "templates" / "index.html"
    if not html_file.exists():
        raise HTTPException(404, "Demo UI not found.")
    return html_file.read_text()


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    agent = getattr(request.app.state, "agent", None)
    vllm_url = getattr(request.app.state, "vllm_base_url", "")
    if agent is not None:
        return HealthResponse(status="ok", model="Qwen3-Omni-30B-A3B", vllm_url=vllm_url)
    return HealthResponse(status="starting", model="not loaded", vllm_url=vllm_url)


@app.post("/analyze/stream")
async def analyze_stream(
    request: Request,
    video: UploadFile = File(...),
    query: str = Form(...),
) -> StreamingResponse:
    """Upload video + query, stream node-by-node progress via SSE."""

    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(503, "Agent not ready.")

    if not query.strip():
        raise HTTPException(422, "Query cannot be empty.")

    # --- Save uploaded file ------------------------------------------------
    session_id = uuid.uuid4().hex[:12]
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    video_path = session_dir / video.filename
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    file_mb = video_path.stat().st_size / (1024 * 1024)
    if file_mb > MAX_UPLOAD_MB:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(422, f"File too large: {file_mb:.0f} MB (max {MAX_UPLOAD_MB} MB).")

    # --- Extract audio via ffmpeg ------------------------------------------
    audio_path = session_dir / "audio.wav"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                str(audio_path),
            ],
            capture_output=True, timeout=120, check=True,
        )
    except Exception as e:
        logger.warning("Audio extraction failed: %s — continuing video-only", e)
        audio_path = None

    # --- Check video duration ----------------------------------------------
    try:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        duration = float(probe.stdout.strip())
        if duration > MAX_VIDEO_DURATION:
            shutil.rmtree(session_dir, ignore_errors=True)
            raise HTTPException(
                422,
                f"Video too long: {duration:.0f}s (max {MAX_VIDEO_DURATION}s).",
            )
    except HTTPException:
        raise
    except Exception:
        duration = None  # couldn't probe — allow through

    # --- Stream inference ---------------------------------------------------
    def event_stream() -> Generator[str, None, None]:
        try:
            start = time.time()
            accumulated_state: Dict[str, Any] = {}

            audio_str = str(audio_path) if audio_path and audio_path.exists() else None

            for event in agent.process_stream(
                video_path=str(video_path),
                audio_path=audio_str,
                query=query,
                use_reasoning=True,
                use_reflection=True,
                use_multi_step=True,
            ):
                node = event["node"]
                state_update = event["state"]

                # Guard: some nodes (e.g. cleanup) may return None or empty
                if not state_update:
                    state_update = {}

                # Accumulate state from each node
                accumulated_state.update(state_update)

                # Send node progress event
                progress: Dict[str, Any] = {
                    "node": node,
                    "elapsed": round(time.time() - start, 1),
                }

                # Send FULL node outputs for live thinking panel
                if node == "planning":
                    if "plan" in state_update:
                        progress["plan"] = state_update["plan"]
                    if "multi_step_plan" in state_update:
                        progress["multi_step_plan"] = state_update["multi_step_plan"]
                    if "query_type" in state_update:
                        progress["query_type"] = state_update["query_type"]

                if node == "temporal_indexing" and "temporal_index" in state_update:
                    ti = state_update.get("temporal_index")
                    progress["temporal_segments"] = ti.count("\n") + 1 if ti else 0
                    progress["temporal_index"] = ti  # full caption text

                if node in ("direct", "reasoning", "multi_step"):
                    if "response" in state_update:
                        progress["response"] = state_update["response"]  # full response
                    if "reasoning_chain" in state_update:
                        progress["reasoning_chain"] = state_update["reasoning_chain"]
                    if "reasoning_trace" in state_update:
                        progress["reasoning_trace"] = state_update["reasoning_trace"]
                    if "steps_executed" in state_update:
                        progress["steps_executed"] = state_update["steps_executed"]
                    if "evidence" in state_update:
                        progress["evidence"] = state_update["evidence"]

                if node == "reflection":
                    if "reflection" in state_update:
                        progress["reflection"] = state_update["reflection"]
                    if "hallucination_assessment" in state_update:
                        progress["hallucination_assessment"] = state_update["hallucination_assessment"]
                    if "was_refined" in state_update:
                        progress["was_refined"] = state_update["was_refined"]

                yield "data: " + json.dumps(progress) + "\n\n"

            # --- Build final result from accumulated state ------------------
            result: Dict[str, Any] = {
                "response": accumulated_state.get("response", ""),
                "plan": accumulated_state.get("plan", {}),
                "reasoning_mode": accumulated_state.get("reasoning_mode", "full"),
                "modalities_used": {"video": True, "audio": audio_str is not None},
            }

            # Optional fields
            for key in [
                "reasoning_chain", "reasoning_confidence", "reasoning_trace",
                "steps_executed", "multi_step_plan", "evidence",
                "reflection", "was_refined", "original_response",
                "refinement_history", "hallucination_assessment",
            ]:
                if key in accumulated_state:
                    result[key] = accumulated_state[key]

            if "reasoning_confidence" in accumulated_state:
                result["confidence"] = accumulated_state["reasoning_confidence"]

            yield "data: " + json.dumps({"result": result}) + "\n\n"

        except Exception as e:
            logger.error("Stream error: %s", e, exc_info=True)
            yield "data: " + json.dumps({"error": str(e)}) + "\n\n"
        finally:
            try:
                shutil.rmtree(session_dir, ignore_errors=True)
            except Exception:
                pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Serve CLI entry point
# ---------------------------------------------------------------------------
def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    vllm_base_url: str = "",
    config_path: str = "configs/agent_config.yaml",
    max_video_duration: int = 300,
    reload: bool = False,
) -> None:
    """Start the Sonic O1 Agent demo server.

    Parameters
    ----------
    host : str
        Bind address.
    port : int
        Bind port.
    vllm_base_url : str
        URL of the running vLLM server (e.g. http://localhost:8080/v1).
    config_path : str
        Path to agent_config.yaml.
    max_video_duration : int
        Maximum video duration in seconds.
    reload : bool
        Enable auto-reload for development.
    """
    global MAX_VIDEO_DURATION
    MAX_VIDEO_DURATION = max_video_duration

    app.state.vllm_base_url = vllm_base_url
    app.state.config_path = config_path

    print(f"{'=' * 60}")
    print(f"  🎬 Sonic O1 Agent — Demo Server")
    print(f"  http://localhost:{port}")
    if vllm_base_url:
        print(f"  vLLM backend: {vllm_base_url}")
    print(f"{'=' * 60}")

    uvicorn.run(app, host=host, port=port, reload=reload)