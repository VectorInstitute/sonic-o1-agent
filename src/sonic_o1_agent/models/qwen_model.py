"""Qwen3-Omni model wrapper with vLLM for efficient inference.

Supports two backends:
  1. **Embedded vLLM** (default) — loads model in-process via ``vllm.LLM``.
     This is the original behaviour; nothing changes for existing callers.
  2. **vLLM server** — connects to a running ``vllm serve`` instance via the
     OpenAI-compatible API.  Activated when ``vllm_base_url`` is set in config
     or via the ``VLLM_BASE_URL`` environment variable.

The public API (``load``, ``generate``, ``unload``, etc.) is identical
regardless of which backend is active.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

import base64
import gc
import logging
import multiprocessing
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from sonic_o1_agent.core.multimodal_engine import process_mm_info

logger = logging.getLogger(__name__)


class Qwen3OmniModel:
    """Qwen3-Omni wrapper with vLLM for efficient multi-GPU inference.

    Supports Instruct and Thinking variants with audio chunking.
    Supports text-only mode for reasoning steps.
    Supports remote vLLM server via OpenAI SDK when ``vllm_base_url`` is set.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Qwen3-Omni model.

        Args:
            config: Configuration dictionary with model settings.
                Optional key ``vllm_base_url`` (str) switches to server mode.
        """
        self.model_path = config.get("model_path", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
        self.use_thinking = config.get("use_thinking", False)

        self.gpu_memory_utilization = config.get("gpu_memory_utilization", 0.85)
        self.tensor_parallel_size = config.get(
            "tensor_parallel_size", torch.cuda.device_count()
        )
        self.max_num_seqs = config.get("max_num_seqs", 1)
        self.max_model_len = config.get("max_model_len", 65536)

        gen_config = config.get("generation_config", {})
        self.temperature = gen_config.get("temperature", 0.0)
        self.top_p = gen_config.get("top_p", 0.95)
        self.top_k = gen_config.get("top_k", 20)
        self.max_tokens = gen_config.get("max_new_tokens", 8192)

        self.default_max_frames = config.get("max_frames", 256)
        self.default_min_frames = config.get("min_frames", 64)

        self.audio_feature_rate = 25.0

        self.limit_mm_per_prompt = config.get(
            "limit_mm_per_prompt", {"image": 1, "video": 1, "audio": 1}
        )

        # --- Backend selection ------------------------------------------------
        # Server mode: connect to a running vLLM serve instance.
        # Embedded mode: load model in-process (original behaviour).
        self.vllm_base_url: str = config.get(
            "vllm_base_url", os.environ.get("VLLM_BASE_URL", "")
        )
        self._server_mode = bool(self.vllm_base_url)

        # Server-mode client (lazy — created in load())
        self._client: Optional[Any] = None  # openai.OpenAI

        # Embedded-mode objects (lazy — created in load())
        self.llm: Optional[Any] = None  # vllm.LLM
        self.processor: Optional[Any] = None  # Qwen3OmniMoeProcessor

        self.stats = {
            "total_samples": 0,
            "audio_chunks_sampled": 0,
        }

        if self._server_mode:
            logger.info(
                "Qwen3OmniModel: server mode enabled (vLLM @ %s)", self.vllm_base_url
            )
        else:
            logger.info("Qwen3OmniModel: embedded mode (in-process vLLM)")

    # ======================================================================
    # LOAD / UNLOAD
    # ======================================================================

    def load(self) -> None:
        """Load the model backend.

        - Server mode: creates an ``openai.OpenAI`` client.  No weights loaded.
        - Embedded mode: loads model via ``vllm.LLM`` (original behaviour).
        """
        if self._server_mode:
            self._load_server()
        else:
            self._load_embedded()

    def _load_server(self) -> None:
        """Create OpenAI client for the remote vLLM server."""
        from openai import OpenAI

        self._client = OpenAI(
            base_url=self.vllm_base_url,
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
        )
        mode = "Thinking" if self.use_thinking else "Instruct"
        logger.info(
            "Connected to vLLM server at %s (model: %s, mode: %s)",
            self.vllm_base_url,
            self.model_path,
            mode,
        )

    def _load_embedded(self) -> None:
        """Load model weights in-process via vllm.LLM (original path)."""
        from transformers import Qwen3OmniMoeProcessor
        from vllm import LLM  # noqa: F811

        try:
            self._clear_vllm_cache()

            # Suppress verbose vLLM logging
            vllm_loggers = [
                "vllm",
                "vllm.engine",
                "vllm.worker",
                "vllm.multiproc_executor",
                "vllm.parallel_state",
                "vllm.config",
                "vllm.engine.arg_utils",
            ]
            for logger_name in vllm_loggers:
                logging.getLogger(logger_name).setLevel(logging.WARNING)

            os.environ["VLLM_USE_V1"] = "0"
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
            if self.max_model_len > 65536:
                os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

            logger.info(f"Loading Qwen3-Omni model from {self.model_path} with vLLM")
            logger.info(
                f"Using {self.tensor_parallel_size} GPUs for tensor parallelism"
            )
            logger.info(f"Context length: {self.max_model_len} tokens")

            self.llm = LLM(
                model=self.model_path,
                trust_remote_code=True,
                gpu_memory_utilization=self.gpu_memory_utilization,
                tensor_parallel_size=self.tensor_parallel_size,
                limit_mm_per_prompt=self.limit_mm_per_prompt,
                max_num_seqs=self.max_num_seqs,
                max_model_len=self.max_model_len,
                seed=1234,
                disable_log_stats=True,
                enforce_eager=False,
                enable_prefix_caching=True,
                mm_processor_kwargs={"cache_gb": 0},
            )

            self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)

            mode = "Thinking" if self.use_thinking else "Instruct"
            logger.info(f"Successfully loaded Qwen3-Omni with vLLM ({mode} mode)")

        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen3-Omni model with vLLM: {e}")

    def unload(self) -> None:
        """Release resources.

        - Server mode: drops the client reference.
        - Embedded mode: full GPU cleanup (original behaviour).
        """
        if self._server_mode:
            self._client = None
            logger.info("Server client released")
            return

        # --- Embedded cleanup (unchanged) ---
        if self.llm is not None:
            try:
                del self.llm
            except Exception as e:
                logger.warning(f"Error deleting llm object: {e}")
            self.llm = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        if dist.is_initialized():
            try:
                dist.destroy_process_group()
                logger.info("Distributed process group destroyed")
            except Exception as e:
                logger.warning(f"Failed to destroy process group: {e}")

        try:
            active_children = multiprocessing.active_children()
            if active_children:
                logger.info(
                    f"Found {len(active_children)} active child processes. "
                    "Terminating..."
                )
                for child in active_children:
                    try:
                        child.terminate()
                        child.join(timeout=0.5)
                        if child.is_alive():
                            child.kill()
                    except Exception as e:
                        logger.warning(f"Failed to kill child {child.pid}: {e}")
        except Exception as e:
            logger.warning(f"Error during manual process cleanup: {e}")

        logger.info("Model unloaded, memory cleared, and child processes terminated")

    # ======================================================================
    # GENERATE — main entry point
    # ======================================================================

    def generate(
        self,
        video_path: Optional[str],
        audio_path: Optional[str],
        prompt: str,
        max_frames: Optional[int] = None,
        max_audio_chunks: Optional[int] = None,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate response from video and/or audio.

        Routes to the server or embedded backend transparently.
        Signature and return type are identical for both backends.

        Args:
            video_path: Video file path or None
            audio_path: Audio file path or None
            prompt: Text prompt
            max_frames: Maximum frames to use
            max_audio_chunks: Maximum audio chunks
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (generated_text, metadata_dict)
        """
        if self._server_mode:
            return self._generate_via_server(
                video_path, audio_path, prompt, max_frames, max_audio_chunks, **kwargs
            )
        else:
            return self._generate_embedded(
                video_path, audio_path, prompt, max_frames, max_audio_chunks, **kwargs
            )

    # ======================================================================
    # SERVER MODE — OpenAI SDK path
    # ======================================================================

    def _ensure_server_client(self) -> None:
        """Ensure the OpenAI client is available, lazy-load if needed."""
        if self._client is None:
            self._load_server()
        assert self._client is not None, (
            "Server client not initialised. Is vllm_base_url set?"
        )

    def _generate_via_server(
        self,
        video_path: Optional[str],
        audio_path: Optional[str],
        prompt: str,
        max_frames: Optional[int] = None,
        max_audio_chunks: Optional[int] = None,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate response via the remote vLLM server (OpenAI SDK).

        Performs **client-side frame extraction** using the same
        ``video_processor`` as the embedded path, then sends pre-sampled
        frames as base64 JPEGs to the vLLM server.  This gives full
        control over frame count (``max_frames`` / ``min_frames`` from
        config) and avoids vLLM's default 32-frame sampling and frame
        mismatch errors.

        Audio is sent as base64 ``input_audio``.

        Supports ``video_start``/``video_end``/``audio_start``/``audio_end``
        kwargs for segment-level processing.

        Args:
            video_path: Local video file path or None
            audio_path: Local audio file path or None
            prompt: Text prompt
            max_frames: Maximum video frames to sample
            max_audio_chunks: Not used in server mode
            **kwargs: Additional params (temperature, top_p, video_start,
                video_end, audio_start, audio_end, etc.)

        Returns:
            Tuple of (generated_text, metadata_dict)
        """
        self._ensure_server_client()

        has_video = video_path is not None and Path(video_path).exists()
        has_audio = audio_path is not None and Path(audio_path).exists()

        # Text-only mode (reasoning steps)
        if not has_video and not has_audio:
            return self._generate_text_only_server(prompt, kwargs)

        # --- Build multimodal content ------------------------------------
        content: List[Dict[str, Any]] = []
        metadata: Dict[str, Any] = {}

        # --- Video: client-side frame extraction -------------------------
        if has_video:
            assert video_path is not None
            actual_max_frames = max_frames or self.default_max_frames
            actual_min_frames = min(self.default_min_frames, actual_max_frames)

            video_start = kwargs.get("video_start")
            video_end = kwargs.get("video_end")

            try:
                frames_b64, video_meta = self._extract_frames_for_server(
                    video_path=video_path,
                    max_frames=actual_max_frames,
                    min_frames=actual_min_frames,
                    video_start=video_start,
                    video_end=video_end,
                )

                # Send frames as video/jpeg (vLLM treats this as pre-extracted)
                content.append(
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:video/jpeg;base64,{','.join(frames_b64)}",
                        },
                    }
                )
                metadata["video_metadata"] = [video_meta]
                logger.info(
                    "Server mode: %d frames extracted (max=%d, %s)",
                    video_meta.get("frames_sampled", 0),
                    actual_max_frames,
                    f"{video_start:.0f}s-{video_end:.0f}s"
                    if video_start is not None
                    else "full",
                )
            except Exception as e:
                logger.warning("Frame extraction failed: %s — sending raw video", e)
                # Fallback: send raw video as base64
                try:
                    video_b64 = self._encode_video_base64(video_path)
                    content.append(
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
                        }
                    )
                except Exception as e2:
                    logger.error("Video encoding also failed: %s", e2)

        # --- Audio: base64 -----------------------------------------------
        if has_audio:
            assert audio_path is not None
            audio_start = kwargs.get("audio_start")
            audio_end = kwargs.get("audio_end")
            actual_audio = audio_path

            # Cut audio segment if time range specified
            temp_audio = None
            if audio_start is not None and audio_end is not None:
                actual_audio = self._ffmpeg_cut_audio(
                    audio_path, audio_start, audio_end
                )
                if actual_audio != audio_path:
                    temp_audio = actual_audio

            try:
                audio_b64 = self._encode_audio_base64(actual_audio)
                audio_ext = Path(actual_audio).suffix.lstrip(".") or "wav"
                content.append(
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": audio_ext},
                    }
                )
                metadata["audio_path"] = actual_audio
                logger.info("Server mode: audio via base64 (%s)", audio_ext)
            except Exception as e:
                logger.warning("Audio encoding failed: %s — skipping", e)
            finally:
                if temp_audio:
                    try:
                        Path(temp_audio).unlink(missing_ok=True)
                    except Exception:
                        pass

        # --- Text prompt -------------------------------------------------
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        # --- Send to vLLM ------------------------------------------------
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        max_tokens = kwargs.get("max_new_tokens", self.max_tokens)

        self.stats["total_samples"] += 1

        try:
            logger.info("Server mode: sending request to vLLM...")
            assert self._client is not None

            completion = self._client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            response_text = completion.choices[0].message.content or ""
            logger.info("Server mode: received response (%d chars)", len(response_text))
            return response_text, metadata

        except Exception as e:
            logger.error("Server mode generation failed: %s", e, exc_info=True)
            raise RuntimeError(f"Server mode generation failed: {e}")

    def _extract_frames_for_server(
        self,
        video_path: str,
        max_frames: int,
        min_frames: int,
        video_start: Optional[float] = None,
        video_end: Optional[float] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Extract and encode video frames client-side for the vLLM server.

        Uses the same frame sampling logic as the embedded mode
        (``process_video_with_metadata``) but returns base64-encoded
        JPEG strings instead of tensors.

        Args:
            video_path: Path to video file.
            max_frames: Maximum frames to sample.
            min_frames: Minimum frames to sample.
            video_start: Start time in seconds (optional).
            video_end: End time in seconds (optional).

        Returns:
            Tuple of (list of base64 JPEG strings, metadata dict).
        """
        import io

        from PIL import Image

        from sonic_o1_agent.core.video_processor import process_video_with_metadata

        frames_tensor, video_meta = process_video_with_metadata(
            video_path=video_path,
            max_frames=max_frames,
            min_frames=min_frames,
            video_start=video_start,
            video_end=video_end,
        )

        # Convert tensor frames to base64 JPEGs
        frames_b64: List[str] = []
        # frames_tensor shape: (N, C, H, W) float
        for i in range(frames_tensor.shape[0]):
            frame = frames_tensor[i].permute(1, 2, 0).byte().numpy()
            img = Image.fromarray(frame)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            frames_b64.append(b64)

        logger.info(
            "Extracted %d frames from %s (max=%d, min=%d)",
            len(frames_b64),
            video_path,
            max_frames,
            min_frames,
        )

        return frames_b64, video_meta

    @staticmethod
    def _ffmpeg_cut_audio(
        input_path: str,
        start_sec: float,
        end_sec: float,
    ) -> str:
        """Cut an audio segment using ffmpeg. Returns temp path or original."""
        import subprocess
        import tempfile

        duration = end_sec - start_sec
        if duration <= 0:
            return input_path

        suffix = Path(input_path).suffix or ".wav"

        try:
            tmp = tempfile.NamedTemporaryFile(
                suffix=suffix, prefix="sonic_aseg_", delete=False
            )
            tmp_path = tmp.name
            tmp.close()

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(start_sec),
                    "-i",
                    input_path,
                    "-t",
                    str(duration),
                    "-c",
                    "copy",
                    "-avoid_negative_ts",
                    "1",
                    tmp_path,
                ],
                capture_output=True,
                timeout=60,
                check=True,
            )
            return tmp_path

        except Exception as e:
            logger.warning(
                "Audio cut failed for %.1fs-%.1fs: %s — sending full audio",
                start_sec,
                end_sec,
                e,
            )
            return input_path

    @staticmethod
    def _encode_video_base64(video_path: str) -> str:
        """Read a video file and return its base64-encoded content (fallback)."""
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _generate_text_only_server(
        self, prompt: str, kwargs: Dict
    ) -> Tuple[str, Dict[str, Any]]:
        """Text-only generation via server (for reasoning steps)."""
        self._ensure_server_client()

        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        max_tokens = kwargs.get("max_new_tokens", self.max_tokens)

        logger.info("Server mode: text-only generation...")
        assert self._client is not None

        try:
            completion = self._client.chat.completions.create(
                model=self.model_path,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_body={"modalities": ["text"]},
            )

            response_text = completion.choices[0].message.content or ""
            logger.info(
                "Server mode: text-only response (%d chars)", len(response_text)
            )
            return response_text, {}

        except Exception as e:
            logger.error("Server text-only generation failed: %s", e, exc_info=True)
            raise RuntimeError(f"Server text-only generation failed: {e}")

    @staticmethod
    def _encode_audio_base64(audio_path: str) -> str:
        """Read an audio file and return its base64-encoded content."""
        with open(audio_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # ======================================================================
    # EMBEDDED MODE — original in-process vLLM path (UNCHANGED)
    # ======================================================================

    def _generate_embedded(
        self,
        video_path: Optional[str],
        audio_path: Optional[str],
        prompt: str,
        max_frames: Optional[int] = None,
        max_audio_chunks: Optional[int] = None,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate response using embedded vLLM engine (original code).

        This method is the original ``generate()`` body, completely unchanged.
        """
        from vllm import SamplingParams

        if self.llm is None or self.processor is None:
            try:
                logger.warning(
                    "Model found unloaded in generate(), attempting lazy load..."
                )
                self.load()
            except Exception:
                raise RuntimeError("Model not loaded. Call load() first.")

        assert self.llm is not None
        assert self.processor is not None

        # Determine modality mode
        has_video = video_path is not None
        has_audio = audio_path is not None

        # Allow text-only mode for reasoning steps
        if not has_video and not has_audio:
            modality_mode = "text-only"
            logger.info(f"Modality mode: {modality_mode}")
            return self._generate_text_only(prompt, kwargs)
        elif has_video and has_audio:
            modality_mode = "video+audio"
        elif has_video:
            modality_mode = "video-only"
        else:
            modality_mode = "audio-only"

        logger.info(f"Modality mode: {modality_mode}")

        if has_video:
            assert video_path is not None
            video_path_obj = Path(video_path)
            if not video_path_obj.exists():
                raise FileNotFoundError(f"Video file not found: {video_path_obj}")

        actual_max_frames = (
            max_frames if max_frames is not None else self.default_max_frames
        )

        try:
            logger.info(
                f"Processing: frames={actual_max_frames if has_video else 'N/A'}, "
                f"max_audio_chunks={max_audio_chunks}"
            )

            content = []

            if has_video:
                video_content = {
                    "type": "video",
                    "video": str(video_path),
                    "max_frames": actual_max_frames,
                    "min_frames": min(self.default_min_frames, actual_max_frames),
                }
                # Optional time-range for segment-level processing
                if kwargs.get("video_start") is not None:
                    video_content["video_start"] = kwargs["video_start"]
                if kwargs.get("video_end") is not None:
                    video_content["video_end"] = kwargs["video_end"]
                content.append(video_content)

            if has_audio and audio_path is not None and os.path.exists(audio_path):
                try:
                    import av

                    test_container = av.open(audio_path)
                    if len(test_container.streams.audio) > 0:
                        audio_content = {
                            "type": "audio",
                            "audio": str(audio_path),
                        }
                        # Optional time-range for segment-level processing
                        if kwargs.get("audio_start") is not None:
                            audio_content["audio_start"] = kwargs["audio_start"]
                        if kwargs.get("audio_end") is not None:
                            audio_content["audio_end"] = kwargs["audio_end"]
                        content.append(audio_content)
                    else:
                        logger.info(
                            f"Audio file {audio_path} has no audio stream, skipping"
                        )
                        has_audio = False
                    test_container.close()
                except Exception as e:
                    logger.warning(
                        f"Could not verify audio file {audio_path}: {e}. Skipping audio."
                    )
                    has_audio = False

            content.append({"type": "text", "text": prompt})

            conversation = [{"role": "user", "content": content}]

            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )

            audios, images, videos, mm_metadata = process_mm_info(
                conversation,
                use_audio_in_video=False,
                max_audio_duration=None,
                max_audio_chunks=max_audio_chunks,
                audio_chunk_duration_sec=kwargs.get("audio_chunk_duration_sec", 10.0),
            )

            if audios is not None:
                audios = [a for a in audios if len(a) > 0]
                if len(audios) == 0:
                    audios = None

            if audios is None and "<|audio_pad|>" in text:
                text = text.replace("<|audio_pad|>", "").strip()
                logger.info("Removed <|audio_pad|> from prompt (no audio available)")

            if max_audio_chunks is not None and audios is not None:
                self.stats["audio_chunks_sampled"] += 1

            self.stats["total_samples"] += 1

            inputs = {"prompt": text, "multi_modal_data": {}}

            if audios is not None:
                inputs["multi_modal_data"]["audio"] = audios
            if images is not None:
                inputs["multi_modal_data"]["image"] = images
            if videos is not None:
                inputs["multi_modal_data"]["video"] = videos

            temperature = kwargs.get("temperature", self.temperature)
            top_p = kwargs.get("top_p", self.top_p)
            top_k = kwargs.get("top_k", self.top_k)
            max_tokens = kwargs.get("max_new_tokens", self.max_tokens)

            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
            )

            logger.info("Generating response...")

            outputs = self.llm.generate([inputs], sampling_params=sampling_params)
            response_text = outputs[0].outputs[0].text

            logger.info(f"Generated response ({len(response_text)} chars)")

            return response_text, mm_metadata

        except Exception as e:
            error_msg = str(e)

            is_cache_error = (
                "Expected a cached item" in error_msg
                or "mm_hash" in error_msg
                or "AssertionError" in error_msg
            )
            is_engine_dead = (
                "EngineDeadError" in error_msg
                or "EngineCore" in error_msg
                or "process_input_sockets" in error_msg
            )
            is_context_error = any(
                keyword in error_msg.lower()
                for keyword in [
                    "context",
                    "token",
                    "length",
                    "limit",
                    "maximum",
                    "exceed",
                    "longer than",
                ]
            )
            is_oom = "out of memory" in error_msg.lower() or "OOM" in error_msg

            if is_engine_dead or is_cache_error:
                logger.error(f"Engine/Cache error: {e}")
                self._reload_engine()
                raise RuntimeError(f"Engine/cache error (engine reloaded): {e}")

            if is_context_error:
                logger.error(f"Context length error: {e}")
                self._reload_engine()
                raise RuntimeError(f"Context length exceeded: {e}")

            if is_oom:
                logger.error(f"OOM error: {e}")
                self.unload()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._clear_vllm_cache()
                self.load()
                raise RuntimeError(f"Out of memory (engine reloaded): {e}")

            logger.error(f"Generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Generation failed: {e}")

    def _generate_text_only(
        self, prompt: str, kwargs: Dict
    ) -> Tuple[str, Dict[str, Any]]:
        """Text-only generation for embedded mode (original code, unchanged)."""
        from vllm import SamplingParams

        logger.info("Running text-only generation...")

        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        top_k = kwargs.get("top_k", self.top_k)
        max_tokens = kwargs.get("max_new_tokens", self.max_tokens)

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )

        assert self.llm is not None
        assert self.processor is not None

        # Wrap in chat template so the model sees role markers
        conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        formatted_prompt = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        inputs = {"prompt": formatted_prompt, "multi_modal_data": {}}

        try:
            outputs = self.llm.generate([inputs], sampling_params=sampling_params)
            response_text: str = outputs[0].outputs[0].text

            logger.info(f"Generated text-only response ({len(response_text)} chars)")
            return response_text, {}

        except Exception as e:
            logger.error(f"Text-only generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Text-only generation failed: {e}")

    # ======================================================================
    # EMBEDDED MODE — helper methods (UNCHANGED)
    # ======================================================================

    def _clear_vllm_cache(self):
        """Clear vLLM multimodal cache."""
        vllm_cache = Path(
            os.environ.get("VLLM_CACHE_ROOT", Path.home() / ".cache/vllm")
        )
        mm_cache = vllm_cache / "multimodal_cache"
        if mm_cache.exists():
            try:
                shutil.rmtree(mm_cache, ignore_errors=True)
                logger.debug(f"Cleared multimodal cache: {mm_cache}")
            except Exception as e:
                logger.warning(f"Failed to clear cache: {e}")

    def _is_engine_alive(self) -> bool:
        """Check if vLLM engine is responsive."""
        if self.llm is None:
            return False

        from vllm import SamplingParams

        try:
            self.llm.generate(
                [{"prompt": "test", "multi_modal_data": {}}],
                SamplingParams(max_tokens=1),
            )
            return True
        except Exception:
            return False

    def _reload_engine(self):
        """Reload vLLM engine after crash."""
        logger.warning("Engine crashed, attempting reload")

        try:
            self.unload()
        except Exception as e:
            logger.warning(f"Error during unload: {e}")

        self._clear_vllm_cache()
        time.sleep(15)

        try:
            self.load()
            logger.info("Engine reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload engine: {e}")
            raise RuntimeError(f"Could not recover from engine crash: {e}")

    # ======================================================================
    # INFO / STATS
    # ======================================================================

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and configuration."""
        return {
            "model_path": self.model_path,
            "model_type": "Thinking" if self.use_thinking else "Instruct",
            "backend": "vLLM-server" if self._server_mode else "vLLM-embedded",
            "vllm_base_url": self.vllm_base_url if self._server_mode else None,
            "native_video": True,
            "native_audio": True,
            "text_only_support": True,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "default_max_frames": self.default_max_frames,
            "default_min_frames": self.default_min_frames,
            "max_model_len": self.max_model_len,
            "audio_feature_rate": self.audio_feature_rate,
            "statistics": self.stats,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get model statistics."""
        return self.stats
