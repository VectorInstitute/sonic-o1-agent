"""Qwen3-Omni model wrapper with vLLM for efficient inference.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

import gc
import logging
import multiprocessing
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from transformers import Qwen3OmniMoeProcessor
from vllm import LLM, SamplingParams

from sonic_o1_agent.core.multimodal_engine import process_mm_info

logger = logging.getLogger(__name__)


class Qwen3OmniModel:
    """Qwen3-Omni wrapper with vLLM for efficient multi-GPU inference.

    Supports Instruct and Thinking variants with audio chunking.
    Supports text-only mode for reasoning steps.
    """

    AUDIO_TOKENS_PER_SEC = 25
    VIDEO_TOKENS_PER_FRAME = 250

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Qwen3-Omni model.

        Args:
            config: Configuration dictionary with model settings
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

        self.llm: Optional[LLM] = None
        self.processor: Optional[Qwen3OmniMoeProcessor] = None

        self.stats = {
            "total_samples": 0,
            "audio_chunks_sampled": 0,
        }

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

    def load(self) -> None:
        """Load the Qwen3 model with vLLM."""
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
            # Suppress vLLM verbose output
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

    def generate(
        self,
        video_path: Optional[str],
        audio_path: Optional[str],
        prompt: str,
        max_frames: Optional[int] = None,
        max_audio_chunks: Optional[int] = None,
        **kwargs,
    ) -> tuple[str, dict[str, Any]]:
        """Generate response from video and/or audio.

        Supports modality ablation: video-only, audio-only, both, text-only.

        Args:
            video_path: Video file path or None
            audio_path: Audio file path or None
            prompt: Text prompt
            max_frames: Maximum frames to use
            max_audio_chunks: Maximum audio chunks
            **kwargs: Additional generation parameters

        Returns:
            Generated text response
        """
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
                    logger.warning(f"Could not verify audio file {audio_path}: {e}")
                    audio_content = {
                        "type": "audio",
                        "audio": str(audio_path),
                    }
                    if kwargs.get("audio_start") is not None:
                        audio_content["audio_start"] = kwargs["audio_start"]
                    if kwargs.get("audio_end") is not None:
                        audio_content["audio_end"] = kwargs["audio_end"]
                    content.append(audio_content)

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
    ) -> tuple[str, dict[str, Any]]:
        """Generate response for text-only queries (no video/audio).

        Used for reasoning steps that don't need multimodal input.
        Applies the chat template so the model sees proper role
        markers and the generation prompt (``<|im_start|>assistant``).

        Args:
            prompt: Text prompt
            kwargs: Additional generation parameters

        Returns:
            Tuple of (generated text response, empty metadata dict)
        """
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

    def unload(self) -> None:
        """Aggressively cleanup vLLM to prevent zombie processes."""
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

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and configuration."""
        return {
            "model_path": self.model_path,
            "model_type": "Thinking" if self.use_thinking else "Instruct",
            "backend": "vLLM",
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
