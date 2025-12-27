"""
VieNeu-TTS MLX Backend for macOS Apple Silicon

This module provides native MLX support for VieNeu-TTS on Apple Silicon (M1/M2/M3/M4).
Uses mlx-lm for the Qwen backbone and NeuCodec with PyTorch MPS encoder + MLX decoder.

Architecture:
    Reference Audio → PyTorch Encoder (MPS) → codes (cached)
    Text → mlx-lm backbone (MLX) → speech codes → MLX Decoder → Audio
                                        └── Zero-copy unified memory ──┘
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Generator

import librosa
import numpy as np

# Check for macOS
if sys.platform != "darwin":
    raise ImportError(
        "vieneu_tts_mlx is only supported on macOS with Apple Silicon. "
        "Use vieneu_tts for other platforms."
    )

# Lazy imports for MLX and PyTorch
_mlx_available = None
_mps_available = None


def _check_mlx():
    global _mlx_available
    if _mlx_available is None:
        try:
            import mlx.core as mx
            import mlx_lm
            _mlx_available = True
        except ImportError:
            _mlx_available = False
    return _mlx_available


def _check_mps():
    global _mps_available
    if _mps_available is None:
        try:
            import torch
            _mps_available = torch.backends.mps.is_available()
        except ImportError:
            _mps_available = False
    return _mps_available


def _linear_overlap_add(frames: list[np.ndarray], stride: int) -> np.ndarray:
    """Linear overlap-add for smooth audio concatenation"""
    assert len(frames)
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]

    total_size = 0
    for i, frame in enumerate(frames):
        frame_end = stride * i + frame.shape[-1]
        total_size = max(total_size, frame_end)

    sum_weight = np.zeros(total_size, dtype=dtype)
    out = np.zeros(*shape, total_size, dtype=dtype)

    offset: int = 0
    for frame in frames:
        frame_length = frame.shape[-1]
        t = np.linspace(0, 1, frame_length + 2, dtype=dtype)[1:-1]
        weight = np.abs(0.5 - (t - 0.5))

        out[..., offset : offset + frame_length] += weight * frame
        sum_weight[offset : offset + frame_length] += weight
        offset += stride
    assert sum_weight.min() > 0
    return out / sum_weight


class MLXVieNeuTTS:
    """
    Apple Silicon optimized VieNeu-TTS using MLX.

    Uses:
    - mlx-lm for Qwen backbone inference (~200-250 tok/s on M4)
    - PyTorch MPS for NeuCodec encoder (one-time per voice)
    - PyTorch MPS for NeuCodec decoder (fallback, or MLX decoder if available)

    Example:
        >>> from vieneu_tts_mlx import MLXVieNeuTTS
        >>> tts = MLXVieNeuTTS()
        >>> ref_codes = tts.encode_reference("reference.wav")
        >>> audio = tts.infer("Xin chào!", ref_codes, "Xin chào thế giới")
    """

    def __init__(
        self,
        backbone_repo: str = "pnnbao-ump/VieNeu-TTS",
        codec_repo: str = "neuphonic/neucodec",
        use_mlx_decoder: bool = False,
    ):
        """
        Initialize MLX VieNeu-TTS.

        Args:
            backbone_repo: HuggingFace repo for the Qwen backbone model
            codec_repo: HuggingFace repo for NeuCodec
            use_mlx_decoder: If True, use MLX decoder (zero-copy).
                           If False, use PyTorch MPS decoder.
        """
        if not _check_mlx():
            raise ImportError(
                "MLX is required for MLXVieNeuTTS. Install with:\n"
                "  pip install mlx mlx-lm"
            )

        if not _check_mps():
            raise ImportError(
                "PyTorch with MPS support is required for the NeuCodec encoder.\n"
                "Install with: pip install torch torchaudio"
            )

        # Constants (same as VieNeuTTS)
        self.sample_rate = 24_000
        self.max_context = 2048
        self.hop_length = 480
        self.streaming_overlap_frames = 1
        self.streaming_frames_per_chunk = 25
        self.streaming_lookforward = 5
        self.streaming_lookback = 50
        self.streaming_stride_samples = self.streaming_frames_per_chunk * self.hop_length

        # Flags
        self._use_mlx_decoder = use_mlx_decoder
        self._mlx_decoder_loaded = False

        # Load models
        self._load_backbone(backbone_repo)
        self._load_codec(codec_repo, use_mlx_decoder)

    def _load_backbone(self, backbone_repo: str):
        """Load Qwen backbone using mlx-lm"""
        print(f"Loading MLX backbone from: {backbone_repo} ...")

        from mlx_lm import load

        # mlx-lm handles model loading and conversion automatically
        self.model, self.tokenizer = load(backbone_repo)

        print("   ✅ MLX backbone loaded successfully")

    def _load_codec(self, codec_repo: str, use_mlx_decoder: bool):
        """Load NeuCodec - encoder on MPS, decoder on MLX (optional)"""
        import torch
        from neucodec import NeuCodec

        print(f"Loading codec from: {codec_repo} ...")

        # Load full codec on MPS for encoder
        self.codec = NeuCodec.from_pretrained(codec_repo)
        self.codec.eval().to("mps")

        print("   ✅ NeuCodec encoder loaded on MPS")

        if use_mlx_decoder:
            try:
                from .decoder import MLXNeuCodecDecoder
                self.mlx_decoder = MLXNeuCodecDecoder.from_pretrained(codec_repo)
                self._mlx_decoder_loaded = True
                print("   ✅ NeuCodec decoder loaded on MLX (zero-copy)")
            except ImportError as e:
                print(f"   ⚠️ MLX decoder not available: {e}")
                print("   → Using PyTorch MPS decoder instead")
                self._mlx_decoder_loaded = False
                self._use_mlx_decoder = False

    def encode_reference(self, ref_audio_path: str | Path) -> np.ndarray:
        """
        Encode reference audio to codes using NeuCodec (PyTorch MPS).

        This only needs to be called once per voice - the codes can be cached
        and reused for all subsequent inferences with that voice.

        Args:
            ref_audio_path: Path to reference audio file (WAV, MP3, etc.)

        Returns:
            np.ndarray: Reference audio codes (typically ~500 integers for 10s audio)
        """
        import torch

        wav, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        # Keep on CPU - neucodec will handle device transfer internally
        # wav_tensor = wav_tensor.to("mps")  # Don't move to MPS, causes issues

        with torch.no_grad():
            ref_codes = self.codec.encode_code(audio_or_path=wav_tensor)
            # Ensure we move to CPU before converting to numpy
            if hasattr(ref_codes, 'cpu'):
                ref_codes = ref_codes.cpu()
            ref_codes = ref_codes.squeeze(0).squeeze(0)
            if hasattr(ref_codes, 'numpy'):
                ref_codes = ref_codes.numpy()

        return ref_codes

    def infer(
        self,
        text: str,
        ref_codes: np.ndarray,
        ref_text: str,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> np.ndarray:
        """
        Generate speech from text using MLX backbone.

        Args:
            text: Input text to convert to speech
            ref_codes: Reference audio codes from encode_reference()
            ref_text: Transcription of the reference audio
            temperature: Sampling temperature (default: 1.0)
            top_k: Top-k sampling parameter (default: 50)

        Returns:
            np.ndarray: Generated audio waveform at 24kHz
        """
        # Build prompt
        prompt = self._build_prompt(ref_codes, ref_text, text)

        # Generate with mlx-lm
        output_str = self._generate_mlx(prompt, temperature, top_k)

        # Decode speech tokens to audio
        wav = self._decode(output_str)

        return wav

    def infer_stream(
        self,
        text: str,
        ref_codes: np.ndarray,
        ref_text: str,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Generator[np.ndarray, None, None]:
        """
        Stream speech generation from text using MLX backbone.

        Args:
            text: Input text to convert to speech
            ref_codes: Reference audio codes from encode_reference()
            ref_text: Transcription of the reference audio
            temperature: Sampling temperature (default: 1.0)
            top_k: Top-k sampling parameter (default: 50)

        Yields:
            np.ndarray: Chunks of generated audio waveform at 24kHz
        """
        # Build prompt
        prompt = self._build_prompt(ref_codes, ref_text, text)

        # Streaming generation with mlx-lm
        audio_cache: list[np.ndarray] = []
        token_cache: list[str] = [f"<|speech_{idx}|>" for idx in ref_codes]
        n_decoded_samples: int = 0
        n_decoded_tokens: int = len(ref_codes)

        for token_str in self._generate_mlx_stream(prompt, temperature, top_k):
            token_cache.append(token_str)

            # Check if we have enough tokens to decode a chunk
            if len(token_cache[n_decoded_tokens:]) >= self.streaming_frames_per_chunk + self.streaming_lookforward:
                # Decode chunk
                tokens_start = max(
                    n_decoded_tokens - self.streaming_lookback - self.streaming_overlap_frames,
                    0
                )
                tokens_end = (
                    n_decoded_tokens
                    + self.streaming_frames_per_chunk
                    + self.streaming_lookforward
                    + self.streaming_overlap_frames
                )
                sample_start = (n_decoded_tokens - tokens_start) * self.hop_length
                sample_end = (
                    sample_start
                    + (self.streaming_frames_per_chunk + 2 * self.streaming_overlap_frames) * self.hop_length
                )

                curr_codes = token_cache[tokens_start:tokens_end]
                recon = self._decode("".join(curr_codes))
                recon = recon[sample_start:sample_end]
                audio_cache.append(recon)

                # Post-process with overlap-add
                processed_recon = _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)
                new_samples_end = len(audio_cache) * self.streaming_stride_samples
                processed_recon = processed_recon[n_decoded_samples:new_samples_end]
                n_decoded_samples = new_samples_end
                n_decoded_tokens += self.streaming_frames_per_chunk

                yield processed_recon

        # Final chunk
        remaining_tokens = len(token_cache) - n_decoded_tokens
        if remaining_tokens > 0:
            tokens_start = max(
                len(token_cache) - (self.streaming_lookback + self.streaming_overlap_frames + remaining_tokens),
                0
            )
            sample_start = (
                len(token_cache) - tokens_start - remaining_tokens - self.streaming_overlap_frames
            ) * self.hop_length

            curr_codes = token_cache[tokens_start:]
            recon = self._decode("".join(curr_codes))
            recon = recon[sample_start:]
            audio_cache.append(recon)

            processed_recon = _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)
            yield processed_recon[n_decoded_samples:]

    def _build_prompt(self, ref_codes: np.ndarray, ref_text: str, input_text: str) -> str:
        """Build the prompt string for generation"""
        from utils.phonemize_text import phonemize_with_dict

        ref_text_phoneme = phonemize_with_dict(ref_text)
        input_text_phoneme = phonemize_with_dict(input_text)

        codes_str = "".join([f"<|speech_{int(idx)}|>" for idx in ref_codes])

        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text_phoneme} {input_text_phoneme}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )

        return prompt

    def _generate_mlx(self, prompt: str, temperature: float, top_k: int) -> str:
        """Generate completion using mlx-lm"""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        # Create sampler with temperature
        sampler = make_sampler(temp=temperature, top_p=1.0)

        # Generate tokens
        output = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_context,
            sampler=sampler,
            verbose=False,
        )

        # Extract only the new tokens (after the prompt)
        # mlx-lm returns the full output including prompt
        if output.startswith(prompt):
            output = output[len(prompt):]

        # Stop at end token
        if "<|SPEECH_GENERATION_END|>" in output:
            output = output.split("<|SPEECH_GENERATION_END|>")[0]

        return output

    def _generate_mlx_stream(
        self,
        prompt: str,
        temperature: float,
        top_k: int,
    ) -> Generator[str, None, None]:
        """Stream generation using mlx-lm"""
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        # Create sampler with temperature
        sampler = make_sampler(temp=temperature, top_p=1.0)

        # Stream tokens
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_context,
            sampler=sampler,
        ):
            token_str = response.text if hasattr(response, 'text') else response

            # Stop at end token
            if "<|SPEECH_GENERATION_END|>" in token_str:
                break

            yield token_str

    def _decode(self, codes_str: str) -> np.ndarray:
        """Decode speech tokens to audio waveform"""
        import torch

        # Extract speech token IDs
        speech_ids = [int(num) for num in re.findall(r"<\|speech_(\d+)\|>", codes_str)]

        if len(speech_ids) == 0:
            raise ValueError(
                "No valid speech tokens found in the output. "
                "The model may not have generated proper speech tokens."
            )

        # Use MLX decoder if available (zero-copy path)
        if self._use_mlx_decoder and self._mlx_decoder_loaded:
            import mlx.core as mx
            codes_mx = mx.array(speech_ids, dtype=mx.int32)[None, None, :]  # [1, 1, T]
            recon = self.mlx_decoder.decode_code(codes_mx)
            recon = np.array(recon)  # MLX array to numpy (zero-copy on unified memory)
            return recon[0, 0, :]

        # Fallback to PyTorch MPS decoder
        with torch.no_grad():
            codes = torch.tensor(speech_ids, dtype=torch.long)[None, None, :].to("mps")
            recon = self.codec.decode_code(codes).cpu().numpy()

        return recon[0, 0, :]
