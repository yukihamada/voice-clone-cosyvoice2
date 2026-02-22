"""
CosyVoice2 Voice Cloning — RunPod Serverless Handler
"""
import base64
import io
import os
import sys
import tempfile
import traceback

print("=== Handler starting ===", flush=True)

# Patch: huggingface_hub>=0.26 removed cached_download, but modelscope still imports it
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, 'cached_download'):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download
        print("[patch] Added cached_download shim to huggingface_hub", flush=True)
except Exception:
    pass

# Phase 1: Basic imports
try:
    import runpod
    print("[phase1] runpod imported", flush=True)
except Exception as e:
    print(f"[phase1] FATAL: runpod import failed: {e}", flush=True)
    sys.exit(1)

# Phase 2: ML imports (deferred to avoid blocking startup)
_torch = None
_torchaudio = None
_CosyVoice2 = None
_model = None

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/pretrained_models/CosyVoice2-0.5B")


def _ensure_imports():
    global _torch, _torchaudio, _CosyVoice2
    if _torch is not None:
        return
    print("[init] Importing torch...", flush=True)
    import torch
    _torch = torch
    print(f"[init] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        mem_gb = getattr(props, 'total_memory', 0) / 1e9
        print(f"[init] GPU: {torch.cuda.get_device_name(0)}, mem={mem_gb:.1f}GB", flush=True)
    import torchaudio
    _torchaudio = torchaudio
    # Use soundfile backend (TorchCodec not installed)
    try:
        torchaudio.set_audio_backend("soundfile")
    except Exception:
        pass
    print(f"[init] torchaudio {torchaudio.__version__}", flush=True)

    sys.path.insert(0, "/app/CosyVoice")
    sys.path.insert(0, "/app/CosyVoice/third_party/Matcha-TTS")

    from cosyvoice.cli.cosyvoice import CosyVoice2
    _CosyVoice2 = CosyVoice2
    print("[init] CosyVoice2 imported", flush=True)


def _get_model():
    global _model
    if _model is not None:
        return _model

    _ensure_imports()

    if not os.path.exists(os.path.join(MODEL_DIR, "cosyvoice.yaml")):
        print(f"[init] Downloading model to {MODEL_DIR}...", flush=True)
        from huggingface_hub import snapshot_download
        snapshot_download("FunAudioLLM/CosyVoice2-0.5B", local_dir=MODEL_DIR)
        print("[init] Model downloaded.", flush=True)

    print(f"[init] Loading CosyVoice2 from {MODEL_DIR}...", flush=True)
    _model = _CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False)
    print(f"[init] CosyVoice2 ready. sample_rate={_model.sample_rate}", flush=True)
    return _model


def decode_audio(audio_input: str) -> str:
    """Decode audio from base64 or URL, convert to 16kHz mono WAV via ffmpeg."""
    if audio_input.startswith(("http://", "https://")):
        import requests
        raw = requests.get(audio_input, timeout=30).content
    else:
        raw = base64.b64decode(audio_input)

    # Write raw bytes to a temp file (may be WebM, OGG, MP3, WAV, etc.)
    tmp_in = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
    tmp_in.write(raw)
    tmp_in.flush()
    tmp_in.close()

    tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_out.close()

    # Use ffmpeg to convert any format to 16kHz mono WAV
    import subprocess
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_in.name, "-ar", "16000", "-ac", "1", tmp_out.name],
        capture_output=True, timeout=30,
    )
    os.unlink(tmp_in.name)

    if result.returncode != 0:
        os.unlink(tmp_out.name)
        stderr = result.stderr.decode(errors="replace")
        raise RuntimeError(f"ffmpeg failed: {stderr[:500]}")

    return tmp_out.name


def handler(job: dict) -> dict:
    inp = job.get("input", {})
    text = inp.get("text", "").strip()
    if not text:
        return {"error": "text is required"}

    # Quick health-check mode
    if text == "__ping__":
        return {"status": "ok", "message": "handler alive"}

    mode = inp.get("mode", "zero_shot")
    prompt_audio = inp.get("prompt_audio", "")
    prompt_text = inp.get("prompt_text", "")
    instruct_text = inp.get("instruct_text", "")
    speaker_id = inp.get("speaker_id", "")
    speed = float(inp.get("speed", 1.0))
    fmt = inp.get("format", "mp3")

    try:
        model = _get_model()

        if mode == "zero_shot":
            if not prompt_audio:
                return {"error": "prompt_audio required for zero_shot"}
            wav_path = decode_audio(prompt_audio)
            results = list(model.inference_zero_shot(text, prompt_text, wav_path, stream=False))
            os.unlink(wav_path)
        elif mode == "cross_lingual":
            if not prompt_audio:
                return {"error": "prompt_audio required for cross_lingual"}
            wav_path = decode_audio(prompt_audio)
            results = list(model.inference_cross_lingual(text, wav_path, stream=False))
            os.unlink(wav_path)
        elif mode == "instruct":
            if not prompt_audio or not instruct_text:
                return {"error": "prompt_audio and instruct_text required"}
            wav_path = decode_audio(prompt_audio)
            results = list(model.inference_instruct2(text, instruct_text, wav_path, stream=False))
            os.unlink(wav_path)
        elif mode == "sft":
            available = model.list_available_spks()
            if not available:
                return {"error": "No SFT speakers available in this model"}
            spk = speaker_id or available[0]
            results = list(model.inference_sft(text, spk, stream=False))
        else:
            return {"error": f"Unknown mode: {mode}"}

        if not results:
            return {"error": "No audio generated"}

        combined = _torch.cat([r["tts_speech"] for r in results], dim=-1)
        if speed != 1.0 and 0.5 <= speed <= 2.0:
            try:
                combined, _ = _torchaudio.sox_effects.apply_effects_tensor(
                    combined, model.sample_rate, [["tempo", str(speed)]]
                )
            except Exception as e:
                print(f"[warn] sox_effects failed (speed adjust): {e}", flush=True)

        duration_ms = int(combined.shape[-1] / model.sample_rate * 1000)

        # Save as WAV first (torchaudio always supports WAV)
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav.close()
        _torchaudio.save(tmp_wav.name, combined, model.sample_rate)

        # Convert to requested format via ffmpeg
        if fmt in ("mp3", "ogg", "flac"):
            import subprocess
            tmp_out = tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False)
            tmp_out.close()
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_wav.name, "-q:a", "2", tmp_out.name],
                capture_output=True, timeout=30,
            )
            os.unlink(tmp_wav.name)
            with open(tmp_out.name, "rb") as f:
                audio_bytes = f.read()
            os.unlink(tmp_out.name)
        else:
            with open(tmp_wav.name, "rb") as f:
                audio_bytes = f.read()
            os.unlink(tmp_wav.name)
            fmt = "wav"

        return {
            "audio_base64": base64.b64encode(audio_bytes).decode(),
            "sample_rate": model.sample_rate,
            "format": fmt,
            "duration_ms": duration_ms,
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


print("=== Registering handler with RunPod ===", flush=True)
runpod.serverless.start({"handler": handler})
