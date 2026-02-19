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

import runpod
print("runpod imported", flush=True)
import torch
print(f"torch {torch.__version__} imported, cuda={torch.cuda.is_available()}", flush=True)
import torchaudio
print(f"torchaudio {torchaudio.__version__} imported", flush=True)

sys.path.append("/app/CosyVoice")
sys.path.append("/app/CosyVoice/third_party/Matcha-TTS")

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/pretrained_models/CosyVoice2-0.5B")

# Lazy-loaded model — initialized on first request so runpod.serverless.start()
# is called immediately and the worker doesn't crash before registering.
_model = None


def _get_model():
    global _model
    if _model is not None:
        return _model

    print(f"[lazy-init] Importing CosyVoice2...", flush=True)
    from cosyvoice.cli.cosyvoice import CosyVoice2
    print(f"[lazy-init] CosyVoice2 imported", flush=True)

    if not os.path.exists(os.path.join(MODEL_DIR, "cosyvoice.yaml")):
        print(f"[lazy-init] Model not found at {MODEL_DIR}, downloading...", flush=True)
        from huggingface_hub import snapshot_download
        snapshot_download("FunAudioLLM/CosyVoice2-0.5B", local_dir=MODEL_DIR)
        print("[lazy-init] Model downloaded.", flush=True)

    print(f"[lazy-init] Loading CosyVoice2 from {MODEL_DIR}...", flush=True)
    _model = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False)
    print(f"[lazy-init] CosyVoice2 ready. sample_rate={_model.sample_rate}", flush=True)
    return _model


def decode_audio(audio_input: str) -> str:
    if audio_input.startswith(("http://", "https://")):
        import requests
        raw = requests.get(audio_input, timeout=30).content
    else:
        raw = base64.b64decode(audio_input)
    waveform, sr = torchaudio.load(io.BytesIO(raw))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    torchaudio.save(tmp.name, waveform, 16000)
    return tmp.name


def handler(job: dict) -> dict:
    inp = job.get("input", {})
    text = inp.get("text", "").strip()
    if not text:
        return {"error": "text is required"}

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
            spk = speaker_id or model.list_available_spks()[0]
            results = list(model.inference_sft(text, spk, stream=False))
        else:
            return {"error": f"Unknown mode: {mode}"}

        if not results:
            return {"error": "No audio generated"}

        combined = torch.cat([r["tts_speech"] for r in results], dim=-1)
        if speed != 1.0 and 0.5 <= speed <= 2.0:
            combined, _ = torchaudio.sox_effects.apply_effects_tensor(
                combined, model.sample_rate, [["tempo", str(speed)]]
            )

        buf = io.BytesIO()
        torchaudio.save(buf, combined, model.sample_rate, format=fmt)

        return {
            "audio_base64": base64.b64encode(buf.getvalue()).decode(),
            "sample_rate": model.sample_rate,
            "format": fmt,
            "duration_ms": int(combined.shape[-1] / model.sample_rate * 1000),
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


print("=== Starting RunPod handler ===", flush=True)
runpod.serverless.start({"handler": handler})
