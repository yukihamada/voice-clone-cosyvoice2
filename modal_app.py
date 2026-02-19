"""
CosyVoice2 Voice Cloning — Modal Serverless GPU App

Deploy:  modal deploy modal_app.py
Test:    modal run modal_app.py
Serve:   modal serve modal_app.py  (hot-reload dev)
"""
import modal

app = modal.App("cosyvoice2-clone")

# Build the image with all dependencies baked in
cosyvoice_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "sox", "libsox-dev", "wget")
    .pip_install(
        "torch==2.4.0",
        "torchaudio==2.4.0",
        "modelscope",
        "fastapi[standard]",
        "requests",
        "conformer",
        "deepspeed",
        "diffusers",
        "grpcio",
        "grpcio-tools",
        "hydra-core",
        "HyperPyYAML",
        "inflect",
        "librosa",
        "lightning",
        "matplotlib",
        "onnxruntime-gpu",
        "openai-whisper",
        "protobuf",
        "pydantic",
        "rich",
        "soundfile",
        "tensorboard",
        "wget",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git /app/CosyVoice",
        "cd /app/CosyVoice && pip install -r requirements.txt || true",
    )
    .run_commands(
        # Download model at build time so it's baked into the image
        "python3 -c \""
        "from modelscope import snapshot_download; "
        "snapshot_download('iic/CosyVoice2-0.5B', local_dir='/app/pretrained_models/CosyVoice2-0.5B'); "
        "print('Model downloaded!')"
        "\"",
    )
)


@app.cls(
    image=cosyvoice_image,
    gpu="A10G",
    timeout=300,
    container_idle_timeout=120,
    allow_concurrent_inputs=4,
)
class CosyVoiceService:
    @modal.enter()
    def load_model(self):
        import sys
        sys.path.append("/app/CosyVoice")
        sys.path.append("/app/CosyVoice/third_party/Matcha-TTS")
        from cosyvoice.cli.cosyvoice import CosyVoice2

        self.model = CosyVoice2(
            "/app/pretrained_models/CosyVoice2-0.5B",
            load_jit=False,
            load_trt=False,
        )
        print(f"CosyVoice2 ready! sample_rate={self.model.sample_rate}")

    @modal.method()
    def clone_voice(
        self,
        text: str,
        prompt_audio_b64: str,
        prompt_text: str = "",
        mode: str = "zero_shot",
        instruct_text: str = "",
        speaker_id: str = "",
        speed: float = 1.0,
        fmt: str = "mp3",
    ) -> dict:
        import base64
        import io
        import os
        import tempfile
        import torch
        import torchaudio

        # Decode reference audio
        if prompt_audio_b64:
            raw = base64.b64decode(prompt_audio_b64)
            waveform, sr = torchaudio.load(io.BytesIO(raw))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            torchaudio.save(tmp.name, waveform, 16000)
            wav_path = tmp.name
        else:
            wav_path = None

        try:
            if mode == "zero_shot":
                if not wav_path:
                    return {"error": "prompt_audio required for zero_shot"}
                results = list(self.model.inference_zero_shot(
                    text, prompt_text, wav_path, stream=False
                ))
            elif mode == "cross_lingual":
                if not wav_path:
                    return {"error": "prompt_audio required for cross_lingual"}
                results = list(self.model.inference_cross_lingual(
                    text, wav_path, stream=False
                ))
            elif mode == "instruct":
                if not wav_path or not instruct_text:
                    return {"error": "prompt_audio and instruct_text required"}
                results = list(self.model.inference_instruct2(
                    text, instruct_text, wav_path, stream=False
                ))
            elif mode == "sft":
                spk = speaker_id or self.model.list_available_spks()[0]
                results = list(self.model.inference_sft(text, spk, stream=False))
            else:
                return {"error": f"Unknown mode: {mode}"}

            if not results:
                return {"error": "No audio generated"}

            combined = torch.cat([r["tts_speech"] for r in results], dim=-1)

            if speed != 1.0 and 0.5 <= speed <= 2.0:
                combined, _ = torchaudio.sox_effects.apply_effects_tensor(
                    combined, self.model.sample_rate, [["tempo", str(speed)]]
                )

            buf = io.BytesIO()
            torchaudio.save(buf, combined, self.model.sample_rate, format=fmt)

            return {
                "audio_base64": base64.b64encode(buf.getvalue()).decode(),
                "sample_rate": self.model.sample_rate,
                "format": fmt,
                "duration_ms": int(combined.shape[-1] / self.model.sample_rate * 1000),
                "speakers": self.model.list_available_spks(),
            }
        except Exception as e:
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}
        finally:
            if wav_path:
                os.unlink(wav_path)

    @modal.method()
    def list_speakers(self) -> list:
        return self.model.list_available_spks()


# --- Web endpoint (FastAPI) ---
@app.function(
    image=cosyvoice_image,
    gpu="A10G",
    timeout=300,
    container_idle_timeout=120,
    allow_concurrent_inputs=4,
)
@modal.web_endpoint(method="POST", docs=True)
def clone(body: dict):
    """HTTP endpoint for voice cloning. POST JSON with text + prompt_audio (base64)."""
    svc = CosyVoiceService()
    return svc.clone_voice.remote(
        text=body.get("text", ""),
        prompt_audio_b64=body.get("prompt_audio", ""),
        prompt_text=body.get("prompt_text", ""),
        mode=body.get("mode", "zero_shot"),
        instruct_text=body.get("instruct_text", ""),
        speaker_id=body.get("speaker_id", ""),
        speed=float(body.get("speed", 1.0)),
        fmt=body.get("format", "mp3"),
    )


@app.function(image=cosyvoice_image, gpu="A10G", timeout=60, container_idle_timeout=120)
@modal.web_endpoint(method="GET", docs=True)
def health():
    """Health check endpoint."""
    svc = CosyVoiceService()
    speakers = svc.list_speakers.remote()
    return {"status": "ok", "speakers": speakers}


# --- Local test ---
@app.local_entrypoint()
def main():
    svc = CosyVoiceService()
    speakers = svc.list_speakers.remote()
    print(f"Available speakers: {speakers}")
    print("Service is running! Use the web endpoint URL for API calls.")
