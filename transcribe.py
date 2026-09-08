#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["whisperx>=3.8", "huggingface_hub>=0.25"]
# ///
"""
transcribe.py – meeting recording -> transcript with speaker labels, fully local.

One-shot (interactive speaker naming at the end):
    uv run transcribe.py run <video|audio|url> -l cs

Step by step (each command is idempotent and prints what it did; good for scripts and AI agents):
    uv run transcribe.py devices                             which GPU and ASR backend will be used
    uv run transcribe.py status      -w work/x               what exists, what is next
    uv run transcribe.py download    <url>        -w work/x  SharePoint share link or direct URL
    uv run transcribe.py audio       <media>      -w work/x  ffmpeg -> mono 16 kHz wav
    uv run transcribe.py transcribe  -w work/x -l cs         WhisperX + word alignment
                                     [--device auto|cuda|rocm|mps|xpu|cpu] [--backend auto|faster-whisper|torch]
    uv run transcribe.py diarize     -w work/x               pyannote speaker labels (needs HF token)
    uv run transcribe.py speakers show -w work/x [--json]    who talked how much, with sample lines
    uv run transcribe.py speakers set  -w work/x SPEAKER_03=teacher SPEAKER_07=teacher --default student
    uv run transcribe.py speakers ask  -w work/x             interactive naming (terminal)
    uv run transcribe.py write       -w work/x [--only teacher]   txt + srt (optionally only some labels)

Work dir layout (fixed names, so any step can be re-run or done by hand):
    meta.json  media.*  audio.wav  transcript.json  diarized.json  speakers.json  transcript.txt  transcript.srt  transcript.tsv
Add --json to status / speakers show for machine-readable output.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

# ----------------------------------------------------------------------------- utils

def log(msg: str) -> None:
    print(f"» {msg}", flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def fmt_ts(t: float, sep: str = ",") -> str:
    h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", sep)


def fmt_min(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


class Work:
    """Fixed file layout inside the work dir."""

    def __init__(self, path: Path):
        self.dir = path
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = path / "meta.json"
        self.wav = path / "audio.wav"
        self.transcript = path / "transcript.json"
        self.diarized = path / "diarized.json"
        self.speakers = path / "speakers.json"
        self.txt = path / "transcript.txt"
        self.srt = path / "transcript.srt"
        self.tsv = path / "transcript.tsv"

    @property
    def media(self) -> Path | None:
        found = sorted(p for p in self.dir.glob("media.*") if p.suffix not in (".cookies", ".part"))
        return found[0] if found else None

    def read_meta(self) -> dict:
        return json.load(open(self.meta, encoding="utf-8")) if self.meta.exists() else {}

    def update_meta(self, **kv) -> None:
        m = self.read_meta(); m.update(kv)
        json.dump(m, open(self.meta, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    def result(self) -> dict:
        """Best available transcript (diarized if present)."""
        for p in (self.diarized, self.transcript):
            if p.exists():
                return json.load(open(p, encoding="utf-8"))
        die(f"no transcript in {self.dir} – run `transcribe` first")


# ----------------------------------------------------------------------------- download

SHAREPOINT_RE = re.compile(r"https://([^/]+)/:[a-z]:/g/personal/([^/]+)/([A-Za-z0-9_-]+)")


def resolve_download_url(url: str) -> str:
    m = SHAREPOINT_RE.match(url)
    if m:
        host, user, share_id = m.groups()
        return f"https://{host}/personal/{user}/_layouts/15/download.aspx?share={share_id}"
    return url


def stage_download(w: Work, url: str) -> Path:
    if w.media:
        log(f"download: already have {w.media.name} ({w.media.stat().st_size / 1e6:.0f} MB), skipping")
        return w.media
    direct = resolve_download_url(url)
    ext = Path(urllib.parse.urlparse(direct).path).suffix if not SHAREPOINT_RE.match(url) else ""
    part = w.dir / "media.part"
    log(f"download: {direct}")
    if not shutil.which("curl"):
        die("curl not found")
    cj = w.dir / "media.cookies"
    out = subprocess.run(["curl", "-sS", "-L", "-c", str(cj), "-b", str(cj), "-o", str(part),
                          "-w", "%{http_code} %{content_type} %{filename_effective}", direct],
                         capture_output=True, text=True)
    cj.unlink(missing_ok=True)
    status = out.stdout.strip()
    if out.returncode != 0 or not status.startswith("200"):
        part.unlink(missing_ok=True)
        die(f"download failed ({status or out.stderr.strip()}). If the link needs a login, "
            "download in a browser and use `audio <file>` instead.")
    if "text/html" in status:
        part.unlink(missing_ok=True)
        die("got an HTML page instead of media – the link needs a login. Download in a browser and use `audio <file>`.")
    if not ext:
        ctype = status.split()[1] if len(status.split()) > 1 else ""
        ext = {"video/mp4": ".mp4", "audio/mpeg": ".mp3", "audio/wav": ".wav", "audio/x-m4a": ".m4a",
               "video/webm": ".webm", "video/x-matroska": ".mkv"}.get(ctype.split(";")[0], ".bin")
    dest = w.dir / f"media{ext}"
    part.rename(dest)
    w.update_meta(source=url)
    log(f"download: saved {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


# ----------------------------------------------------------------------------- audio

def stage_audio(w: Work, src: Path | None) -> Path:
    if w.wav.exists():
        log(f"audio: {w.wav.name} exists, skipping")
        return w.wav
    src = src or w.media
    if src is None or not src.exists():
        die("audio: no media file – pass a path or run `download` first")
    if not shutil.which("ffmpeg"):
        die("ffmpeg not found – install it (pacman -S ffmpeg / apt install ffmpeg / "
            "brew install ffmpeg / winget install Gyan.FFmpeg)")
    log(f"audio: extracting mono 16 kHz wav from {src.name}")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", str(w.wav)], check=True)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                "csv=p=0", str(w.wav)], capture_output=True, text=True).stdout or 0)
    w.update_meta(source=w.read_meta().get("source", str(src)), duration_s=round(dur, 1))
    log(f"audio: {w.wav.name}, {dur / 60:.1f} min")
    return w.wav


# ----------------------------------------------------------------------------- device / backend

# WhisperX transcribes through CTranslate2, which ships CUDA and CPU kernels only.
# The rest of the pipeline – the pyannote VAD, wav2vec2 alignment, pyannote
# diarization – is plain PyTorch and runs on any device torch supports, AMD ROCm
# included. So on a non-CUDA GPU we keep all of that and swap out only the ASR
# step for the transformers implementation of Whisper (`--backend torch`), which
# is torch all the way down. transformers is already a WhisperX dependency.

DEVICE_CHOICES = ["auto", "cuda", "rocm", "mps", "xpu", "cpu"]
BACKEND_CHOICES = ["auto", "faster-whisper", "torch"]
COMPUTE_CHOICES = ["auto", "float16", "bfloat16", "float32", "int8"]

VENDOR_NAME = {"nvidia": "NVIDIA", "amd": "AMD", "apple": "Apple", "intel": "Intel", "cpu": "CPU"}

ROCM_HINT = ("Install a ROCm build of PyTorch, e.g.\n"
             "    pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "
             "'torch[device-gfx1201]' torchaudio\n"
             "  with your GPU's arch instead of gfx1201 (gfx1100 = RX 7900, gfx1030 = RX 6800/6900;\n"
             "  rocminfo prints it). See the AMD section of the README.")


def device_label(vendor: str, name: str) -> str:
    """'AMD Radeon RX 9070 XT', not 'AMD AMD Radeon RX 9070 XT'."""
    v = VENDOR_NAME.get(vendor, vendor)
    return name if name.upper().startswith(v.upper()) else f"{v} {name}"


def import_torch():
    try:
        import torch
    except ImportError:
        die("PyTorch is not installed – run the script with `uv run transcribe.py` "
            "or install torch into the venv you are using")
    return torch


def cuda_api_gpu(torch) -> tuple[str, str] | None:
    """(vendor, name) of the GPU torch exposes through its cuda API, or None.

    A ROCm build reports AMD cards through that same API: device strings stay
    "cuda", torch.version.cuda is None and torch.version.hip holds the ROCm version.
    """
    if not torch.cuda.is_available():
        return None
    vendor = "amd" if getattr(torch.version, "hip", None) else "nvidia"
    return vendor, torch.cuda.get_device_name(0)


def resolve_device(requested: str, quiet: bool = False) -> tuple[str, str, str]:
    """--device -> (torch device, vendor, device name), or exit with a fixable message."""
    torch = import_torch()
    hip = getattr(torch.version, "hip", None)
    gpu = cuda_api_gpu(torch)
    has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    has_xpu = bool(getattr(torch, "xpu", None) and torch.xpu.is_available())

    if requested == "cpu":
        return "cpu", "cpu", "CPU"

    if requested == "auto":
        if gpu:
            return "cuda", gpu[0], gpu[1]
        if has_mps:
            return "mps", "apple", "Apple Silicon GPU"
        if has_xpu:
            return "xpu", "intel", torch.xpu.get_device_name(0)
        if not quiet:  # `devices --json` must keep stdout pure JSON
            log("device: no GPU found, falling back to cpu (slow – try -m medium)")
        return "cpu", "cpu", "CPU"

    if requested in ("cuda", "rocm"):
        if gpu is None:
            if requested == "rocm" and not hip:
                build = (f"a CUDA {torch.version.cuda} build" if getattr(torch.version, "cuda", None)
                         else "CPU-only")
                die(f"--device rocm, but PyTorch {torch.__version__} is {build}.\n  {ROCM_HINT}")
            if hip:
                die(f"PyTorch {torch.__version__} has ROCm {hip} but sees no GPU.\n"
                    "  Check rocm-smi, that your user is in the render and video groups (Linux),\n"
                    "  and if the card is not officially supported try HSA_OVERRIDE_GFX_VERSION\n"
                    "  (11.0.0 for RDNA3, 10.3.0 for RDNA2).")
            die("no GPU available – use --device cpu (slow) or fix the GPU setup")
        vendor, name = gpu
        if requested == "rocm" and vendor != "amd":
            die(f"--device rocm, but this PyTorch is a CUDA build driving {name} – use --device cuda")
        return "cuda", vendor, name

    if requested == "mps":
        if not has_mps:
            die("Metal (mps) not available – needs Apple Silicon and a torch build with MPS")
        return "mps", "apple", "Apple Silicon GPU"

    if not has_xpu:
        die("Intel XPU not available – needs a torch build with XPU support and the oneAPI runtime")
    return "xpu", "intel", torch.xpu.get_device_name(0)


def resolve_backend(requested: str, vendor: str) -> str:
    """Which Whisper implementation to transcribe with."""
    if requested != "auto":
        return requested
    # CTranslate2 has no ROCm/Metal/XPU backend, so everything but NVIDIA and CPU goes through torch.
    return "faster-whisper" if vendor in ("nvidia", "cpu") else "torch"


def resolve_compute_type(requested: str, backend: str, device: str) -> str:
    if requested != "auto":
        return requested
    if device == "cpu":
        return "int8" if backend == "faster-whisper" else "float32"
    return "float16"


def log_device(device: str, vendor: str, name: str, backend: str, compute: str) -> None:
    torch = import_torch()
    if getattr(torch.version, "hip", None):
        runtime = f"ROCm {torch.version.hip}"
    elif getattr(torch.version, "cuda", None):
        runtime = f"CUDA {torch.version.cuda}"
    else:
        runtime = "cpu build"
    mem = ""
    try:
        if device == "cuda":
            free, total = torch.cuda.mem_get_info()
        elif device == "xpu" and hasattr(torch.xpu, "mem_get_info"):
            free, total = torch.xpu.mem_get_info()
        else:
            free = total = None
        if free is not None:
            mem = f", {free / 2**30:.1f} GiB free of {total / 2**30:.1f}"
    except Exception:
        pass
    log(f"device: {device_label(vendor, name)} [torch {torch.__version__} / {runtime}]{mem}")
    log(f"backend: {backend} ({compute})")
    if backend == "faster-whisper" and vendor not in ("nvidia", "cpu"):
        log("backend: CTranslate2 has no non-CUDA GPU backend – this needs a patched build "
            "(e.g. ctranslate2-rocm). Drop --backend to use the torch one.")


@contextlib.contextmanager
def muffled_stderr():
    """MIOpen prints kernel-compile failures straight to fd 2, around the exception.
    The probe below expects those, so keep the wall of C++ errors off the terminal."""
    try:
        fd = sys.stderr.fileno()
        saved = os.dup(fd)
    except (AttributeError, OSError, ValueError):
        yield  # stderr is not a real file (pytest, notebooks) - nothing to muffle
        return
    try:
        with tempfile.TemporaryFile() as sink:
            sys.stderr.flush()
            os.dup2(sink.fileno(), fd)
            yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, fd)
        os.close(saved)


def check_norm_kernels(device: str) -> None:
    """Make sure normalisation layers work, and route around MIOpen if they do not.

    pyannote's SincNet (the VAD and the diarization segmenter) uses
    InstanceNorm1d(affine=True), which torch hands to cuDNN/MIOpen. MIOpen compiles
    that kernel at runtime, and some ROCm builds – the Windows ones as of ROCm 7.13 –
    ship a HIPRTC include path that cannot compile it, so every call dies with
    miopenStatusUnknownError. PyTorch's own kernel is fine and costs nothing
    noticeable here, so probe once and switch the whole process over if needed.
    """
    torch = import_torch()
    cudnn = getattr(torch.backends, "cudnn", None)
    if device != "cuda" or cudnn is None or not cudnn.enabled:
        return
    try:
        norm = torch.nn.InstanceNorm1d(1, affine=True).to(device)
        with muffled_stderr(), torch.no_grad():
            norm(torch.zeros(1, 1, 4096, device=device))
            torch.cuda.synchronize()
    except RuntimeError as e:
        cudnn.enabled = False
        first = str(e).strip().splitlines()[0]
        log(f"device: cuDNN/MIOpen cannot run normalisation layers here ({first}) "
            f"– using PyTorch's own kernels instead")


def is_oom(exc: BaseException) -> bool:
    """CUDA says 'CUDA out of memory', ROCm 'HIP out of memory', MPS and XPU differ again."""
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


def empty_cache(device: str) -> None:
    torch = import_torch()
    mod = {"cuda": torch.cuda, "mps": getattr(torch, "mps", None),
           "xpu": getattr(torch, "xpu", None)}.get(device)
    if mod is not None and hasattr(mod, "empty_cache"):
        mod.empty_cache()


def cmd_devices(as_json: bool) -> None:
    """What torch can see here, and what --device auto would pick."""
    torch = import_torch()
    gpu = cuda_api_gpu(torch)
    device, vendor, name = resolve_device("auto", quiet=True)
    backend = resolve_backend("auto", vendor)
    try:
        import ctranslate2
        ct2 = f"{ctranslate2.__version__}, {ctranslate2.get_cuda_device_count()} CUDA device(s)"
    except Exception as e:
        ct2 = f"unavailable ({type(e).__name__})"
    info = {
        "torch": torch.__version__,
        "torch_build": ("ROCm " + torch.version.hip) if getattr(torch.version, "hip", None)
                       else ("CUDA " + torch.version.cuda) if getattr(torch.version, "cuda", None)
                       else "cpu",
        "gpu": device_label(*gpu) if gpu else None,
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "xpu": bool(getattr(torch, "xpu", None) and torch.xpu.is_available()),
        "ctranslate2": ct2,
        "device": device,
        "vendor": vendor,
        "backend": backend,
    }
    if as_json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return
    for k, v in info.items():
        print(f"  {k}: {v}")
    print(f"\n--device auto picks {device} ({device_label(vendor, name)}) "
          f"with the {backend} backend.")


# ----------------------------------------------------------------------------- preflight

PYANNOTE_MODELS = ["pyannote/speaker-diarization-community-1", "pyannote/segmentation-3.0"]


def check_hf() -> None:
    from huggingface_hub import get_token, auth_check
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    if not get_token():
        die("no HuggingFace token. Diarization needs one:\n"
            "  1. create a Read token at https://huggingface.co/settings/tokens\n"
            "  2. run:  python -c 'from huggingface_hub import login; login()'")
    for repo in PYANNOTE_MODELS:
        try:
            auth_check(repo)
        except GatedRepoError:
            die(f"no access to gated model {repo}.\n"
                f"  open https://huggingface.co/{repo}, click 'Agree and access repository', re-run.")
        except RepositoryNotFoundError:
            die(f"model {repo} not found (renamed?) – see https://huggingface.co/pyannote")
    log("huggingface: token OK, pyannote models accessible")


# ----------------------------------------------------------------------------- transcribe

SAMPLE_RATE = 16000

# WhisperX's VAD defaults; the torch backend reuses them so both backends cut the
# audio into the same speech chunks and produce comparable segments.
VAD_CHUNK_S, VAD_ONSET, VAD_OFFSET = 30, 0.500, 0.363

# pyannote runs the VAD 32 windows at a time, which barely occupies a GPU: on an
# RX 9070 XT that made the VAD the slowest stage of the whole run, above Whisper
# itself. 128 is ~5x faster for byte-identical chunks and peaks under 1 GiB.
VAD_BATCH = 128

# faster-whisper model names -> the equivalent transformers repo, for the ones
# where it is not just openai/whisper-<name>.
HF_WHISPER_REPOS = {
    "large": "openai/whisper-large-v3",
    "turbo": "openai/whisper-large-v3-turbo",
    "distil-large-v2": "distil-whisper/distil-large-v2",
    "distil-large-v3": "distil-whisper/distil-large-v3",
    "distil-medium.en": "distil-whisper/distil-medium.en",
    "distil-small.en": "distil-whisper/distil-small.en",
}


def hf_whisper_repo(model_name: str) -> str:
    if "/" in model_name:  # already a HuggingFace repo id
        return model_name
    return HF_WHISPER_REPOS.get(model_name, f"openai/whisper-{model_name}")


def vad_chunks(audio, device: str) -> list[dict]:
    """Speech regions merged into <=30 s chunks – the same segmentation WhisperX feeds
    to CTranslate2, using WhisperX's own pyannote VAD (pure torch, so it runs anywhere).
    The VAD weights ship inside the whisperx wheel, no HuggingFace token needed."""
    import torch
    from whisperx.vads import Pyannote
    vad = Pyannote(torch.device(device), token=None,
                   vad_onset=VAD_ONSET, vad_offset=VAD_OFFSET, chunk_size=VAD_CHUNK_S)
    segmentation = getattr(vad.vad_pipeline, "_segmentation", None)
    if device != "cpu" and segmentation is not None:
        segmentation.batch_size = VAD_BATCH
    raw = vad({"waveform": Pyannote.preprocess_audio(audio), "sample_rate": SAMPLE_RATE})
    return Pyannote.merge_chunks(raw, VAD_CHUNK_S, onset=VAD_ONSET, offset=VAD_OFFSET)


def detect_language_torch(model, processor, feats) -> str:
    import torch
    try:
        with torch.no_grad():
            ids = model.detect_language(feats)
        text = processor.tokenizer.decode(ids[0], skip_special_tokens=False)
    except Exception:
        with torch.no_grad():
            out = model.generate(feats, max_new_tokens=1)
        text = processor.tokenizer.decode(out[0], skip_special_tokens=False)
    m = re.search(r"<\|([a-z]{2,3})\|>", text)
    if not m:
        die("could not detect the language – pass -l/--language (e.g. -l cs)")
    return m.group(1)


def transcribe_torch(audio, model_name: str, language: str | None, device: str, compute: str,
                     batch_size: int, beam_size: int) -> dict:
    """Whisper via transformers instead of CTranslate2, so it runs on ROCm/Metal/XPU.

    Returns the same {"segments": [{text, start, end}], "language": ...} shape as
    WhisperX's own pipeline, so alignment and diarization are unchanged downstream.
    """
    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    repo = hf_whisper_repo(model_name)
    dtype = getattr(torch, compute, None)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:  # torch.int8 exists but cannot load weights
        die(f"--compute-type {compute} does not work with the torch backend; use float16, bfloat16 or float32")
    english_only = repo.endswith(".en")
    if english_only:
        language = "en"

    # VAD first, then let it go: the two models never need to be resident together.
    log("transcribe: voice activity detection")
    chunks = vad_chunks(audio, device)
    if not chunks:
        die("transcribe: no speech found in the audio")
    empty_cache(device)

    log(f"transcribe: loading {repo} ({compute}) on {device} via transformers")
    processor = AutoProcessor.from_pretrained(repo)
    try:
        model = WhisperForConditionalGeneration.from_pretrained(repo, dtype=dtype)
    except TypeError:  # transformers < 4.56 spells it torch_dtype
        model = WhisperForConditionalGeneration.from_pretrained(repo, torch_dtype=dtype)
    model = model.to(device).eval()

    def features(batch):
        f = processor([audio[int(c["start"] * SAMPLE_RATE):int(c["end"] * SAMPLE_RATE)] for c in batch],
                      sampling_rate=SAMPLE_RATE, return_tensors="pt",
                      return_attention_mask=True, padding="max_length", truncation=True)
        mask = getattr(f, "attention_mask", None)
        return f.input_features.to(device, dtype), (mask.to(device) if mask is not None else None)

    if language is None:
        language = detect_language_torch(model, processor, features(chunks[:1])[0])
        log(f"transcribe: detected language {language}")

    log(f"transcribe: {len(chunks)} speech chunks, batch_size={batch_size}, beam_size={beam_size}")
    segments: list[dict] = []
    i = 0
    while i < len(chunks):
        batch = chunks[i:i + batch_size]
        try:
            feats, mask = features(batch)
            kw = {"num_beams": beam_size}
            if not english_only:
                kw.update(language=language, task="transcribe")
            if mask is not None:
                kw["attention_mask"] = mask
            with torch.no_grad():
                ids = model.generate(feats, **kw)
        except Exception as e:
            if is_oom(e) and batch_size > 1:
                batch_size //= 2
                log(f"transcribe: GPU out of memory, retrying with batch_size={batch_size}")
                empty_cache(device)
                continue
            raise
        for c, text in zip(batch, processor.batch_decode(ids, skip_special_tokens=True)):
            segments.append({"text": text.strip(), "start": round(c["start"], 3),
                             "end": round(c["end"], 3)})
        i += len(batch)
        print(f"Progress: {100 * i / len(chunks):.2f}%...", flush=True)

    del model
    empty_cache(device)
    return {"segments": segments, "language": language}


def transcribe_faster_whisper(audio, model_name: str, language: str | None, device: str, compute: str,
                              batch_size: int, beam_size: int) -> dict:
    import whisperx
    log(f"transcribe: loading {model_name} ({compute}) on {device} via faster-whisper")
    model = whisperx.load_model(model_name, device, compute_type=compute, language=language,
                                asr_options={"beam_size": beam_size, "best_of": beam_size})
    log(f"transcribe: batch_size={batch_size}, beam_size={beam_size}")
    while True:
        try:
            return model.transcribe(audio, batch_size=batch_size, language=language, print_progress=True)
        except Exception as e:
            if is_oom(e) and batch_size > 1:
                batch_size //= 2
                log(f"transcribe: GPU out of memory, retrying with batch_size={batch_size}")
                empty_cache(device)
                continue
            raise


def stage_transcribe(w: Work, model_name: str, language: str | None, device: str, batch_size: int,
                     force: bool = False, backend: str = "auto", compute_type: str = "auto",
                     beam_size: int = 5) -> dict:
    if w.transcript.exists() and not force:
        log(f"transcribe: {w.transcript.name} exists, skipping (use --force to redo)")
        return json.load(open(w.transcript, encoding="utf-8"))
    if not w.wav.exists():
        die("transcribe: audio.wav missing – run `audio` first")
    device, vendor, name = resolve_device(device)
    backend = resolve_backend(backend, vendor)
    compute = resolve_compute_type(compute_type, backend, device)
    log_device(device, vendor, name, backend, compute)
    check_norm_kernels(device)

    import whisperx
    audio = whisperx.load_audio(str(w.wav))
    log(f"transcribe: {len(audio) / SAMPLE_RATE / 60:.0f} min of audio")
    run = transcribe_torch if backend == "torch" else transcribe_faster_whisper
    result = run(audio, model_name, language, device, compute, batch_size, beam_size)

    lang = result.get("language", language)
    log(f"transcribe: aligning words (language={lang})")
    align_model, meta = whisperx.load_align_model(language_code=lang, device=device)
    result = whisperx.align(result["segments"], align_model, meta, audio, device, return_char_alignments=False)
    result["language"] = lang
    json.dump(result, open(w.transcript, "w", encoding="utf-8"), ensure_ascii=False)
    w.update_meta(model=model_name, language=lang, backend=backend, device=f"{vendor}:{device}")
    if w.diarized.exists():
        w.diarized.unlink()
        log("transcribe: removed stale diarized.json")
    words = sum(len(s["text"].split()) for s in result["segments"])
    log(f"transcribe: {len(result['segments'])} segments, {words} words -> {w.transcript.name}")
    return result


# ----------------------------------------------------------------------------- diarize

def stage_diarize(w: Work, device: str, force: bool = False) -> dict:
    if w.diarized.exists() and not force:
        log(f"diarize: {w.diarized.name} exists, skipping (use --force to redo)")
        return json.load(open(w.diarized, encoding="utf-8"))
    if not w.transcript.exists():
        die("diarize: transcript.json missing – run `transcribe` first")
    if not w.wav.exists():
        die("diarize: audio.wav missing – run `audio` first")
    device, vendor, name = resolve_device(device)
    check_norm_kernels(device)
    check_hf()
    import whisperx
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers
    # pyannote is plain PyTorch, so this stage needs no backend switch – it already
    # runs wherever torch runs, AMD included.
    log(f"diarize: running pyannote on {name} (a few minutes)")
    audio = whisperx.load_audio(str(w.wav))
    result = json.load(open(w.transcript, encoding="utf-8"))
    segs = DiarizationPipeline(device=device)(audio)
    result = assign_word_speakers(segs, result)
    json.dump(result, open(w.diarized, "w", encoding="utf-8"), ensure_ascii=False)
    n = len({s.get("speaker") for s in result["segments"]})
    log(f"diarize: {n} speakers -> {w.diarized.name}. Next: `speakers show`")
    return result


# ----------------------------------------------------------------------------- speakers

# Speaker labels end up in file names via `write --only PREFIX`, so keep out
# whitespace and path punctuation - but \w is Unicode-aware, so Czech names like
# Mazgal or Zizka with diacritics are fine.
LABEL_RE = r"[\w.-]+"


def speaker_stats(result: dict, samples: int = 3) -> list[dict]:
    stats: dict[str, dict] = {}
    for seg in result["segments"]:
        spk = seg.get("speaker", "UNKNOWN")
        st = stats.setdefault(spk, {"id": spk, "seconds": 0.0, "first_s": seg["start"], "last_s": seg["end"],
                                    "segments": 0, "_segs": []})
        st["seconds"] += seg["end"] - seg["start"]
        st["last_s"] = seg["end"]
        st["segments"] += 1
        st["_segs"].append(seg)
    out = []
    for st in sorted(stats.values(), key=lambda s: -s["seconds"]):
        longest = sorted(st.pop("_segs"), key=lambda x: -(x["end"] - x["start"]))[:samples]
        st["samples"] = [{"at": fmt_min(x["start"]), "text": x["text"].strip()}
                         for x in sorted(longest, key=lambda x: x["start"])]
        st["seconds"] = round(st["seconds"], 1)
        out.append(st)
    return out


def load_speakers_cfg(w: Work) -> dict:
    return json.load(open(w.speakers, encoding="utf-8")) if w.speakers.exists() else {"default": None, "speakers": {}}


def cmd_speakers_show(w: Work, top: int, as_json: bool, samples: int) -> None:
    result = w.result()
    if not any("speaker" in s for s in result["segments"]):
        die("no speaker labels yet – run `diarize` first")
    stats = speaker_stats(result, samples)
    cfg = load_speakers_cfg(w)
    total = sum(s["seconds"] for s in stats) or 1
    for s in stats:
        s["share"] = round(100 * s["seconds"] / total, 1)
        s["label"] = cfg["speakers"].get(s["id"], cfg.get("default"))
    if as_json:
        print(json.dumps({"speakers": stats[:top], "total_speakers": len(stats), "default": cfg.get("default")},
                         ensure_ascii=False, indent=2))
        return
    print(f"{len(stats)} speakers, showing top {min(top, len(stats))} by talk time. "
          f"default label: {cfg.get('default') or '(none, keeps SPEAKER_XX)'}\n")
    for s in stats[:top]:
        lb = f" -> {s['label']}" if s["label"] else ""
        print(f"{s['id']}{lb}  {s['seconds'] / 60:.1f} min ({s['share']:.0f}%), "
              f"{fmt_min(s['first_s'])}–{fmt_min(s['last_s'])}, {s['segments']} segments")
        for smp in s["samples"]:
            t = smp["text"]
            print(f"   [{smp['at']}] {t[:160]}{'…' if len(t) > 160 else ''}")
        print()
    rest = stats[top:]
    if rest:
        print(f"... {len(rest)} more speakers, {sum(s['seconds'] for s in rest) / 60:.1f} min total")
    print("\nNext: `speakers set SPEAKER_XX=label ... --default label`, then `write`.")


def cmd_speakers_set(w: Work, pairs: list[str], default: str | None, clear: bool) -> None:
    cfg = {"default": None, "speakers": {}} if clear else load_speakers_cfg(w)
    known = {s.get("speaker") for s in w.result()["segments"]}
    for p in pairs:
        if "=" not in p:
            die(f"expected SPEAKER_XX=label, got {p!r}")
        k, v = p.split("=", 1)
        if k not in known:
            die(f"{k} is not a speaker in this transcript (see `speakers show`)")
        if not re.fullmatch(LABEL_RE, v):
            die(f"label {v!r}: use letters, digits, _ . - only (no spaces)")
        cfg["speakers"][k] = v
    if default is not None:
        cfg["default"] = default
    json.dump(cfg, open(w.speakers, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log(f"speakers: saved {w.speakers} ({len(cfg['speakers'])} named, default={cfg['default']!r})")


def cmd_speakers_ask(w: Work, top: int, default_label: str) -> None:
    result = w.result()
    stats = speaker_stats(result)
    total = sum(s["seconds"] for s in stats) or 1
    print("=" * 72)
    print(f" {len(stats)} speakers. Name the top {min(top, len(stats))}; Enter = default, '=' = same as previous.")
    print("=" * 72)
    mapping: dict[str, str] = {}
    prev = None
    for s in stats[:top]:
        print(f"\n{s['id']}  {s['seconds'] / 60:.1f} min ({100 * s['seconds'] / total:.0f}%), "
              f"{fmt_min(s['first_s'])}–{fmt_min(s['last_s'])}")
        for smp in s["samples"]:
            print(f"   [{smp['at']}] {smp['text'][:160]}")
        while True:
            ans = input(f"   label for {s['id']} [{default_label}]: ").strip()
            if ans == "=" and prev:
                ans = prev
            if ans and not re.fullmatch(LABEL_RE, ans):
                print("   use letters, digits, _ . - only (no spaces)")
                continue
            break
        if ans:
            mapping[s["id"]] = ans; prev = ans
    rest = len(stats) - min(top, len(stats))
    default = default_label
    if rest > 0:
        default = input(f"\nlabel for the remaining {rest} speakers [{default_label}]: ").strip() or default_label
    json.dump({"default": default, "speakers": mapping}, open(w.speakers, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log(f"speakers: saved {w.speakers}")


# ----------------------------------------------------------------------------- write

def write_tsv(segments, path: Path, label) -> None:
    """One segment per row: start/end in ms (same as WhisperX's .tsv), speaker label (empty if none), text."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("start\tend\tspeaker\ttext\n")
        for seg in segments:
            text = " ".join(seg["text"].split())
            f.write(f"{round(seg['start'] * 1000)}\t{round(seg['end'] * 1000)}\t{label(seg) or ''}\t{text}\n")


def stage_write(w: Work, only: str | None = None, out_stem: str | None = None) -> tuple[Path, Path, Path]:
    result = w.result()
    cfg = load_speakers_cfg(w)
    has_speakers = any("speaker" in s for s in result["segments"])
    if only and not has_speakers:
        die("--only needs speaker labels – run `diarize` first")

    def label(seg):
        if not has_speakers:
            return None
        spk = seg.get("speaker", "UNKNOWN")
        return cfg["speakers"].get(spk, cfg.get("default") or spk)

    stem = out_stem or ("transcript" if not only else f"transcript-{only}")
    txt, srt, tsv = w.dir / f"{stem}.txt", w.dir / f"{stem}.srt", w.dir / f"{stem}.tsv"
    kept = skipped = words = 0
    rows = []
    with open(txt, "w", encoding="utf-8") as f, open(srt, "w", encoding="utf-8") as g:
        last = object(); gap = None; n = 0
        started = False  # no blank lines before the first block
        for seg in result["segments"]:
            text = seg["text"].strip(); lb = label(seg)
            if only and not (lb or "").startswith(only):
                skipped += 1
                if gap is None: gap = seg["start"]
                continue
            kept += 1; words += len(text.split()); rows.append(seg)
            if gap is not None:
                f.write(("\n\n" if started else "")
                        + f"[… {fmt_ts(gap, '.')[:-4]}–{fmt_ts(seg['start'], '.')[:-4]} omitted]")
                gap = None; last = object(); started = True
            if lb is None:
                f.write(text + "\n")
            else:
                if lb != last:
                    f.write(("\n\n" if started else "") + f"[{lb}] "); last = lb
                f.write(text + " ")
            started = True
            n += 1
            pre = f"[{lb}] " if lb else ""
            g.write(f"{n}\n{fmt_ts(seg['start'])} --> {fmt_ts(seg['end'])}\n{pre}{text}\n\n")
    extra = f", kept {kept} / skipped {skipped} segments" if only else ""
    log(f"write: {txt} ({words} words{extra})")
    write_tsv(rows, tsv, label)
    log(f"write: {srt}")
    log(f"write: {tsv}")
    return txt, srt, tsv


# ----------------------------------------------------------------------------- status

def cmd_status(w: Work, as_json: bool) -> None:
    meta = w.read_meta()
    st = {
        "workdir": str(w.dir),
        "media": w.media.name if w.media else None,
        "audio": w.wav.exists(),
        "transcript": w.transcript.exists(),
        "diarized": w.diarized.exists(),
        "speakers_named": w.speakers.exists(),
        "written": w.txt.exists(),
        "meta": meta,
    }
    if not st["audio"] and not st["media"]:
        nxt = "download <url>  or  audio <media-file>"
    elif not st["audio"]:
        nxt = "audio"
    elif not st["transcript"]:
        nxt = "transcribe"
    elif not st["diarized"]:
        nxt = "diarize  (or `write` for a transcript without speakers)"
    elif not st["speakers_named"]:
        nxt = "speakers show, then speakers set ... --default ..."
    else:
        nxt = "write  (done if transcript.txt is current)"
    st["next"] = nxt
    if as_json:
        print(json.dumps(st, ensure_ascii=False, indent=2))
        return
    for k in ("media", "audio", "transcript", "diarized", "speakers_named", "written"):
        v = st[k]
        mark = "✓" if v else "·"
        print(f"  {mark} {k}" + (f": {v}" if isinstance(v, str) else ""))
    if meta:
        print("  meta: " + ", ".join(f"{k}={v}" for k, v in meta.items()))
    print(f"\nnext: {nxt}")


# ----------------------------------------------------------------------------- run (one-shot)

def cmd_run(w: Work, inp: str, a) -> None:
    is_url = inp.startswith(("http://", "https://"))
    if is_url:
        stage_download(w, inp)
        stage_audio(w, None)
    else:
        p = Path(inp)
        if not p.exists():
            die(f"input not found: {p}")
        stage_audio(w, p)
    if not a.no_diarize:
        check_hf()  # fail early, before the long transcription
    stage_transcribe(w, a.model, a.language, a.device, a.batch_size,
                     backend=a.backend, compute_type=a.compute_type, beam_size=a.beam_size)
    if not a.no_diarize:
        stage_diarize(w, a.device)
        if w.speakers.exists():
            log(f"speakers: using existing {w.speakers.name}")
        elif sys.stdin.isatty() and not a.yes:
            cmd_speakers_ask(w, a.top, a.default_label)
        else:
            log("speakers: non-interactive, keeping SPEAKER_XX (use `speakers show/set` later)")
    stage_write(w)


# ----------------------------------------------------------------------------- cli

def default_workdir(inp: str | None) -> Path:
    if not inp:
        die("-w/--workdir is required for this command")
    if inp.startswith(("http://", "https://")):
        stem = Path(urllib.parse.urlparse(inp).path).stem if not SHAREPOINT_RE.match(inp) else "recording"
    else:
        stem = Path(inp).stem
    return Path("work") / stem


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_work(p, required=False):
        p.add_argument("-w", "--workdir", required=required, help="work dir (default: ./work/<input stem>)")

    def add_gpu(p):
        p.add_argument("--device", default="auto", choices=DEVICE_CHOICES,
                       help="auto-detects NVIDIA/AMD/Apple/Intel GPUs; rocm forces an AMD card")

    def add_asr(p):
        p.add_argument("-m", "--model", default="large-v3")
        p.add_argument("-l", "--language", default=None, help="e.g. cs, en; default auto-detect")
        p.add_argument("-b", "--batch-size", type=int, default=4)
        p.add_argument("--backend", default="auto", choices=BACKEND_CHOICES,
                       help="faster-whisper needs CUDA or CPU; torch runs on any GPU torch supports "
                            "(default: torch on non-NVIDIA GPUs)")
        p.add_argument("--compute-type", default="auto", choices=COMPUTE_CHOICES,
                       help="float16 on GPU by default; try float32 if an AMD card gives empty output")
        p.add_argument("--beam-size", type=int, default=5)

    p = sub.add_parser("run", help="all stages in one go (interactive speaker naming)")
    p.add_argument("input"); add_work(p); add_gpu(p); add_asr(p)
    p.add_argument("--no-diarize", action="store_true")
    p.add_argument("--default-label", default="speaker")
    p.add_argument("--top", type=int, default=6)
    p.add_argument("-y", "--yes", action="store_true", help="never prompt")

    p = sub.add_parser("devices", help="which GPU and ASR backend this machine will use")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("status", help="what exists in the work dir and what to do next")
    add_work(p, True); p.add_argument("--json", action="store_true")

    p = sub.add_parser("download", help="fetch media from a URL into the work dir")
    p.add_argument("url"); add_work(p)

    p = sub.add_parser("audio", help="extract audio.wav from a media file (or the downloaded one)")
    p.add_argument("media", nargs="?"); add_work(p)

    p = sub.add_parser("transcribe", help="WhisperX transcription + alignment")
    add_work(p, True); add_gpu(p); add_asr(p)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("diarize", help="pyannote speaker labels")
    add_work(p, True); add_gpu(p); p.add_argument("--force", action="store_true")

    p = sub.add_parser("speakers", help="inspect / name speakers")
    ssub = p.add_subparsers(dest="scmd", required=True)
    q = ssub.add_parser("show"); add_work(q, True)
    q.add_argument("--top", type=int, default=8); q.add_argument("--samples", type=int, default=3)
    q.add_argument("--json", action="store_true")
    q = ssub.add_parser("set"); add_work(q, True)
    q.add_argument("pairs", nargs="*", metavar="SPEAKER_XX=label")
    q.add_argument("--default", help="label for all un-named speakers")
    q.add_argument("--clear", action="store_true", help="start from an empty mapping")
    q = ssub.add_parser("ask"); add_work(q, True)
    q.add_argument("--top", type=int, default=6); q.add_argument("--default-label", default="speaker")

    p = sub.add_parser("write", help="produce transcript.txt / .srt")
    add_work(p, True)
    p.add_argument("--only", metavar="PREFIX", help="keep only speakers whose label starts with PREFIX")
    p.add_argument("--name", help="output file stem (default transcript / transcript-PREFIX)")

    a = ap.parse_args(argv)

    if a.cmd == "run":
        cmd_run(Work(Path(a.workdir) if a.workdir else default_workdir(a.input)), a.input, a)
    elif a.cmd == "devices":
        cmd_devices(a.json)
    elif a.cmd == "status":
        cmd_status(Work(Path(a.workdir)), a.json)
    elif a.cmd == "download":
        w = Work(Path(a.workdir) if a.workdir else default_workdir(a.url))
        stage_download(w, a.url); log(f"next: audio -w {w.dir}")
    elif a.cmd == "audio":
        w = Work(Path(a.workdir) if a.workdir else default_workdir(a.media))
        stage_audio(w, Path(a.media) if a.media else None); log(f"next: transcribe -w {w.dir}")
    elif a.cmd == "transcribe":
        stage_transcribe(Work(Path(a.workdir)), a.model, a.language, a.device, a.batch_size, a.force,
                         backend=a.backend, compute_type=a.compute_type, beam_size=a.beam_size)
        log("next: diarize (speaker labels) or write (plain transcript)")
    elif a.cmd == "diarize":
        stage_diarize(Work(Path(a.workdir)), a.device, a.force)
    elif a.cmd == "speakers":
        w = Work(Path(a.workdir))
        if a.scmd == "show":
            cmd_speakers_show(w, a.top, a.json, a.samples)
        elif a.scmd == "set":
            cmd_speakers_set(w, a.pairs, a.default, a.clear); log("next: write")
        else:
            cmd_speakers_ask(w, a.top, a.default_label); log("next: write")
    elif a.cmd == "write":
        stage_write(Work(Path(a.workdir)), a.only, a.name)


if __name__ == "__main__":
    main()
