#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Web UI for transcribe.py: recording -> corrected transcript_clean.txt, no terminal.

    uv run webui/server.py            (or: python3 webui/server.py)
    open http://127.0.0.1:8765

Standard library only. Listens on 127.0.0.1 and nowhere else. transcribe.py is only
ever driven through its own CLI stages, in a subprocess, one GPU job at a time.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis as A  # noqa: E402

REPO = HERE.parent
SCRIPT = REPO / "transcribe.py"
WORK = REPO / "work"
STATIC = HERE / "static"
SETTINGS_PATH = WORK / ".webui-settings.json"

MEDIA_EXT = {".mp4", ".mkv", ".mov", ".flv", ".webm", ".ts", ".m4a", ".mp3", ".wav", ".ogg", ".opus", ".flac", ".aac"}
LABEL_RE = re.compile(r"[\w.-]+")  # same rule as transcribe.py's speaker labels


def videos_dir() -> str:
    """The desktop's Videos folder (XDG), else ~/Videos."""
    if shutil.which("xdg-user-dir"):
        out = subprocess.run(["xdg-user-dir", "VIDEOS"], capture_output=True, text=True).stdout.strip()
        if out and out != str(Path.home()) and Path(out).is_dir():
            return out
    return str(Path.home() / "Videos")


# Everything here is a starting point the user changes in the UI; nothing about a
# particular machine, recording setup or language is assumed. track None = the first
# track with sound; language "" = Whisper detects it.
DEFAULTS = {"videos_dir": None, "names": [], "track": None, "language": "", "model": "large-v3", "llm_url": ""}

# Free VRAM a stage needs, in GiB. README: large-v3 ~6 GB on faster-whisper (NVIDIA),
# ~8 GB on the torch backend (every other GPU); diarization ~2 GB. They never run at once.
VRAM_NEED = {"tiny": 1.0, "base": 1.0, "small": 2.0, "medium": 4.0, "turbo": 4.0}
VRAM_NEED_LARGE = {"nvidia": 6.0, "other": 8.0}
VRAM_NEED_DIARIZE = 2.0
LLM_DEFAULT_URL = "http://127.0.0.1:8080"  # llama-server's default port

CLI: list[str] = []
PORT = 8765
WORK_IGNORED: bool | None = None  # transcripts are private; they must never be committable


def check_work_ignored() -> bool | None:
    """True if git ignores work/, False if a transcript there could be committed, None outside git."""
    if not shutil.which("git"):
        return None
    inside = A.run(["git", "-C", str(REPO), "rev-parse", "--is-inside-work-tree"])
    if inside.returncode != 0:
        return None
    probe = A.run(["git", "-C", str(REPO), "check-ignore", "-q", "work/any/transcript_clean.txt"])
    tracked = A.run(["git", "-C", str(REPO), "ls-files", "work"]).stdout.strip()
    return probe.returncode == 0 and not tracked
LOCK = threading.RLock()


def now_ts() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def read_json(p: Path, default=None):
    try:
        return json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(p: Path, data) -> None:
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    tmp.replace(p)


# ----------------------------------------------------------------------------- settings

def load_settings() -> dict:
    s = dict(DEFAULTS)
    s.update(read_json(SETTINGS_PATH, {}) or {})
    s["videos_dir"] = s["videos_dir"] or videos_dir()
    return s


def save_settings(patch: dict) -> dict:
    s = load_settings()
    for k, v in patch.items():
        if k not in DEFAULTS:
            continue
        if k == "names":
            v = [n.strip() for n in v if n.strip()]
            bad = [n for n in v if not LABEL_RE.fullmatch(n)]
            if bad:
                raise ValueError(f"name {bad[0]!r}: use letters, digits, _ . - only (no spaces)")
        if k == "track":
            v = max(1, int(v)) if v else None
        s[k] = v
    WORK.mkdir(exist_ok=True)
    write_json(SETTINGS_PATH, s)
    return s


# ----------------------------------------------------------------------------- recordings

PROBES: dict[str, tuple[str, dict]] = {}


def probe_cached(p: Path) -> dict:
    sig = A.file_sig(p)
    hit = PROBES.get(str(p))
    if hit and hit[0] == sig:
        return hit[1]
    info = A.probe(p)
    PROBES[str(p)] = (sig, info)
    return info


def valid_id(i: str) -> bool:
    return bool(i) and "/" not in i and "\\" not in i and not i.startswith(".") and i != "regression"


def scan() -> dict[str, dict]:
    """Recordings: media files in the videos folder, plus work dirs whose media lives elsewhere."""
    recs: dict[str, dict] = {}
    vd = Path(load_settings()["videos_dir"]).expanduser()
    if vd.is_dir():
        for p in vd.iterdir():
            if p.is_file() and p.suffix.lower() in MEDIA_EXT and valid_id(p.stem):
                recs.setdefault(p.stem, {"id": p.stem, "media": p})
    if WORK.is_dir():
        for d in WORK.iterdir():
            if d.is_dir() and valid_id(d.name) and d.name not in recs and \
                    ((d / "meta.json").exists() or (d / "audio.wav").exists()):
                src = (read_json(d / "meta.json", {}) or {}).get("source")
                media = Path(src) if src and not src.startswith("http") and Path(src).exists() else None
                recs[d.name] = {"id": d.name, "media": media}
    return recs


def get_rec(rid: str) -> dict:
    rec = scan().get(rid)
    if not rec:
        raise ApiError(404, f"no recording {rid!r}")
    return rec


def wdir(rid: str) -> Path:
    return WORK / rid


def rec_status(rec: dict) -> dict:
    w = wdir(rec["id"])
    meta = read_json(w / "meta.json", {}) or {}
    f = lambda n: (w / n).exists()
    review = read_json(w / "review.json", {}) or {}
    clean = w / "transcript_clean.txt"
    clean_state = None
    if clean.exists():
        written = review.get("written") or {}
        if written.get("sha") != sha_file(clean):
            clean_state = "other"      # hand-made, or edited outside the UI
        elif written.get("rev") != review.get("rev"):
            clean_state = "stale"      # edits since the last export
        else:
            clean_state = "current"
    st = {"audio": f("audio.wav"), "transcript": f("transcript.json"), "diarized": f("diarized.json"),
          "speakers": f("speakers.json"), "review": bool(review), "clean": clean_state}
    if clean_state == "current":
        stage = "done"
    elif st["review"]:
        stage = "reviewing"
    elif st["speakers"] and st["transcript"]:
        stage = "named"
    elif st["diarized"]:
        stage = "diarized"
    elif st["transcript"]:
        stage = "transcribed"
    elif st["audio"]:
        stage = "audio"
    else:
        stage = "new"
    out = {"id": rec["id"], "files": st, "stage": stage, "meta": meta, "media": None}
    m = rec["media"]
    if m and m.exists():
        stt = m.stat()
        info = probe_cached(m)
        out["media"] = {"path": str(m), "name": m.name, "size": stt.st_size, "mtime": stt.st_mtime,
                        "duration": info.get("duration"), "tracks": len(info.get("audio_tracks", [])),
                        "video": bool(info.get("video")),
                        # still being written (a recording in progress)
                        "growing": time.time() - stt.st_mtime < 20}
    out["duration"] = meta.get("duration_s") or (out["media"] or {}).get("duration")
    out["job"] = job_summary(active_job(rec["id"]))
    return out


def sha_file(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


def move_aside(w: Path, names: list[str]) -> str | None:
    """Never delete a stage output: park it in previous/<timestamp>/."""
    dest = w / "previous" / now_ts()
    moved = []
    for n in names:
        if (w / n).exists():
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(w / n), str(dest / n))
            moved.append(n)
    return str(dest.relative_to(w)) if moved else None


REDO_FILES = {
    "audio": ["audio.wav", "transcript.json", "diarized.json", "speakers.json", "transcript.txt",
              "transcript.srt", "transcript.tsv", "review.json"],
    "transcribe": ["transcript.json", "diarized.json", "speakers.json", "transcript.txt", "transcript.srt",
                   "transcript.tsv", "review.json"],
    "diarize": ["diarized.json", "speakers.json", "transcript.txt", "transcript.srt", "transcript.tsv",
                "review.json"],
}


# ----------------------------------------------------------------------------- GPU

_gpu_cache: tuple[float, dict] | None = None


def _read(p: str) -> str:
    try:
        return open(p, "rb").read().decode("utf-8", "replace")
    except OSError:
        return ""


def _ppid(pid: int) -> int | None:
    s = _read(f"/proc/{pid}/stat")
    m = re.match(r"\d+ \(.*\) \S (\d+)", s, re.S)
    return int(m.group(1)) if m else None


def our_pids() -> set[int]:
    with LOCK:
        return {j.proc.pid for j in JOBS.values() if j.proc and j.proc.poll() is None}


def proc_info(pid: int, name: str, ours: set[int]) -> dict:
    cmd = _read(f"/proc/{pid}/cmdline").replace("\0", " ").strip()
    cg = _read(f"/proc/{pid}/cgroup")
    units = [u for u in re.findall(r"/([^/\n]+\.service)", cg) if not u.startswith(("user@", "session-"))]
    unit = units[-1] if units else None
    user_unit = unit is not None and "/user@" in cg
    ctr = re.search(r"docker-([0-9a-f]{12})|/docker/([0-9a-f]{12})", cg)
    container = (ctr.group(1) or ctr.group(2)) if ctr else None
    what = Path((cmd.split() or [name])[0]).name or Path(name).name
    p, mine = pid, False
    for _ in range(6):
        if p in ours:
            mine = True; break
        p = _ppid(p) or 0
        if p <= 1:
            break
    hint = None
    if not mine:
        if unit and user_unit:
            hint = f"systemctl --user stop {unit}"
        elif unit:
            hint = f"sudo systemctl stop {unit}"
        elif container:
            hint = f"docker stop {container}"
    return {"pid": pid, "what": what, "cmd": cmd[:300], "unit": unit, "user_unit": user_unit,
            "container": container, "ours": mine, "stop_hint": hint}


def _amd_card() -> tuple[Path, int, int] | None:
    """(device dir, total, used bytes) of the AMD card with the most VRAM, from sysfs."""
    best = None
    for dev in Path("/sys/class/drm").glob("card[0-9]*/device"):
        try:
            total = int((dev / "mem_info_vram_total").read_text())
            used = int((dev / "mem_info_vram_used").read_text())
        except (OSError, ValueError):
            continue
        if not best or total > best[1]:
            best = (dev, total, used)
    return best


def _drm_vram_by_pid(driver: str) -> dict[int, int]:
    """VRAM per process in bytes, from /proc/<pid>/fdinfo (Linux 5.19+ for amdgpu). A client
    opened through several fds is counted once."""
    out: dict[int, int] = {}
    for pd in Path("/proc").iterdir():
        if not pd.name.isdigit():
            continue
        seen = set()
        try:
            fds = list((pd / "fd").iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                if not os.readlink(fd).startswith("/dev/dri/"):
                    continue
                info = (pd / "fdinfo" / fd.name).read_text()
            except OSError:
                continue
            if f"drm-driver:\t{driver}" not in info:
                continue
            cid = re.search(r"drm-client-id:\s*(\d+)", info)
            mem = re.search(r"drm-memory-vram:\s*(\d+)\s*(KiB|MiB)?", info)
            if not mem or (cid and cid.group(1) in seen):
                continue
            if cid:
                seen.add(cid.group(1))
            out[int(pd.name)] = out.get(int(pd.name), 0) + int(mem.group(1)) * {"KiB": 1024, "MiB": 1 << 20}.get(mem.group(2), 1)
    return out


def _amd_status() -> dict | None:
    card = _amd_card()
    if not card:
        return None
    dev, total, used = card
    ours = our_pids()
    procs = []
    for pid, b in _drm_vram_by_pid("amdgpu").items():
        if b < 1 << 20:
            continue
        info = proc_info(pid, _read(f"/proc/{pid}/comm").strip(), ours)
        info["mb"] = b >> 20
        procs.append(info)
    procs.sort(key=lambda p: -(p["mb"] or 0))
    name = _read(str(dev / "product_name")).strip() or "AMD GPU"
    mb = lambda x: x >> 20
    return {"available": True, "vendor": "amd", "name": name, "total_mb": mb(total), "used_mb": mb(used),
            "free_mb": mb(total - used), "procs": procs,
            "unattributed_mb": max(0, mb(used) - sum(p["mb"] or 0 for p in procs))}


def gpu_status(fresh: bool = False) -> dict:
    global _gpu_cache
    if not fresh and _gpu_cache and time.time() - _gpu_cache[0] < 2:
        return _gpu_cache[1]
    if not shutil.which("nvidia-smi"):
        st = _amd_status() or {"available": False,
                               "reason": "no NVIDIA or AMD GPU memory readings here, so free VRAM can't be checked"}
    else:
        q = A.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
                   "--format=csv,noheader,nounits"])
        if q.returncode != 0 or not q.stdout.strip():
            st = {"available": False, "reason": (q.stderr or q.stdout).strip()[:200] or "nvidia-smi failed"}
        else:
            name, total, used, free = [x.strip() for x in q.stdout.splitlines()[0].split(",")]
            apps = A.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                          "--format=csv,noheader,nounits"])
            ours = our_pids()
            procs = []
            for line in apps.stdout.splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) < 3 or not parts[0].isdigit():
                    continue
                info = proc_info(int(parts[0]), parts[1], ours)
                info["mb"] = int(parts[2]) if parts[2].isdigit() else None
                procs.append(info)
            procs.sort(key=lambda p: -(p["mb"] or 0))
            st = {"available": True, "vendor": "nvidia", "name": name, "total_mb": int(total), "used_mb": int(used),
                  "free_mb": int(free), "procs": procs,
                  "unattributed_mb": max(0, int(used) - sum(p["mb"] or 0 for p in procs))}
    _gpu_cache = (time.time(), st)
    return st


_asr_device: dict | None = None


def asr_device() -> dict:
    """What transcribe.py itself will run on (`devices --json`): device, vendor, backend."""
    global _asr_device
    if _asr_device is None:
        out = subprocess.run(CLI + ["devices", "--json"], cwd=REPO, capture_output=True, text=True)
        try:
            _asr_device = json.loads(out.stdout)
        except ValueError:
            _asr_device = {}
    return _asr_device


def vram_need(model: str) -> float:
    fw = asr_device().get("backend") == "faster-whisper"
    return VRAM_NEED.get(model, VRAM_NEED_LARGE["nvidia" if fw else "other"])


# ----------------------------------------------------------------------------- jobs

class Blocked(Exception):
    pass


class JobFailed(Exception):
    pass


class Job:
    _n = 0

    def __init__(self, rid: str, opts: dict):
        Job._n += 1
        self.id = f"j{Job._n}"
        self.rid = rid
        self.opts = opts
        self.state = "queued"
        self.step = None
        self.lines: list[dict] = []
        self.first_line = 0
        self.progress: float | None = None
        self.error: str | None = None
        self.blocked: dict | None = None
        self.proc: subprocess.Popen | None = None
        self.cancelled = False
        self.created = time.time()
        self.finished: float | None = None
        self.steps = ["audio", "transcribe"] + (["diarize"] if opts.get("diarize", True) else [])
        self.done_steps: list[str] = []

    def log(self, text: str, kind: str = "log") -> None:
        with LOCK:
            self.lines.append({"n": self.first_line + len(self.lines), "t": round(time.time(), 1),
                               "kind": kind, "text": text})
            if len(self.lines) > 3000:
                drop = len(self.lines) - 2500
                self.lines = self.lines[drop:]
                self.first_line += drop


JOBS: dict[str, Job] = collections.OrderedDict()
QUEUE: collections.deque[Job] = collections.deque()
WAKE = threading.Condition(LOCK)


def active_job(rid: str) -> Job | None:
    with LOCK:
        for j in reversed(list(JOBS.values())):
            if j.rid == rid and (j.state in ("queued", "running", "blocked", "failed")
                                 or (j.finished and time.time() - j.finished < 30)):
                return j
    return None


def job_summary(j: Job | None) -> dict | None:
    if not j:
        return None
    last = next((l["text"] for l in reversed(j.lines) if l["kind"] == "step"), None)
    return {"id": j.id, "rid": j.rid, "state": j.state, "step": j.step, "steps": j.steps,
            "done_steps": j.done_steps, "progress": j.progress, "error": j.error, "blocked": j.blocked,
            "last": last, "opts": j.opts,
            "queue_pos": list(QUEUE).index(j) + 1 if j in QUEUE else None}


def run_cli(job: Job, args: list[str]) -> None:
    cmd = CLI + args
    job.log("transcribe.py " + " ".join(args), "cmd")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1, env=env)
    job.proc = proc
    oom = False
    tail: list[str] = []
    assert proc.stdout
    for raw in proc.stdout:
        line = raw.rstrip()
        if not line.strip():
            continue
        m = re.match(r"Progress: ([\d.]+)%", line.strip())
        if m:
            job.progress = float(m.group(1)); continue
        if re.search(r"out of memory|CUDA_ERROR_OUT_OF_MEMORY|CUBLAS_STATUS_ALLOC_FAILED", line, re.I):
            oom = True
        kind = "step" if line.startswith("»") else "error" if line.startswith("ERROR:") else "log"
        job.log(line.lstrip("» ") if kind == "step" else line, kind)
        tail = (tail + [line])[-15:]
    rc = proc.wait()
    job.proc = None
    if job.cancelled:
        raise JobFailed("cancelled")
    if rc != 0:
        err = next((l for l in reversed(tail) if l.startswith("ERROR:")), None)
        if oom and "halv" not in (err or ""):
            g = gpu_status(fresh=True)
            job.blocked = {"reason": "oom", "gpu": g}
            raise JobFailed("The GPU ran out of memory. " + holders_sentence(g))
        raise JobFailed((err or (tail[-1] if tail else f"exit code {rc}")).removeprefix("ERROR: "))


def holders_sentence(g: dict) -> str:
    if not g.get("available"):
        return ""
    others = [p for p in g.get("procs", []) if not p["ours"]]
    if not others:
        return f"{g['free_mb'] / 1024:.1f} GB of {g['total_mb'] / 1024:.0f} GB free; nothing else is listed as using it."
    return "Holding VRAM: " + ", ".join(
        f"{p['what']} (pid {p['pid']}{', ' + p['unit'] if p['unit'] else ''}, {p['mb'] / 1024:.1f} GB)"
        for p in others if p["mb"]) + "."


def gpu_gate(job: Job, what: str, need_gb: float) -> None:
    if job.opts.get("force_gpu"):
        job.log(f"skipping the VRAM check for {what} (started anyway)", "note")
        return
    dev = asr_device()
    if dev.get("device") == "cpu":
        job.log(f"{what} runs on the CPU here, no VRAM needed", "note")
        return
    g = gpu_status(fresh=True)
    if not g.get("available"):
        job.log(f"can't check free VRAM ({g.get('reason')}), going ahead", "note")
        return
    if g["free_mb"] < need_gb * 1024:
        job.blocked = {"reason": "vram", "what": what, "need_gb": need_gb, "gpu": g}
        raise Blocked(f"{what} needs about {need_gb:.0f} GB of free VRAM, "
                      f"only {g['free_mb'] / 1024:.1f} GB is free. " + holders_sentence(g))
    job.log(f"VRAM: {g['free_mb'] / 1024:.1f} GB free, {what} needs ~{need_gb:.0f} GB", "note")


def step_audio(job: Job, rec: dict) -> None:
    w = wdir(rec["id"])
    if (w / "audio.wav").exists():
        job.log("audio.wav exists, keeping it", "note")
        return
    media = rec["media"]
    if not media or not media.exists():
        raise JobFailed("the recording file is gone, so there is nothing to extract audio from")
    info = probe_cached(media)
    tracks = info.get("audio_tracks", [])
    if not tracks:
        raise JobFailed(f"{media.name} has no audio track")
    track = min(max(1, int(job.opts.get("track") or 1)), len(tracks))
    w.mkdir(parents=True, exist_ok=True)
    meta = read_json(w / "meta.json", {}) or {}
    # transcribe.py keeps an existing "source", so the meta points at the real
    # recording and not at the temporary single-track copy
    meta.update(source=str(media), audio_track=track, audio_tracks=len(tracks))
    write_json(w / "meta.json", meta)
    if len(tracks) == 1:
        run_cli(job, ["audio", str(media), "-w", str(w)])
        return
    with tempfile.TemporaryDirectory(prefix=".track-", dir=w) as tmp:
        one = Path(tmp) / f"track{track}.mka"
        job.log(f"taking audio track {track} of {len(tracks)}", "note")
        A.extract_track(media, track, one)
        run_cli(job, ["audio", str(one), "-w", str(w)])


def step_transcribe(job: Job, rec: dict) -> None:
    w = wdir(rec["id"])
    if (w / "transcript.json").exists():
        job.log("transcript.json exists, keeping it", "note")
        return
    model = job.opts.get("model") or "large-v3"
    gpu_gate(job, "Transcription", vram_need(model))
    args = ["transcribe", "-w", str(w), "-m", model]
    lang = (job.opts.get("language") or "").strip()
    if lang and lang != "auto":
        args += ["-l", lang]
    job.progress = 0
    run_cli(job, args)


def step_diarize(job: Job, rec: dict) -> None:
    w = wdir(rec["id"])
    if (w / "diarized.json").exists():
        job.log("diarized.json exists, keeping it", "note")
        return
    gpu_gate(job, "Diarization", VRAM_NEED_DIARIZE)
    job.progress = None
    run_cli(job, ["diarize", "-w", str(w)])


STEPS = {"audio": step_audio, "transcribe": step_transcribe, "diarize": step_diarize}


def run_job(job: Job) -> None:
    rec = scan().get(job.rid)
    try:
        if not rec:
            raise JobFailed("recording not found")
        for name in job.steps:
            if name in job.done_steps:
                continue
            job.step = name
            job.progress = None
            STEPS[name](job, rec)
            job.done_steps.append(name)
            if name == "audio":
                start_analysis(job.rid)
        job.state = "done"
        job.step = None
        job.log("finished", "done")
    except Blocked as e:
        job.state = "blocked"
        job.error = str(e)
        job.log(str(e), "error")
    except JobFailed as e:
        job.state = "cancelled" if job.cancelled else "failed"
        job.error = None if job.cancelled else str(e)
        job.log("cancelled" if job.cancelled else str(e), "error")
    except Exception as e:  # noqa: BLE001 - report anything to the UI rather than dying silently
        job.state = "failed"
        job.error = f"{type(e).__name__}: {e}"
        job.log(traceback.format_exc(), "error")
    finally:
        job.finished = time.time()
        job.proc = None


def worker() -> None:
    while True:
        with WAKE:
            while not QUEUE:
                WAKE.wait()
            job = QUEUE.popleft()
            job.state = "running"
        run_job(job)


def enqueue(job: Job) -> None:
    with WAKE:
        JOBS[job.id] = job
        job.state = "queued"
        job.error = None
        job.blocked = None
        job.finished = None
        QUEUE.append(job)
        WAKE.notify()


# ----------------------------------------------------------------------------- background analysis

ANALYSING: dict[str, str] = {}


def start_analysis(rid: str) -> None:
    """Silence + waveform (seconds) and scene detection (a CPU pass over the video),
    off the request path and alongside the GPU job."""
    with LOCK:
        if rid in ANALYSING:
            return
        ANALYSING[rid] = "audio"

    def go():
        try:
            w = wdir(rid)
            if (w / "audio.wav").exists():
                A.ensure_audio_analysis(w)
            rec = scan().get(rid)
            m = rec and rec["media"]
            if m and m.exists() and probe_cached(m).get("video"):
                ANALYSING[rid] = "video"
                A.ensure_scenes(w, m)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        finally:
            with LOCK:
                ANALYSING.pop(rid, None)
    threading.Thread(target=go, daemon=True).start()


# ----------------------------------------------------------------------------- review + export

DOCS: dict[str, tuple[tuple, dict]] = {}


DOC_LOCKS: collections.defaultdict[str, threading.Lock] = collections.defaultdict(threading.Lock)


def get_doc(rid: str) -> dict:
    with DOC_LOCKS[rid]:
        return _get_doc(rid)


def _get_doc(rid: str) -> dict:
    w = wdir(rid)
    src = w / "diarized.json" if (w / "diarized.json").exists() else w / "transcript.json"
    if not src.exists():
        raise ApiError(409, "there is no transcript yet")
    if not (w / "audio.wav").exists():
        raise ApiError(409, "audio.wav is missing")
    ana = w / "analysis.json"
    key = (str(src), A.file_sig(src), A.file_sig(w / "audio.wav"), A.file_sig(ana) if ana.exists() else "")
    hit = DOCS.get(rid)
    if hit and hit[0] == key:
        return hit[1]
    doc = A.build_doc(w)
    key = (str(src), A.file_sig(src), A.file_sig(w / "audio.wav"), A.file_sig(ana) if ana.exists() else "")
    DOCS[rid] = (key, doc)
    return doc


def speakers_cfg(rid: str) -> dict:
    cfg = read_json(wdir(rid) / "speakers.json", None) or {}
    return {"default": cfg.get("default"), "speakers": cfg.get("speakers") or {}}


def load_review(rid: str) -> dict:
    return read_json(wdir(rid) / "review.json", {}) or {}


def resolve_name(p: dict, edit: dict, cfg: dict) -> str:
    return edit.get("spk") or cfg["speakers"].get(p["spk"] or "") or cfg["default"] or p["spk"] or "?"


def render_clean(doc: dict, review: dict, cfg: dict) -> tuple[str, dict]:
    """transcript_clean.txt: a timestamped block per speaker turn, one paragraph each.
    Line breaks the user typed are kept; dropped pieces are left out."""
    edits = review.get("pieces") or {}
    turns: list[dict] = []
    unnamed = set()
    for p in doc["pieces"]:
        e = edits.get(p["id"]) or {}
        if e.get("drop"):
            continue
        text = e.get("text", p["text"])
        text = "\n".join(" ".join(l.split()) for l in text.split("\n")).strip()
        if not text:
            continue
        name = resolve_name(p, e, cfg)
        if not LABEL_RE.fullmatch(name) or re.fullmatch(r"SPEAKER_\d+|\?", name):
            unnamed.add(name)
        if turns and turns[-1]["name"] == name:
            turns[-1]["parts"].append(text)
        else:
            turns.append({"name": name, "t": p["s"], "parts": [text]})
    blocks = []
    for t in turns:
        body = " ".join(t["parts"])
        body = "\n".join(l.strip() for l in body.split("\n") if l.strip())
        blocks.append(f"[{A.fmt_t(t['t'])}] {t['name']}:\n{body}")
    return ("\n\n".join(blocks) + "\n") if blocks else "", {"turns": len(turns), "unnamed": sorted(unnamed)}


# ----------------------------------------------------------------------------- local LLM (optional)

_llm_cache: tuple[float, dict] | None = None


def llm_status() -> dict:
    """An OpenAI-compatible llama-server that is already running. Never started from here."""
    global _llm_cache
    if _llm_cache and time.time() - _llm_cache[0] < 10:
        return _llm_cache[1]
    urls = [load_settings().get("llm_url") or LLM_DEFAULT_URL]
    st = {"available": False, "tried": urls}
    for u in urls:
        try:
            with urllib.request.urlopen(u.rstrip("/") + "/v1/models", timeout=0.4) as r:
                d = json.load(r)
            ids = [m.get("id") for m in d.get("data", [])]
            st = {"available": True, "url": u.rstrip("/"), "model": ids[0] if ids else None}
            break
        except (OSError, ValueError, urllib.error.URLError):
            continue
    _llm_cache = (time.time(), st)
    return st


LLM_PROMPT = """You proofread speech-to-text transcripts made by Whisper. {language}Whisper \
mishears words, glues words together or splits them, invents non-words and gets punctuation wrong.

Fix only what is clearly transcribed wrong. Rules:
- Keep slang, dialect, colloquial or non-standard forms, swearing and filler words exactly as \
spoken. Do not turn speech into standard written language.
- Keep the word order; do not shorten, add, summarise or polish anything.
- If you are not sure, leave the text as it is.
- Keep markers such as […] where they are.

You get a JSON list of {{"id", "text"}} segments from one speaker, in order. Reply with ONLY a JSON \
list of {{"id", "text"}} for the segments you changed, in the same language. If nothing, reply []."""


def llm_suggest(items: list[dict], context: str, language: str | None) -> list[dict]:
    st = llm_status()
    if not st.get("available"):
        raise ApiError(409, "no local LLM server is running")
    body = {"model": st.get("model") or "local", "temperature": 0.2,
            "messages": [{"role": "system", "content": LLM_PROMPT.format(
                language=f"The language is {language}. " if language else "")},
                         {"role": "user", "content": (f"Context (what was said just before): {context}\n\n" if context else "")
                          + json.dumps(items, ensure_ascii=False)}]}
    req = urllib.request.Request(st["url"] + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.load(r)
    except (OSError, urllib.error.URLError) as e:
        raise ApiError(502, f"the LLM server did not answer: {e}")
    text = d["choices"][0]["message"]["content"]
    m = re.search(r"\[.*\]", text, re.S)
    try:
        out = json.loads(m.group(0)) if m else []
    except ValueError:
        raise ApiError(502, "the LLM answered with something that is not a JSON list")
    known = {i["id"]: i["text"] for i in items}
    return [{"id": o["id"], "text": o["text"]} for o in out
            if isinstance(o, dict) and o.get("id") in known and isinstance(o.get("text"), str)
            and o["text"].strip() and o["text"].strip() != known[o["id"]].strip()]


# ----------------------------------------------------------------------------- HTTP

class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # the browser drops audio range requests all the time while seeking
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


class ApiError(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


FRAMES: collections.OrderedDict[tuple, bytes] = collections.OrderedDict()
TRACK_LEVELS: dict[tuple, list] = {}


def track_levels(m: Path) -> list[dict]:
    """Every audio track with its loudness; tracks are only ever told apart by number and sound."""
    info = probe_cached(m)
    key = (str(m), A.file_sig(m))
    if key not in TRACK_LEVELS:
        n = len(info.get("audio_tracks", []))
        res: list = [None] * n
        ts = [threading.Thread(target=lambda i=i: res.__setitem__(i, A.track_level(m, i + 1))) for i in range(n)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        TRACK_LEVELS[key] = res
    tracks = [{**t, **(lv or {})} for t, lv in zip(info.get("audio_tracks", []), TRACK_LEVELS[key])]
    for t in tracks:
        t["silent"] = t.get("max_db", -120) < -60
        # same mean and peak to 0.1 dB: most likely the same mix
        t["same_as"] = next((o["n"] for o in tracks if o["n"] < t["n"] and not o["silent"]
                             and abs(o["mean_db"] - t["mean_db"]) < 0.15 and abs(o["max_db"] - t["max_db"]) < 0.15),
                            None)
    return tracks


def default_track(tracks: list[dict]) -> int | None:
    """The track the user chose to remember, if this file has it and it has sound; else the first one with sound."""
    if not tracks:
        return None
    pref = load_settings().get("track")
    if pref and pref <= len(tracks) and not tracks[pref - 1]["silent"]:
        return pref
    return next((t["n"] for t in tracks if not t["silent"]), 1)


class Handler(BaseHTTPRequestHandler):
    server_version = "transcribe-webui"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet: only errors
        pass

    # --- plumbing
    def _host_ok(self) -> bool:
        # DNS-rebinding guard: a page on another origin that resolves to 127.0.0.1
        # still sends its own Host header
        return self.headers.get("Host", "") in (f"127.0.0.1:{PORT}", f"localhost:{PORT}")

    def _origin_ok(self) -> bool:
        o = self.headers.get("Origin")
        return o is None or o in (f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}")

    def send_json(self, data, code: int = 200) -> None:
        b = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def send_bytes(self, b: bytes, ctype: str, cache: bool = False) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(b)

    def send_file(self, p: Path, ctype: str) -> None:
        size = p.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with open(p, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 16, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    def body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n and "application/json" not in (self.headers.get("Content-Type") or ""):
            raise ApiError(415, "JSON only")
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        self.dispatch("GET")

    def do_POST(self):
        self.dispatch("POST")

    def do_PUT(self):
        self.dispatch("PUT")

    def dispatch(self, method: str) -> None:
        try:
            if not self._host_ok():
                raise ApiError(403, "wrong Host header")
            if method != "GET":
                if not self._origin_ok():
                    raise ApiError(403, "cross-origin request refused")
                # a JSON content type makes browsers preflight cross-site requests,
                # which this server never approves
                if "application/json" not in (self.headers.get("Content-Type") or ""):
                    raise ApiError(415, "JSON only")
            u = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(u.query).items()}
            parts = [unquote(x) for x in u.path.strip("/").split("/") if x]
            if not parts or parts[0] != "api":
                if method != "GET":
                    raise ApiError(405, "not allowed")
                return self.static(u.path)
            self.api(method, parts[1:], q)
        except ApiError as e:
            self.send_json({"error": str(e)}, e.code)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            try:
                self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)
            except OSError:
                pass

    def static(self, path: str) -> None:
        rel = path.lstrip("/") or "index.html"
        p = (STATIC / rel).resolve()
        if not str(p).startswith(str(STATIC.resolve())) or not p.is_file():
            p = STATIC / "index.html"  # client-side routes
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        data = p.read_bytes()
        if p.name == "index.html":  # bust the browser cache whenever the assets change
            for asset in ("app.js", "app.css"):
                v = int((STATIC / asset).stat().st_mtime)
                data = data.replace(f'"/{asset}"'.encode(), f'"/{asset}?v={v}"'.encode())
        self.send_bytes(data, ctype)

    # --- API
    def api(self, method: str, parts: list[str], q: dict) -> None:
        route = (method, parts[0] if parts else "")
        if route == ("GET", "state"):
            with LOCK:
                jobs = [job_summary(j) for j in JOBS.values()
                        if j.state in ("queued", "running", "blocked")]
            recs = sorted((rec_status(r) for r in scan().values()),
                          key=lambda r: -((r["media"] or {}).get("mtime") or 0))
            return self.send_json({"settings": load_settings(), "recordings": recs, "jobs": jobs,
                                   "work_ignored": WORK_IGNORED,
                                   "gpu": gpu_status(), "llm": llm_status(),
                                   "analysing": dict(ANALYSING)})
        if route == ("GET", "gpu"):
            return self.send_json(gpu_status(fresh=True))
        if route == ("PUT", "settings"):
            try:
                return self.send_json(save_settings(self.body()))
            except ValueError as e:
                raise ApiError(400, str(e))
        if parts[0] == "job" and len(parts) >= 2:
            with LOCK:
                job = JOBS.get(parts[1])
            if not job:
                raise ApiError(404, "no such job")
            if method == "GET":
                since = int(q.get("since", 0))
                with LOCK:
                    lines = [l for l in job.lines if l["n"] >= since]
                return self.send_json({**job_summary(job), "lines": lines})
            if method == "POST" and parts[2:] == ["cancel"]:
                job.cancelled = True
                with LOCK:
                    if job in QUEUE:
                        QUEUE.remove(job); job.state = "cancelled"; job.finished = time.time()
                if job.proc and job.proc.poll() is None:
                    job.proc.terminate()
                if job.state in ("blocked", "failed"):
                    job.state = "cancelled"; job.finished = time.time() - 60
                return self.send_json({"ok": True})
            if method == "POST" and parts[2:] == ["retry"]:
                b = self.body()
                if job.state not in ("blocked", "failed"):
                    raise ApiError(409, "this job is not waiting for a retry")
                job.opts["force_gpu"] = bool(b.get("force_gpu"))
                job.cancelled = False
                enqueue(job)
                return self.send_json(job_summary(job))
        if parts[0] == "rec" and len(parts) >= 2:
            return self.rec_api(method, parts[1], parts[2:], q)
        raise ApiError(404, "unknown endpoint")

    def rec_api(self, method: str, rid: str, rest: list[str], q: dict) -> None:
        rec = get_rec(rid)
        w = wdir(rid)
        what = rest[0] if rest else ""

        if method == "GET" and what == "":
            st = rec_status(rec)
            m = rec["media"]
            if m and m.exists():
                st["probe"] = probe_cached(m)
            if st["files"]["transcript"]:
                cfg = speakers_cfg(rid)
                st["speakers_cfg"] = cfg
            return self.send_json(st)

        if method == "GET" and what == "tracks":
            m = rec["media"]
            if not m or not m.exists():
                raise ApiError(404, "the recording file is gone")
            tracks = track_levels(m)
            return self.send_json({"tracks": tracks, "duration": probe_cached(m).get("duration"),
                                   "default": default_track(tracks)})

        if method == "GET" and what == "preview":
            m = rec["media"]
            if not m or not m.exists():
                raise ApiError(404, "the recording file is gone")
            track, t = int(q.get("track", 1)), max(0.0, float(q.get("t", 0)))
            out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-t", "12", "-i", str(m),
                                  "-map", f"0:a:{track - 1}", "-ac", "1", "-ar", "22050", "-c:a", "pcm_s16le",
                                  "-f", "wav", "-"], capture_output=True)
            if out.returncode != 0:
                raise ApiError(500, out.stderr.decode(errors="replace")[-300:])
            return self.send_bytes(out.stdout, "audio/wav")

        if method == "GET" and what == "audio":
            if not (w / "audio.wav").exists():
                raise ApiError(404, "no audio.wav yet")
            return self.send_file(w / "audio.wav", "audio/wav")

        if method == "GET" and what == "frame":
            m = rec["media"]
            if not m or not m.exists():
                raise ApiError(404, "the recording file is gone")
            t = round(max(0.0, float(q.get("t", 0))), 2)
            key = (str(m), t)
            if key not in FRAMES:
                out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(m), "-frames:v", "1",
                                      "-vf", "scale=640:-2", "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", "4", "-"],
                                     capture_output=True)
                if out.returncode != 0 or not out.stdout:
                    raise ApiError(500, "could not grab a frame")
                FRAMES[key] = out.stdout
                while len(FRAMES) > 300:
                    FRAMES.popitem(last=False)
            return self.send_bytes(FRAMES[key], "image/jpeg", cache=True)

        if method == "POST" and what == "start":
            b = self.body()
            j = active_job(rid)
            if j and j.state in ("queued", "running", "blocked"):
                raise ApiError(409, "this recording already has a job")
            m = rec["media"]
            if m and m.exists() and time.time() - m.stat().st_mtime < 20:
                raise ApiError(409, "this file is still being written; finish the recording first")
            s = load_settings()
            track = b.get("track") or (default_track(track_levels(m)) if m and m.exists() else 1)
            opts = {"track": int(track), "language": b.get("language", s["language"]),
                    "model": b.get("model") or s["model"], "diarize": bool(b.get("diarize", True)),
                    "force_gpu": bool(b.get("force_gpu"))}
            job = Job(rid, opts)
            enqueue(job)
            return self.send_json(job_summary(job))

        if method == "POST" and what == "redo":
            b = self.body()
            frm = b.get("from")
            if frm not in REDO_FILES:
                raise ApiError(400, "redo from audio, transcribe or diarize")
            j = active_job(rid)
            if j and j.state in ("queued", "running"):
                raise ApiError(409, "wait for the running job to finish")
            moved = move_aside(w, REDO_FILES[frm])
            DOCS.pop(rid, None)
            return self.send_json({"moved_to": moved})

        if method == "GET" and what == "doc":
            if (w / "audio.wav").exists():
                a = A.load_analysis(w)
                m = rec["media"]
                if a.get("scenes") is None and m and m.exists() and probe_cached(m).get("video"):
                    start_analysis(rid)
            doc = get_doc(rid)
            return self.send_json({**doc, "speakers_cfg": speakers_cfg(rid), "review": load_review(rid),
                                   "analysing": ANALYSING.get(rid), "has_video": bool(
                                       rec["media"] and rec["media"].exists()
                                       and probe_cached(rec["media"]).get("video"))})

        if method == "POST" and what == "speakers":
            b = self.body()
            names = {k: v.strip() for k, v in (b.get("names") or {}).items() if v and v.strip()}
            default = (b.get("default") or "").strip() or None
            for v in list(names.values()) + ([default] if default else []):
                if not LABEL_RE.fullmatch(v):
                    raise ApiError(400, f"name {v!r}: use letters, digits, _ . - only (no spaces)")
            known = {s["id"] for s in get_doc(rid)["speakers"]}
            args = ["speakers", "set", "-w", str(w), "--clear"] + [f"{k}={v}" for k, v in names.items() if k in known]
            if default:
                args += ["--default", default]
            for a in (args, ["write", "-w", str(w)]):
                out = subprocess.run(CLI + a, cwd=REPO, capture_output=True, text=True)
                if out.returncode != 0:
                    raise ApiError(400, (out.stderr.strip().splitlines() or ["transcribe.py failed"])[-1]
                                   .removeprefix("ERROR: "))
            return self.send_json({"speakers_cfg": speakers_cfg(rid)})

        if method == "PUT" and what == "review":
            b = self.body()
            with LOCK:
                cur = load_review(rid)
                if cur and cur.get("rev", 0) != b.get("prev_rev", 0):
                    raise ApiError(409, "the review was saved from another tab; reload to continue")
                data = b.get("review") or {}
                data = {k: data[k] for k in ("pieces", "pauses", "heard", "fingerprint") if k in data}
                data["rev"] = cur.get("rev", 0) + 1
                data["written"] = cur.get("written")
                data["saved"] = time.time()
                write_json(w / "review.json", data)
            return self.send_json({"rev": data["rev"]})

        if what == "clean":
            doc = get_doc(rid)
            review = load_review(rid)
            text, info = render_clean(doc, review, speakers_cfg(rid))
            path = w / "transcript_clean.txt"
            if method == "GET":
                existing = path.read_text(encoding="utf-8") if path.exists() else None
                return self.send_json({"text": text, **info, "path": str(path), "existing": existing,
                                       "state": rec_status(rec)["files"]["clean"]})
            if method == "POST":
                moved = None
                with LOCK:
                    written = review.get("written") or {}
                    if path.exists() and written.get("sha") != sha_file(path) and path.read_text(encoding="utf-8") != text:
                        moved = move_aside(w, ["transcript_clean.txt"])
                    path.write_text(text, encoding="utf-8")
                    review = load_review(rid)
                    review["written"] = {"rev": review.get("rev", 0), "sha": sha_file(path), "at": time.time()}
                    write_json(w / "review.json", review)
                return self.send_json({"path": str(path), "moved_to": moved, "state": "current"})

        if method == "POST" and what == "suggest":
            b = self.body()
            items = [{"id": str(i["id"]), "text": str(i["text"])} for i in b.get("items", [])][:60]
            lang = (read_json(w / "meta.json", {}) or {}).get("language")
            return self.send_json({"suggestions": llm_suggest(items, str(b.get("context", ""))[:1500], lang)})

        raise ApiError(404, "unknown endpoint")


def main() -> None:
    global CLI, PORT
    ap = argparse.ArgumentParser(description="Web UI for transcribe.py")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--videos", help=f"folder with recordings (default: {DEFAULTS['videos_dir']}, remembered)")
    ap.add_argument("--python", help="run transcribe.py with this interpreter instead of `uv run` "
                                     "(e.g. the venv's python on AMD)")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    PORT = a.port
    if a.python:
        CLI = [a.python, str(SCRIPT)]
    elif shutil.which("uv"):
        CLI = ["uv", "run", "--quiet", str(SCRIPT)]
    else:
        CLI = [sys.executable, str(SCRIPT)]
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"ERROR: {tool} not found – install ffmpeg first")
    if a.videos:
        save_settings({"videos_dir": str(Path(a.videos).expanduser().resolve())})
    global WORK_IGNORED
    WORK_IGNORED = check_work_ignored()
    if WORK_IGNORED is False:
        print("WARNING: git does not ignore work/ (or tracks files in it). Transcripts written there could be "
              "committed. Add `work/` to .gitignore.", file=sys.stderr, flush=True)
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=asr_device, daemon=True).start()  # torch import takes a few seconds
    srv = Server(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"» web UI on {url}  (recordings from {load_settings()['videos_dir']}; Ctrl+C to stop)", flush=True)
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for j in JOBS.values():
            if j.proc and j.proc.poll() is None:
                j.proc.terminate()


if __name__ == "__main__":
    main()
