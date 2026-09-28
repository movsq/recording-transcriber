#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["huggingface_hub>=0.25", "numpy", "pyarrow"]
# ///
"""
tests/regression.py – end-to-end regression suite for transcribe.py.

Builds a 9-minute English recording with a known transcript (the 73 LibriSpeech
clips of hf-internal-testing/librispeech_asr_dummy, 0.7 s apart), runs it
through the transcription configs below, and checks:

  - WER against the reference stays under a per-model ceiling, the language is
    detected or kept as en, every word has a timestamp, segments are in order
  - faster-whisper output is byte-identical to the same run at --baseline
    (default: main) – that path is not supposed to move when the torch one does
  - -b 512 on the torch backend runs out of GPU memory, halves down and ends up
    with the same transcript.json as a plain -b 4 run
  - the CLI contracts: `devices --json` stays pure JSON without a GPU, a bad
    --device or --compute-type exits with a one-line ERROR, and speaker labels
    with diacritics go through `speakers set` and `write`

    uv run tests/regression.py                         everything this machine can run
    uv run tests/regression.py --baseline v1           compare against another git ref
    uv run tests/regression.py --only fw-tiny-cpu cli  a subset (a few minutes, no GPU needed)

On AMD there is no uv environment; from the activated venv:
    pip install pyarrow && python tests/regression.py

GPU configs are skipped without a GPU, and fw-large on anything but NVIDIA.
Everything lands in work/regression/ (logs included). Exit code 1 if any check fails.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "transcribe.py"

DATASET = "hf-internal-testing/librispeech_asr_dummy"
PARQUET = "clean/validation-00000-of-00001.parquet"
SR, GAP_S = 16000, 0.7

# On an RTX 3060 (torch 2.8 cu128, whisperx 3.8.6): large-v3 3.83 % on faster-whisper
# and 3.91 % on torch, tiny 10.78 % and 10.17 %. The ceilings leave room for other
# GPUs and library versions, but not for a lost segment (~15 words, +1.3 %).
WER_CEILING = {"large-v3": 5.0, "tiny": 13.0}


@dataclass
class Config:
    name: str
    args: list[str]
    model: str
    backend: str
    needs: str               # "nvidia", "gpu" or "any"
    baseline: bool = False   # also run at --baseline and require byte-identical output


CONFIGS = [
    Config("fw-large", ["-l", "en", "-b", "4"], "large-v3", "faster-whisper", "nvidia", baseline=True),
    Config("torch-large", ["-l", "en", "-b", "4", "--backend", "torch"], "large-v3", "torch", "gpu"),
    Config("torch-large-oom", ["-l", "en", "-b", "512", "--backend", "torch"], "large-v3", "torch", "gpu"),
    Config("fw-tiny-cpu", ["-l", "en", "-m", "tiny", "--device", "cpu"], "tiny", "faster-whisper", "any",
           baseline=True),
    # no -l: covers language detection on the torch backend
    Config("torch-tiny-cpu", ["-m", "tiny", "--device", "cpu", "--backend", "torch"], "tiny", "torch", "any"),
]


@dataclass
class Run:
    cfg: Config
    rev: str
    dir: Path
    code: int = 0
    seconds: float = 0.0
    log: str = ""
    segments: int | None = None
    words: int | None = None
    wer: float | None = None
    extra: list[str] = field(default_factory=list)

    @property
    def transcript(self) -> Path:
        return self.dir / "transcript.json"


# ----------------------------------------------------------------------------- helpers

def log(msg: str) -> None:
    print(f"» {msg}", flush=True)


def norm(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


def wer(hyp: list[str], ref: list[str]) -> float:
    d = list(range(len(ref) + 1))
    for i in range(1, len(hyp) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(ref) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (hyp[i - 1] != ref[j - 1]))
    return 100 * d[len(ref)] / len(ref)


def runner() -> list[str]:
    """How to start transcribe.py: this interpreter if it has WhisperX (an AMD venv), else uv."""
    if importlib.util.find_spec("whisperx"):
        return [sys.executable]
    if not shutil.which("uv"):
        sys.exit("ERROR: WhisperX is not importable here and uv is not on PATH")
    return ["uv", "run"]


def tail(text: str, n: int = 15) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("Progress: ")]
    return "\n".join("      " + ln for ln in lines[-n:])


def build_sample(d: Path) -> tuple[Path, list[str]]:
    wav, ref = d / "sample.wav", d / "reference.txt"
    if not (wav.exists() and ref.exists()):
        import numpy as np
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        log(f"sample: building from {DATASET}")
        rows = pq.read_table(hf_hub_download(DATASET, PARQUET, repo_type="dataset"),
                             columns=["audio", "text"]).to_pylist()
        gap = np.zeros(int(GAP_S * SR), dtype=np.float32)
        parts = []
        for row in rows:
            pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1",
                                  "-ar", str(SR), "pipe:1"], input=row["audio"]["bytes"],
                                 capture_output=True, check=True).stdout
            parts += [np.frombuffer(pcm, dtype=np.float32), gap]
        audio = (np.clip(np.concatenate(parts), -1, 1) * 32767).astype(np.int16)
        d.mkdir(parents=True, exist_ok=True)
        with wave.open(str(wav), "wb") as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR)
            f.writeframes(audio.tobytes())
        ref.write_text("\n".join(r["text"] for r in rows) + "\n", encoding="utf-8")
    words = norm(ref.read_text(encoding="utf-8"))
    with wave.open(str(wav)) as f:
        minutes = f.getnframes() / SR / 60
    log(f"sample: {wav} ({minutes:.1f} min, {len(words)} reference words)")
    return wav, words


def diff_summary(a: Path, b: Path) -> str:
    sa, sb = (json.loads(p.read_text(encoding="utf-8"))["segments"] for p in (a, b))
    if len(sa) != len(sb):
        return f"{len(sa)} vs {len(sb)} segments"
    texts = [(x["text"], y["text"]) for x, y in zip(sa, sb) if x["text"] != y["text"]]
    if texts:
        return f"{len(texts)} segments differ in text, first: {texts[0][0].strip()!r} vs {texts[0][1].strip()!r}"
    times = sum((x["start"], x["end"]) != (y["start"], y["end"]) for x, y in zip(sa, sb))
    return f"same text, {times} segments with different times" if times else "same segments, word-level fields differ"


# ----------------------------------------------------------------------------- suite

class Suite:
    def __init__(self, work: Path, baseline: str | None):
        self.work = work
        self.baseline = baseline
        self.run_with = runner()
        self.failures: list[str] = []
        self.skipped: list[str] = []
        self.runs: list[Run] = []

    def check(self, ok: bool, what: str) -> bool:
        print(f"    {'✓' if ok else '✗'} {what}", flush=True)
        if not ok:
            self.failures.append(what)
        return ok

    def cli(self, *args: str, script: Path = SCRIPT, env: dict | None = None) -> subprocess.CompletedProcess:
        return subprocess.run([*self.run_with, str(script), *args], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", env=env)

    # --- machine

    def devices(self) -> dict:
        p = self.cli("devices", "--json")
        try:
            info = json.loads(p.stdout)
        except json.JSONDecodeError:
            self.check(False, f"`devices --json` prints JSON (exit {p.returncode})\n{tail(p.stdout + p.stderr)}")
            return {"gpu": None, "vendor": "cpu", "device": "cpu", "mps": False, "xpu": False}
        log(f"machine: {info.get('gpu') or 'no GPU'}, torch {info['torch']} ({info['torch_build']}), "
            f"--device auto -> {info['device']} / {info['backend']}")
        return info

    def runnable(self, cfg: Config, info: dict) -> bool:
        has_gpu = bool(info.get("gpu") or info.get("mps") or info.get("xpu"))
        if cfg.needs == "nvidia" and info.get("vendor") != "nvidia":
            self.skipped.append(f"{cfg.name} (needs an NVIDIA GPU)")
            return False
        if cfg.needs == "gpu" and not has_gpu:
            self.skipped.append(f"{cfg.name} (needs a GPU)")
            return False
        return True

    # --- transcription runs

    def transcribe(self, cfg: Config, rev: str, script: Path, sample: Path, ref: list[str]) -> Run:
        run = Run(cfg, rev, self.work / ("checkout" if script == SCRIPT else "baseline") / cfg.name)
        run.dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sample, run.dir / "audio.wav")
        print(f"\n  {cfg.name} @ {rev}: transcribe {' '.join(cfg.args)}", flush=True)
        t0 = time.monotonic()
        p = subprocess.run([*self.run_with, str(script), "transcribe", "-w", str(run.dir), "--force", *cfg.args],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           encoding="utf-8", errors="replace")
        run.seconds, run.code, run.log = time.monotonic() - t0, p.returncode, p.stdout
        (run.dir / "log.txt").write_text(run.log, encoding="utf-8")
        self.runs.append(run)
        if not self.check(run.code == 0 and run.transcript.exists(), f"{cfg.name} @ {rev} exits 0 "
                          f"({run.seconds:.0f} s, log in {run.dir / 'log.txt'})"):
            print(tail(run.log))
            return run

        result = json.loads(run.transcript.read_text(encoding="utf-8"))
        segs = result["segments"]
        run.segments, run.wer = len(segs), wer(norm(" ".join(s["text"] for s in segs)), ref)
        run.words = sum(len(s["text"].split()) for s in segs)  # counted the way transcribe.py logs it
        words = [w for s in segs for w in s.get("words", [])]
        untimed = sum("start" not in w for w in words)
        bad = (sum(b["start"] < a["end"] - 0.01 for a, b in zip(segs, segs[1:]))
               + sum(s["end"] <= s["start"] for s in segs))
        ceiling = WER_CEILING[cfg.model]
        self.check(run.wer <= ceiling, f"{cfg.name} @ {rev}: WER {run.wer:.2f} % ≤ {ceiling} % "
                                       f"({run.segments} segments, {run.words} words)")
        self.check(result.get("language") == "en", f"{cfg.name} @ {rev}: language en (got {result.get('language')!r})")
        self.check(bool(words) and untimed == 0, f"{cfg.name} @ {rev}: every word has a timestamp "
                                                 f"({len(words) - untimed}/{len(words)})")
        self.check(bad == 0, f"{cfg.name} @ {rev}: no zero-length or out-of-order segments ({bad} found)")
        meta = json.loads((run.dir / "meta.json").read_text(encoding="utf-8"))
        if "backend" in meta:  # refs before the torch backend do not record it
            self.check(meta["backend"] == cfg.backend, f"{cfg.name} @ {rev}: ran on {cfg.backend} "
                                                       f"(meta says {meta['backend']})")
        return run

    def check_identical(self, a: Run, b: Run, what: str) -> None:
        if a.code or b.code or not (a.transcript.exists() and b.transcript.exists()):
            return  # already failed
        same = a.transcript.read_bytes() == b.transcript.read_bytes()
        self.check(same, f"{what}: transcript.json byte-identical"
                         + ("" if same else f" – {diff_summary(a.transcript, b.transcript)}"))
        if same:
            a.extra.append(f"identical to {b.rev if b.cfg is a.cfg else b.cfg.name}")

    def check_oom(self, oom: Run, plain: Run | None) -> None:
        if oom.code:
            return
        sizes = [int(n) for n in re.findall(r"GPU out of memory, retrying with batch_size=(\d+)", oom.log)]
        if not self.check(bool(sizes), f"{oom.cfg.name}: ran out of GPU memory and retried"):
            return
        oom.extra.append(f"OOM -> b={sizes[-1]}")
        print(f"    · halved {len(sizes)}x: 512 -> {' -> '.join(map(str, sizes))}")
        if plain is None:
            print(f"    · {plain_name(oom)} not run, skipping the comparison")
        elif sizes[-1] != 4:  # a bigger card settles above 4; other batch shapes may round differently
            print(f"    · settled at batch_size={sizes[-1]}, not 4 – not comparable byte for byte")
        else:
            self.check_identical(oom, plain, f"{oom.cfg.name} vs {plain.cfg.name}")

    # --- cheap CLI contracts, no model loading

    def cli_checks(self, info: dict, sample: Path, transcript: Path | None) -> None:
        print("\n  cli", flush=True)
        hidden = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}
        p = self.cli("devices", "--json", env=hidden)
        try:
            dev = json.loads(p.stdout)["device"]
            ok = dev == "cpu" or info.get("vendor") != "nvidia"
        except (json.JSONDecodeError, KeyError):
            dev, ok = None, False
        self.check(ok, f"`devices --json` with GPUs hidden: pure JSON on stdout, device cpu (got {dev!r})")

        guard = self.work / "cli" / "guard"
        guard.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sample, guard / "audio.wav")

        def refused(args: list[str], what: str) -> None:
            p = self.cli("transcribe", "-w", str(guard), "--force", "-l", "en", *args)
            lines = p.stderr.strip().splitlines()
            self.check(p.returncode == 1 and bool(lines) and lines[-1].startswith("ERROR:")
                       and "Traceback" not in p.stderr,
                       f"{what}: exit 1 with one-line ERROR ({lines[-1] if lines else f'exit {p.returncode}'})")

        refused(["-m", "tiny", "--device", "cpu", "--backend", "torch", "--compute-type", "int8"],
                "--compute-type int8 --backend torch")
        for device, present in (("rocm", info.get("vendor") == "amd"), ("mps", info.get("mps")),
                                ("xpu", info.get("xpu"))):
            if not present:
                refused(["--device", device], f"--device {device} on a machine without one")

        if transcript is None:
            self.skipped.append("cli speakers/write (needs a successful transcription run)")
            return
        spk = self.work / "cli" / "speakers"
        spk.mkdir(parents=True, exist_ok=True)
        result = json.loads(transcript.read_text(encoding="utf-8"))
        for i, s in enumerate(result["segments"]):  # stand-in for diarize: two alternating speakers
            s["speaker"] = f"SPEAKER_0{i % 2}"
            for w in s.get("words", []):
                w["speaker"] = s["speaker"]
        (spk / "transcript.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        (spk / "diarized.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        n = len(result["segments"])

        p = self.cli("speakers", "set", "-w", str(spk), "--clear", "SPEAKER_00=Žižka", "--default", "Mazgál")
        self.check(p.returncode == 0, f"speakers set SPEAKER_00=Žižka --default Mazgál (exit {p.returncode})")
        for bad in ("Jan Novák", "../x"):
            p = self.cli("speakers", "set", "-w", str(spk), f"SPEAKER_00={bad}")
            self.check(p.returncode == 1 and "ERROR:" in p.stderr, f"speakers set rejects {bad!r} (exit {p.returncode})")
        p = self.cli("write", "-w", str(spk))
        if self.check(p.returncode == 0, f"write (exit {p.returncode})"):
            try:
                txt, srt, tsv = ((spk / f"transcript.{e}").read_text(encoding="utf-8") for e in ("txt", "srt", "tsv"))
                utf8 = True
            except UnicodeDecodeError:
                txt = srt = tsv = ""; utf8 = False
            self.check(utf8, "write: .txt, .srt and .tsv are UTF-8")
            self.check(txt.startswith("[Žižka] ") and txt.endswith("\n") and "[Mazgál] " in txt,
                       f"write: transcript.txt starts at '[Žižka] ', ends with a newline (starts {txt[:12]!r})")
            rows = tsv.splitlines()
            self.check(rows[:1] == ["start\tend\tspeaker\ttext"] and len(rows) == n + 1,
                       f"write: transcript.tsv has the header and {n} rows ({len(rows) - 1})")
            self.check(srt.startswith("1\n") and "[Žižka] " in srt, "write: transcript.srt numbered, with labels")
        p = self.cli("write", "-w", str(spk), "--only", "Ž")
        only = spk / "transcript-Ž.txt"
        body = only.read_text(encoding="utf-8") if only.exists() else ""
        self.check(p.returncode == 0 and "[Žižka] " in body and "[Mazgál]" not in body and "omitted]" in body,
                   "write --only Ž keeps Žižka, marks the rest omitted")


def plain_name(oom: Run) -> str:
    return oom.cfg.name.removesuffix("-oom")


def baseline_script(work: Path, ref: str) -> Path | None:
    """transcribe.py as of `ref`, or None when it matches the checkout (nothing to compare)."""
    p = subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:transcribe.py"], capture_output=True)
    if p.returncode:
        sys.exit(f"ERROR: cannot read transcribe.py at {ref!r}: {p.stderr.decode().strip()}")
    if p.stdout == SCRIPT.read_bytes():
        return None
    dest = work / "baseline" / "transcribe.py"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(p.stdout)
    return dest


def main() -> None:
    names = [c.name for c in CONFIGS] + ["cli"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default="main", metavar="REF",
                    help="git ref whose faster-whisper output must match byte for byte (default: main)")
    ap.add_argument("--no-baseline", action="store_true", help="skip the baseline runs")
    ap.add_argument("--only", nargs="+", choices=names, metavar="NAME",
                    help=f"run only these: {', '.join(names)}")
    ap.add_argument("--workdir", type=Path, default=REPO / "work" / "regression")
    a = ap.parse_args()
    selected = set(a.only or names)

    suite = Suite(a.workdir, None if a.no_baseline else a.baseline)
    info = suite.devices()
    sample, ref = build_sample(a.workdir)

    base = None
    if suite.baseline and any(c.baseline and c.name in selected for c in CONFIGS):
        base = baseline_script(a.workdir, suite.baseline)
        if base is None:
            suite.skipped.append(f"baseline runs (transcribe.py at {suite.baseline} is the same as the checkout; "
                                 "pass --baseline <ref>)")

    done: dict[str, Run] = {}
    for cfg in CONFIGS:
        if cfg.name not in selected or not suite.runnable(cfg, info):
            continue
        run = done[cfg.name] = suite.transcribe(cfg, "checkout", SCRIPT, sample, ref)
        if cfg.baseline and base:
            old = suite.transcribe(cfg, suite.baseline, base, sample, ref)
            suite.check_identical(run, old, f"{cfg.name} checkout vs {suite.baseline}")
        if cfg.name.endswith("-oom"):
            suite.check_oom(run, done.get(plain_name(run)))

    if "cli" in selected:
        # any transcript will do for the speaker checks, one from an earlier run included
        found = ([r.transcript for r in done.values() if not r.code]
                 + sorted((a.workdir / "checkout").glob("*/transcript.json")))
        suite.cli_checks(info, sample, found[0] if found else None)

    if suite.runs:
        print(f"\n| config | transcribe.py | segments | words | WER | time | |\n|---|---|---|---|---|---|---|")
        for r in suite.runs:
            if r.wer is None:
                print(f"| {r.cfg.name} | {r.rev} | – | – | – | {r.seconds:.0f} s | failed |")
            else:
                print(f"| {r.cfg.name} | {r.rev} | {r.segments} | {r.words} | {r.wer:.2f} % | "
                      f"{r.seconds:.0f} s | {', '.join(r.extra)} |")
    for s in suite.skipped:
        print(f"skipped: {s}")
    if suite.failures:
        print(f"\n{len(suite.failures)} FAILED:")
        for f in suite.failures:
            print(f"  ✗ {f.splitlines()[0]}")
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
