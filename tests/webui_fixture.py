#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""
tests/webui_fixture.py – a public test recording for the web UI, with known answers.

Builds work/webui-fixture/videos/voice-messages-test.mp4: a 6:44 screen recording of
someone playing voice messages in a chat app, made from two public-domain LibriVox
readings in Czech (downloaded from archive.org on first use):

  voice A  Viktor Dyk, Krysař, chapters 2 and 4, read by Kudrna
  voice B  K. J. Erben, Smrt kmotřenka, read by mch (Multilingual Short Works 022)

Ten messages alternate A, B, A, B, … with exact digital silence between them. Three
audio tracks: 1 = desktop + microphone, 2 = desktop only, 3 = microphone only (room
noise). What the review screen should find, all on track 2:

  1:59  the recording was paused mid-sentence: a jump in the picture while B talks
  2:40  paused, then the message replayed from an earlier point: a jump plus
        ~35 words said twice
  3:09  the picture changes in the middle of a silence: nothing lost, no card
  3:11  B's message starts mid-sentence ("najde tam…"). Whisper tends to capitalise
        it, so this one is reported but not required
  4:45  40 s of silence (the picture changes halfway: no card), then an A message
        stopped mid-sentence, 4 s of silence and played again from its start: one card
        for both ends of that silence
  5:48  25 s of silence, then a plain B message

    uv run tests/webui_fixture.py              build, transcribe track 2 (cs), diarize, check
    uv run tests/webui_fixture.py --build-only just the video, e.g. to click through the UI

To click through the UI with it, run the web UI from a separate checkout: its
--videos folder is remembered in work/.webui-settings.json.
Everything lands in work/webui-fixture/. Exit code 1 if a check fails.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "work" / "webui-fixture"
VIDEO = OUT / "videos" / "voice-messages-test.mp4"
RUN = OUT / "work" / "voice-messages-test"

SOURCES = {  # all public domain (LibriVox)
    "A2": "https://archive.org/download/krysar_2007_librivox/krysar_02_dyk_64kb.mp3",
    "A4": "https://archive.org/download/krysar_2007_librivox/krysar_04_dyk_64kb.mp3",
    "B": "https://archive.org/download/multilingualshortcollection_022_1909_librivox/"
         "msw022_02_smrtkmotrenka_erben_mch_64kb.mp3",
}
# (voice, [(source, from s, to s) or seconds of silence, …]). Parts that follow each other
# directly were split by pausing the recording (no gap, the picture jumps); a number between
# two parts is playback stopped and started again while recording.
# Cuts sit between sentences, except the pause cuts and the late start of message 6.
MESSAGES = [
    ("A", [("A2", 26.728, 57.925)]),
    ("B", [("B", 18.8, 54.668)]),
    ("A", [("A4", 20.474, 49.221)]),
    ("B", [("B", 54.668, 70.232), ("B", 82.426, 102.406)]),   # "…trochu | úctit…" lost
    ("A", [("A4", 60.275, 79.768), ("A4", 65.3, 92.752)]),    # replayed from "Dohodil mu"
    ("B", [("B", 104.884, 132.862)]),                         # starts at "najde"
    ("A", [("A2", 97.534, 118.23)]),
    ("B", [("B", 157.551, 197.003)]),
    ("A", [("A2", 57.926, 62.699), 4.0, ("A2", 57.926, 76.28)]),  # stopped at "ptáci," and played again
    ("B", [("B", 217.59, 243.47)]),
]
# digital silence before each message, and after the last; long waits before the last two
GAPS = [2.0, 1.6, 2.2, 1.8, 2.0, 3.2, 1.7, 2.1, 40.0, 25.0, 2.0]
SILENT_JUMPS = (5, 8)
REPLAYED = "pod jejími okny zahrada v květu"  # said twice in message 9  # the picture changes halfway through the silence before these messages
SR, FPS, W, H = 48000, 30, 1280, 720


# ----------------------------------------------------------------------------- build

def load(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def build() -> dict:
    src = OUT / "src"
    src.mkdir(parents=True, exist_ok=True)
    audio = {}
    for k, url in SOURCES.items():
        p = src / url.rsplit("/", 1)[1]
        if not p.exists():
            print(f"» downloading {p.name}", flush=True)
            urllib.request.urlretrieve(url, p)
        audio[k] = load(p)

    fade = int(0.02 * SR)
    chunks, t = [], 0.0
    truth = {"messages": [], "pauses": [], "silent_jumps": []}
    for i, (voice, parts) in enumerate(MESSAGES):
        chunks.append(np.zeros(int(GAPS[i] * SR), np.float32)); t += GAPS[i]
        if i in SILENT_JUMPS:
            truth["silent_jumps"].append(round(t - GAPS[i] / 2, 2))
        start = t
        prev, gap_at = None, None
        for j, part in enumerate(parts):
            if isinstance(part, (int, float)):
                gap_at = t
                chunks.append(np.zeros(int(part * SR), np.float32)); t += part
                continue
            k, a, b = part
            x = audio[k][int(a * SR):int(b * SR)].copy()
            n = int(0.004 * SR)  # a pause cut is abrupt; only avoid the click
            x[:n] *= np.linspace(0, 1, n); x[-n:] *= np.linspace(1, 0, n)
            if prev is None or gap_at is not None:
                x[:fade] *= np.linspace(0, 1, fade)
            if j == len(parts) - 1 or isinstance(parts[j + 1], (int, float)):
                x[-fade:] *= np.linspace(1, 0, fade)
            replay = prev is not None and a < prev[2] <= b
            if prev is not None and gap_at is None:
                truth["pauses"].append({"t": round(t, 2), "t_end": round(t, 2), "kind": "repeat" if replay else "jump"})
            elif prev is not None:
                truth["pauses"].append({"t": round(gap_at, 2), "t_end": round(t, 2), "kind": "replay"})
            prev, gap_at = part, None
            chunks.append(x); t += len(x) / SR
        truth["messages"].append({"voice": voice, "s": round(start, 2), "e": round(t, 2)})
    chunks.append(np.zeros(int(GAPS[-1] * SR), np.float32)); t += GAPS[-1]
    truth["midstart"] = truth["messages"][5]["s"]
    truth["duration"] = round(t, 2)

    desk = np.concatenate(chunks)
    desk *= 0.7 / np.abs(desk).max()
    noise = np.random.default_rng(7).standard_normal(len(desk)).astype(np.float32)
    mic = np.convolve(noise, np.ones(8, np.float32) / 8, mode="same") * 10 ** (-60 / 20) * 3
    tracks = {1: desk + mic, 2: desk, 3: mic}

    # the picture after a pause of the recording is another window state (here: light/dark);
    # in a silence the list is only scrolled
    pause_ts = [p["t"] for p in truth["pauses"] if p["kind"] != "replay"]
    VIDEO.parent.mkdir(parents=True, exist_ok=True)
    wavs = []
    for n, x in tracks.items():
        p = OUT / f".track{n}.wav"
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
            wf.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
        wavs.append(p)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-"]
    for p in wavs:
        cmd += ["-i", str(p)]
    cmd += ["-map", "0:v", "-map", "1:a", "-map", "2:a", "-map", "3:a", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", str(VIDEO)]
    print(f"» rendering {VIDEO.relative_to(REPO)}", flush=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin
    for f in range(int(t * FPS)):
        tt = f / FPS
        dark = sum(tt >= c for c in pause_ts) % 2
        scroll = 140 * sum(tt >= c for c in truth["silent_jumps"])
        proc.stdin.write(frame(tt, truth["messages"], scroll, dark).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("ERROR: ffmpeg failed")
    for p in wavs:
        p.unlink()
    (OUT / "truth.json").write_text(json.dumps(truth, indent=1))
    return truth


FONT = ImageFont.load_default(size=22)
FONT_B = ImageFont.load_default(size=26)


def frame(tt: float, msgs: list[dict], scroll: int, dark: int) -> Image.Image:
    """A chat window: a list of voice-message bubbles, the playing one advancing."""
    bg, side, text = ((236, 229, 221), (255, 255, 255), (20, 20, 20)) if not dark else \
        ((18, 24, 28), (32, 38, 44), (225, 225, 225))
    bubble = {"A": (217, 253, 211) if not dark else (0, 92, 75), "B": (255, 255, 255) if not dark else (38, 45, 50)}
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    for i, m in enumerate(msgs):
        y = 90 - scroll + i * 92
        if y > H or y + 70 < 64:
            continue
        x0 = 330 if m["voice"] == "B" else W - 450
        d.rounded_rectangle([x0, y, x0 + 420, y + 70], 14, fill=bubble[m["voice"]])
        playing = m["s"] <= tt <= m["e"]
        frac = (tt - m["s"]) / (m["e"] - m["s"]) if playing else float(tt > m["e"])
        d.ellipse([x0 + 14, y + 17, x0 + 50, y + 53], fill=(0, 150, 120) if playing else (120, 120, 120))
        heights = np.random.default_rng(i).random(40)
        for b in range(40):
            h = int(8 + 30 * heights[b])
            d.rectangle([x0 + 64 + b * 7, y + 35 - h // 2, x0 + 67 + b * 7, y + 35 + h // 2],
                        fill=(0, 150, 120) if b / 40 < frac else (150, 150, 150))
        L = int(m["e"] - m["s"])
        d.text((x0 + 350, y + 22), f"{L // 60}:{L % 60:02d}", fill=(130, 130, 130), font=FONT)
    d.rectangle([0, 0, 300, H], fill=side)
    for i, name in enumerate(["Contact one", "Contact two", "Group", "Notes", "Archive"]):
        y = 90 + i * 80
        d.ellipse([15, y, 65, y + 50], fill=(120 + 25 * i, 150, 190 - 20 * i))
        d.text((80, y + 12), name, fill=text, font=FONT)
    d.rectangle([300, 0, W, 64], fill=(0, 128, 105) if not dark else (32, 44, 51))
    d.text((330, 16), "Contact one", fill=(255, 255, 255), font=FONT_B)
    return im


# ----------------------------------------------------------------------------- transcribe + check

def transcribe() -> None:
    cli = ["uv", "run", "--quiet", str(REPO / "transcribe.py")]
    RUN.mkdir(parents=True, exist_ok=True)
    one = RUN.parent / ".track2.mka"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(VIDEO), "-map", "0:a:1", "-c:a", "copy", str(one)],
                   check=True)
    for args in (["audio", str(one)], ["transcribe", "-l", "cs"], ["diarize"]):
        if subprocess.run(cli + args + ["-w", str(RUN)], cwd=REPO).returncode != 0:
            sys.exit(f"ERROR: transcribe.py {args[0]} failed")
    one.unlink()


def check(truth: dict) -> bool:
    sys.path.insert(0, str(REPO / "webui"))
    import analysis as A
    A.ensure_audio_analysis(RUN)
    A.ensure_scenes(RUN, VIDEO)
    doc = A.build_doc(RUN)
    ok = True

    def report(passed: bool, what: str, required: bool = True) -> None:
        nonlocal ok
        ok &= passed or not required
        print(f"  {'ok  ' if passed else 'FAIL' if required else 'miss'}  {what}")

    print("» checks")
    report(doc["language"] == "cs", f"language cs (got {doc['language']})")
    # pyannote may cut a few seconds of one voice off as a voice of its own (typically after
    # a long silence); the UI marks voices under 5 % as "may be a slice of another voice".
    # What must not happen is a word landing on the *other* person.
    main = {s["id"] for s in doc["speakers"] if s["share"] >= 5}
    slices = [f"{s['id']} {s['seconds']} s" for s in doc["speakers"] if s["share"] < 5]
    report(len(main) == 2, f"2 main voices (got {len(main)}" + (f"; short slices: {', '.join(slices)}" if slices else "") + ")")

    def msg_of(t):  # a word counts from where it starts: WhisperX stretches last words over silences
        return next((i for i, m in enumerate(truth["messages"]) if m["s"] - 0.3 <= t <= m["e"]), None)
    per_msg: dict[int, dict] = {}
    for p in doc["pieces"]:
        for w in p["words"]:
            i = msg_of(w["s"])
            if i is not None and p["spk"] in main:
                per_msg.setdefault(i, {}).setdefault(p["spk"], []).append(w)
    label = {"A": set(), "B": set()}
    for i, by in per_msg.items():
        label[truth["messages"][i]["voice"]].add(max(by, key=lambda k: len(by[k])))
    report(len(label["A"]) == len(label["B"]) == 1 and not label["A"] & label["B"],
           f"every message has its own voice's label ({ {k: sorted(v) for k, v in label.items()} })")
    stray = [f"{w['w']!r} at {w['s']:.1f} s" for i, by in per_msg.items() for spk, ws in by.items()
             if spk not in label[truth["messages"][i]["voice"]] for w in ws]
    report(not stray, "no word given to the other voice" + (f" (got {', '.join(stray[:5])})" if stray else ""))
    first_b = next(s["first"] for s in doc["speakers"] if s["id"] in label["B"]) if len(label["B"]) == 1 else None
    report(first_b is not None and abs(first_b - truth["messages"][1]["s"]) < 1.5,
           f"voice B first heard at {truth['messages'][1]['s']} s (got {first_b})")
    # stretches between digital silences never run across two messages
    across = [f"{m['s']}-{m['e']}" for m in doc["messages"]
              if len({msg_of(w["s"]) for p in doc["pieces"] if p["msg"] == m["i"] for w in p["words"]} - {None}) > 1]
    report(not across, f"{len(doc['messages'])} stretches between silences, none spans two messages"
           + (f" (got {across})" if across else ""))

    cards = doc["pauses"]
    def cards_in(t0, t1, slack=1.5):
        return [c for c in cards if c["t"] - slack <= t1 and t0 <= c["t_end"] + slack]
    for p in truth["pauses"]:
        cs = cards_in(p["t"], p["t_end"])
        kinds = {s["kind"] for c in cs for s in c["signals"]}
        want = {"jump", "repeat"} if p["kind"] == "repeat" else {"repeat"} if p["kind"] == "replay" else {"jump"}
        good = len(cs) == 1 and want <= kinds and cs[0]["confidence"] in ("medium", "high")
        if p["kind"] == "replay" and not good:
            # Whisper often transcribes a take that is played again within its 30 s window only
            # once. Then there is nothing to fix: fine, as long as the text is not doubled.
            text = " ".join(w["w"] for q in doc["pieces"] for w in q["words"] if p["t"] - 30 <= w["s"] <= p["t_end"] + 30)
            n = A.norm_token
            said = [n(x) for x in text.split()]
            phrase = [n(x) for x in REPLAYED.split()]
            times = sum(said[k:k + len(phrase)] == phrase for k in range(len(said)))
            report(times == 1, f"stopped, silence, replayed at {p['t']} s: no card, but Whisper wrote the "
                               f"replayed words once, so nothing to fix (got them {times}x)")
            continue
        what = {"jump": "recording paused", "repeat": "paused, then replayed",
                "replay": "stopped, silence, replayed"}[p["kind"]]
        report(good, f"{what} at {p['t']} s: one {'+'.join(sorted(want))} card (got "
                     f"{'; '.join('+'.join(sorted({s['kind'] for s in c['signals']})) + ', ' + c['confidence'] for c in cs) or 'nothing'})")
    for sj in truth["silent_jumps"]:
        report(not any(s["kind"] == "jump" for c in cards_in(sj, sj, 0.5) for s in c["signals"]),
               f"no jump card for the picture change in the silence at {sj} s")
    report(any(s["kind"] == "midstart" for c in cards_in(truth["midstart"], truth["midstart"], 0.5) for s in c["signals"]),
           f"message starting mid-sentence at {truth['midstart']} s", required=False)
    known = [(p["t"], p["t_end"]) for p in truth["pauses"]] + [(truth["midstart"],) * 2]
    extra = [c for c in cards if c["confidence"] != "low" and not any(c["t"] - 1.5 <= b and a <= c["t_end"] + 1.5
                                                                     for a, b in known)]
    report(not extra, f"no other medium/high cards (got {[(c['t'], c['confidence']) for c in extra]})")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build-only", action="store_true", help="only build the video")
    ap.add_argument("--rebuild", action="store_true", help="build the video again and redo the transcription "
                                                            "(the old run is moved to work/webui-fixture/previous/)")
    a = ap.parse_args()
    if a.rebuild and RUN.exists():
        dest = OUT / "previous" / time.strftime("%Y-%m-%d_%H-%M-%S")
        dest.mkdir(parents=True)
        shutil.move(str(RUN), str(dest / RUN.name))
    fresh = a.rebuild or not VIDEO.exists() or not (OUT / "truth.json").exists()
    truth = build() if fresh else json.loads((OUT / "truth.json").read_text())
    print(f"» {VIDEO.relative_to(REPO)}  ({truth['duration']} s, 3 audio tracks)")
    if a.build_only:
        return
    if not (RUN / "diarized.json").exists():
        transcribe()
    sys.exit(0 if check(truth) else 1)


if __name__ == "__main__":
    main()
