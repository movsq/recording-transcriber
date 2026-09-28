"""Signals the web UI reads out of a recording, on top of what transcribe.py produces.

Everything here is derived, cached in work/<name>/analysis.json and cheap to redo:

  - audio tracks and their loudness (which track has the speech)
  - digital silence on the transcribed track. Audio played back on a computer
    and captured digitally (voice messages, clips) is separated by exact digital
    silence, while a live pause still carries room noise. Some sources also
    noise-gate pauses into digital silence, so a "message" here is a stretch
    between silences: every real boundary is one, not every one is a real
    boundary. A microphone recording has no digital silence at all; then the
    whole file is one stretch and nothing below depends on it.
  - video scene changes. Pausing a screen recorder leaves no gap in the
    timestamps, only a jump in the picture.

build_doc() combines them with diarized.json (or transcript.json) into the
structure the review screen edits: pieces (a WhisperX segment, split where it
crosses a message boundary), messages, speakers and pause candidates.
"""
from __future__ import annotations

import array
import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import unicodedata
import wave
from pathlib import Path

SILENCE_DB = -70      # digital silence from playback sits around -90 dB, room noise far above
SILENCE_MIN_S = 0.25
SCENE_MIN = 0.03      # small UI animations stay below this; measured pause jumps were 0.035-0.27

ANALYSIS_VERSION = 1


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


# ----------------------------------------------------------------------------- media probing

def probe(path: Path) -> dict:
    out = run(["ffprobe", "-v", "error", "-show_entries",
               "format=duration:stream=index,codec_type,codec_name,channels,width,height,r_frame_rate",
               "-of", "json", str(path)])
    if out.returncode != 0:
        return {"error": out.stderr.strip() or "ffprobe failed"}
    d = json.loads(out.stdout or "{}")
    streams = d.get("streams", [])
    audio = [{"n": i + 1, "codec": s.get("codec_name"), "channels": s.get("channels")}
             for i, s in enumerate(x for x in streams if x.get("codec_type") == "audio")]
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and s.get("codec_name") not in ("mjpeg", "png")), None)  # cover art is not video
    return {
        "duration": float(d.get("format", {}).get("duration") or 0),
        "audio_tracks": audio,
        "video": {"width": video.get("width"), "height": video.get("height")} if video else None,
    }


def track_level(path: Path, track: int) -> dict:
    """Mean and peak loudness of one audio track (1-based), in dBFS."""
    out = run(["ffmpeg", "-nostats", "-v", "info", "-i", str(path), "-map", f"0:a:{track - 1}",
               "-af", "volumedetect", "-f", "null", "-"])
    mean = re.search(r"mean_volume: (-?[\d.]+|-inf) dB", out.stderr)
    peak = re.search(r"max_volume: (-?[\d.]+|-inf) dB", out.stderr)
    f = lambda m: -120.0 if not m or m.group(1) == "-inf" else float(m.group(1))
    return {"n": track, "mean_db": f(mean), "max_db": f(peak)}


def extract_track(src: Path, track: int, dest: Path) -> None:
    """Copy one audio track out of the recording without re-encoding."""
    out = run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map", f"0:a:{track - 1}", "-vn", "-c:a", "copy",
               str(dest)])
    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg could not extract track {track}: {out.stderr.strip()[-300:]}")


# ----------------------------------------------------------------------------- signals

def silences(wav: Path) -> list[list[float]]:
    out = run(["ffmpeg", "-nostats", "-v", "info", "-i", str(wav), "-af",
               f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_MIN_S}", "-f", "null", "-"])
    res, start = [], None
    for m in re.finditer(r"silence_(start|end): (-?[\d.]+)", out.stderr):
        if m.group(1) == "start":
            start = max(0.0, float(m.group(2)))
        elif start is not None:
            res.append([round(start, 3), round(float(m.group(2)), 3)]); start = None
    if start is not None:  # silence running to the end of the file
        res.append([round(start, 3), round(wav_duration(wav), 3)])
    return res


def scenes(media: Path) -> list[list[float]]:
    """Frame-to-frame scene scores >= SCENE_MIN as [time, score]. Decodes the whole video,
    so it runs niced; ~17 s for 5 min of 1440p60 HEVC on 8 cores."""
    cmd = ["ffmpeg", "-nostats", "-v", "error", "-i", str(media), "-map", "0:v:0", "-an", "-vf",
           f"scale=320:-2,select='gte(scene,{SCENE_MIN})',metadata=print:file=-", "-f", "null", "-"]
    if os.name == "posix":
        cmd = ["nice", "-n", "10"] + cmd
    out = run(cmd)
    res, t = [], None
    for line in out.stdout.splitlines():
        m = re.search(r"pts_time:(-?[\d.]+)", line)
        if m:
            t = float(m.group(1)); continue
        m = re.search(r"scene_score=([\d.]+)", line)
        if m and t is not None:
            res.append([round(t, 3), round(float(m.group(1)), 4)])
    return res


def wav_duration(wav: Path) -> float:
    with wave.open(str(wav), "rb") as w:
        return w.getnframes() / w.getframerate()


def peaks(wav: Path, buckets_per_s: int = 20) -> list[int]:
    """Peak level per 1/buckets_per_s second, 0-100, for drawing waveforms."""
    with wave.open(str(wav), "rb") as w:
        if w.getsampwidth() != 2:
            return []
        rate, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        data = array.array("h", w.readframes(n))
    step = max(1, rate * ch // buckets_per_s)
    out = []
    for i in range(0, len(data), step):
        chunk = data[i:i + step]
        v = max(max(chunk), -min(chunk)) / 32768
        # perceptual-ish scale: -60 dB -> 0, 0 dB -> 100
        db = 20 * __import__("math").log10(v) if v > 0 else -120
        out.append(max(0, min(100, round((db + 60) / 60 * 100))))
    return out


# ----------------------------------------------------------------------------- analysis cache

def file_sig(p: Path) -> str:
    st = p.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def load_analysis(workdir: Path) -> dict:
    p = workdir / "analysis.json"
    if p.exists():
        try:
            return json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {}


# the server builds documents from several threads (page loads, background scene detection)
_LOCK = threading.Lock()


def save_analysis(workdir: Path, a: dict) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".analysis-", suffix=".tmp", dir=workdir)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(a, f)
    os.replace(tmp, workdir / "analysis.json")


def ensure_audio_analysis(workdir: Path) -> dict:
    """Silences and peaks of audio.wav (a second or two); recomputed when audio.wav changes."""
    with _LOCK:
        return _ensure_audio_analysis(workdir)


def _ensure_audio_analysis(workdir: Path) -> dict:
    wav = workdir / "audio.wav"
    a = load_analysis(workdir)
    sig = file_sig(wav)
    if a.get("version") == ANALYSIS_VERSION and a.get("audio_sig") == sig and "silences" in a:
        return a
    a = {"version": ANALYSIS_VERSION, "audio_sig": sig, "duration": round(wav_duration(wav), 3),
         "silences": silences(wav), "peaks": peaks(wav),
         # scene detection belongs to the video, keep it across audio redos
         "scenes": a.get("scenes"), "scenes_sig": a.get("scenes_sig")}
    save_analysis(workdir, a)
    return a


def ensure_scenes(workdir: Path, media: Path) -> dict:
    a = load_analysis(workdir)
    sig = file_sig(media)
    if a.get("scenes_sig") == sig and a.get("scenes") is not None:
        return a
    sc = scenes(media)  # slow, so outside the lock
    with _LOCK:
        a = load_analysis(workdir)  # audio analysis may have landed meanwhile
        a.update(scenes=sc, scenes_sig=sig)
        save_analysis(workdir, a)
    return a


# ----------------------------------------------------------------------------- the review document

def norm_token(w: str) -> str:
    w = unicodedata.normalize("NFC", w.lower())
    return "".join(ch for ch in w if ch.isalnum())


def fmt_t(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def _words(seg: dict) -> list[dict]:
    """Segment words with a start/end on every one (WhisperX leaves digits unaligned)."""
    ws = seg.get("words") or []
    if not ws:
        toks = seg["text"].split()
        span = (seg["end"] - seg["start"]) / max(1, len(toks))
        return [{"w": t, "s": seg["start"] + i * span, "e": seg["start"] + (i + 1) * span, "c": None,
                 "spk": seg.get("speaker")} for i, t in enumerate(toks)]
    out = []
    for i, w in enumerate(ws):
        s, e = w.get("start"), w.get("end")
        if s is None:
            s = out[-1]["e"] if out else seg["start"]
        if e is None:
            nxt = next((x.get("start") for x in ws[i + 1:] if x.get("start") is not None), None)
            e = nxt if nxt is not None else max(s, seg["end"])
        out.append({"w": w["word"], "s": float(s), "e": float(e),
                    "c": round(w["score"], 2) if w.get("score") is not None else None,
                    "spk": w.get("speaker", seg.get("speaker"))})
    return out


def _message_spans(sil: list[list[float]], duration: float) -> list[list[float]]:
    spans, t = [], 0.0
    for s, e in sil:
        if s - t > 0.15:
            spans.append([t, s])
        t = max(t, e)
    if duration - t > 0.15:
        spans.append([t, duration])
    return spans or [[0.0, duration]]


def _msg_index(spans, t_mid: float) -> int:
    """Message containing t_mid, or the nearest one when a word was aligned into a silence."""
    best, dist = 0, float("inf")
    for i, (s, e) in enumerate(spans):
        if s <= t_mid <= e:
            return i
        d = min(abs(t_mid - s), abs(t_mid - e))
        if d < dist:
            best, dist = i, d
    return best


def _word_msg(spans, w) -> int:
    """Message a word belongs to. WhisperX often stretches a message's last word over the
    silence after it, or starts the next message's first word early: when the middle of
    the word falls into a silence, the end that lies inside a message decides."""
    mid = (w["s"] + w["e"]) / 2
    if any(s <= mid <= e for s, e in spans):
        return _msg_index(spans, mid)
    for t in (w["s"] + 0.05, w["e"] - 0.05):
        if any(s <= t <= e for s, e in spans):
            return _msg_index(spans, t)
    return _msg_index(spans, mid)


def build_doc(workdir: Path) -> dict:
    src = workdir / "diarized.json"
    if not src.exists():
        src = workdir / "transcript.json"
    raw = src.read_bytes()
    result = json.loads(raw)
    a = ensure_audio_analysis(workdir)
    duration = a["duration"]
    sil = a["silences"]
    spans = _message_spans(sil, duration)
    diarized = src.name == "diarized.json"

    # 1. pieces: segments cut at message boundaries
    pieces = []
    for si, seg in enumerate(result["segments"]):
        ws = _words(seg)
        if not ws:
            continue
        groups: list[list[dict]] = []
        cur_m = None
        for w in ws:
            m = _word_msg(spans, w)
            if m != cur_m:
                groups.append([]); cur_m = m
            w["m"] = m
            groups[-1].append(w)
        for k, g in enumerate(groups):
            text = seg["text"].strip() if len(groups) == 1 else " ".join(w["w"] for w in g)
            dur: dict[str, float] = {}
            for w in g:
                if w["spk"]:
                    dur[w["spk"]] = dur.get(w["spk"], 0) + (w["e"] - w["s"]) + 0.01
            spk_raw = max(dur, key=dur.get) if dur else seg.get("speaker")
            pieces.append({"id": str(si) if len(groups) == 1 else f"{si}.{k}", "msg": g[0]["m"],
                           "s": round(g[0]["s"], 3), "e": round(g[-1]["e"], 3), "text": text,
                           "spk_raw": spk_raw, "spk": spk_raw,
                           "words": [{"w": w["w"], "s": round(w["s"], 3), "e": round(w["e"], 3), "c": w["c"],
                                      "spk": w["spk"]} for w in g]})

    # 2. a stretch between two digital silences (typically one played-back message) has one
    #    speaker: smooth diarization per stretch, unless the minority voice is too long to be
    #    a diarization slip (then it is a real change, or there was no silence to split on)
    messages = []
    for mi, (s, e) in enumerate(spans):
        ps = [p for p in pieces if p["msg"] == mi]
        if not ps:
            continue
        dur: dict[str, float] = {}
        for p in ps:
            for w in p["words"]:
                if w["spk"]:
                    dur[w["spk"]] = dur.get(w["spk"], 0) + (w["e"] - w["s"]) + 0.01
        total = sum(dur.values())
        top = max(dur, key=dur.get) if dur else None
        minority = total - (dur.get(top, 0) if top else 0)
        smooth = top is not None and (minority < 4.0 or dur[top] / total >= 0.85)
        if smooth:
            for p in ps:
                p["spk"] = top
        messages.append({"i": mi, "s": round(s, 3), "e": round(e, 3), "spk": top if smooth else None,
                         "mixed": bool(top) and not smooth})
    for p in pieces:
        for w in p["words"]:
            del w["spk"]

    # 3. speakers, in order of first appearance
    spk: dict[str, dict] = {}
    for p in pieces:
        if not p["spk"]:
            continue
        st = spk.setdefault(p["spk"], {"id": p["spk"], "first": p["s"], "seconds": 0.0, "samples": []})
        st["seconds"] += p["e"] - p["s"]
        st["samples"].append(p)
    speakers = []
    total = sum(s["seconds"] for s in spk.values()) or 1
    for st in sorted(spk.values(), key=lambda x: x["first"]):
        longest = sorted(st["samples"], key=lambda p: -(p["e"] - p["s"]))[:3]
        speakers.append({"id": st["id"], "first": round(st["first"], 2), "seconds": round(st["seconds"], 1),
                         "share": round(100 * st["seconds"] / total, 1),
                         "samples": [{"piece": p["id"], "s": p["s"], "e": p["e"], "text": p["text"]}
                                     for p in sorted(longest, key=lambda p: p["s"])]})

    scenes_ready = a.get("scenes") is not None
    pauses = find_pauses(pieces, messages, sil, a.get("scenes") or [], duration)
    fp = hashlib.sha1(raw + json.dumps(sil).encode()).hexdigest()[:16]
    return {"source": src.name, "diarized": diarized, "duration": duration, "language": result.get("language"),
            "fingerprint": fp, "silences": sil, "messages": messages, "pieces": pieces, "speakers": speakers,
            "pauses": pauses, "scenes_ready": scenes_ready, "peaks": a["peaks"], "peaks_per_s": 20}


# ----------------------------------------------------------------------------- pause candidates

def find_pauses(pieces, messages, sil, scene_list, duration) -> list[dict]:
    """Places where the recording was probably paused. Three independent signals:

    jump      the picture jumps between two frames while something is being said (a jump
              deep inside the silence between two messages is someone clicking around, and
              loses nothing)
    repeat    a stretch of 5+ words said again shortly after: the message was replayed
    midstart  speech after a silence whose first word is lowercase: it starts mid-sentence
    """
    cands = []
    by_id = {p["id"]: p for p in pieces}

    def silence_at(t, slack=0.8):
        for i, (s, e) in enumerate(sil):
            if s - slack <= t <= e + slack:
                return i, s, e
        return None

    # jumps: cluster frames within 0.4 s, keep the strongest
    clusters: list[list[float]] = []
    for t, sc in scene_list:
        if clusters and t - clusters[-1][2] < 0.4:
            c = clusters[-1]; c[2] = t
            if sc > c[1]:
                c[0], c[1] = t, sc
        else:
            clusters.append([t, sc, t])
    for t, sc, _ in clusters:
        if t < 1.5 or t > duration - 1.5:
            continue  # recording start / stop
        s = silence_at(t, 0)
        if s and t - s[1] > 0.8 and s[2] - t > 0.8:
            continue  # between messages, nothing lost
        # a marker goes before the first word said after the jump
        tgt = next(((p, k) for p in pieces for k, w in enumerate(p["words"]) if w["s"] >= t - 0.05), None)
        cands.append({"kind": "jump", "t": t, "score": sc,
                      "fix": {"type": "marker", "piece": tgt[0]["id"], "word": tgt[1]} if tgt else None})

    # repeats
    toks = []
    for p in pieces:
        for k, w in enumerate(p["words"]):
            n = norm_token(w["w"])
            if n:
                toks.append((n, p["id"], k, w["s"], w["e"]))
    N = 4
    index: dict[tuple, list[int]] = {}
    runs = []
    j = 0
    covered_until = -1
    for j in range(len(toks) - N + 1):
        key = tuple(t[0] for t in toks[j:j + N])
        if j >= covered_until:
            for i in index.get(key, []):
                if i + N > j or not (1.5 < toks[j][3] - toks[i][4] < 300):
                    continue
                k = 0
                while j + k < len(toks) and i + k < j and toks[i + k][0] == toks[j + k][0]:
                    k += 1
                if k >= 5:
                    runs.append([i, j, k]); covered_until = j + k
                    break
        index.setdefault(key, []).append(j)
    groups = []
    for i, j, k in runs:
        g = groups[-1] if groups else None
        # a replay rarely matches word for word ("… no way. No way. And then …"): join runs
        # separated or overlapped by a few words
        if g and -3 <= i - g["i1"] <= 4 and -3 <= j - g["j1"] <= 4:
            g["n"] += k - max(0, g["i1"] - i)
            g["i1"], g["j1"] = max(g["i1"], i + k), max(g["j1"], j + k)
        else:
            groups.append({"i0": i, "i1": i + k, "j0": j, "j1": j + k, "n": k})
    for g in groups:
        if g["n"] < 6 or len({toks[x][0] for x in range(g["i0"], g["i1"])}) < 4:
            continue
        def refs(a, b):
            out: dict[str, list[int]] = {}
            for x in range(a, b):
                out.setdefault(toks[x][1], []).append(toks[x][2])
            return [{"piece": pid, "words": ws} for pid, ws in out.items()]
        e0, e1 = toks[g["i0"]][3], toks[g["i1"] - 1][4]
        l0, l1 = toks[g["j0"]][3], toks[g["j1"] - 1][4]
        cands.append({"kind": "repeat", "t": e1, "t2": l0, "words": g["n"],
                      "earlier": [round(e0, 2), round(e1, 2)], "later": [round(l0, 2), round(l1, 2)],
                      "text": " ".join(by_id[toks[x][1]]["words"][toks[x][2]]["w"] for x in range(g["j0"], min(g["j1"], g["j0"] + 14))),
                      "fix": {"type": "repeat", "earlier": refs(g["i0"], g["i1"]), "later": refs(g["j0"], g["j1"])}})

    # stretches starting mid-sentence. Some sources noise-gate the speaker's pauses into
    # digital silence too, so a "message" can also be the rest of a sentence that simply
    # went on; when the text before it has no sentence end, that is the likelier story.
    prev_text = ""
    for m in messages:
        ps = [p for p in pieces if p["msg"] == m["i"]]
        if not ps:
            continue
        first = ps[0]["text"].lstrip(" \"'„“(…-–")
        ch = next((c for c in first if c.isalpha()), "")
        if ch and ch.islower():
            ended = not prev_text or prev_text.rstrip("\"'”“ )").endswith((".", "!", "?", "…"))
            cands.append({"kind": "midstart", "t": m["s"], "text": " ".join(ps[0]["text"].split()[:8]),
                          "weak": not ended, "fix": {"type": "marker", "piece": ps[0]["id"], "word": 0}})
        prev_text = ps[-1]["text"]

    # merge: one card per event. Signals within 3 s of each other, or at the two ends of
    # the same silence (speech cut off -> silence -> speech resumes), belong together.
    def anchors(c):
        ts = [c["t"]] + ([c["t2"]] if "t2" in c else [])
        out = set()
        for t in ts:
            s = silence_at(t)
            out.add(("s", s[0]) if s else ("t", round(t / 3)))
        return out
    cands.sort(key=lambda c: c["t"])
    cards: list[dict] = []
    for c in cands:
        an = anchors(c)
        card = next((k for k in cards if k["_an"] & an or abs(k["t_end"] - c["t"]) <= 3), None)
        if card is None:
            card = {"_an": set(), "t": c["t"], "t_end": c["t"], "signals": []}
            cards.append(card)
        card["_an"] |= an
        card["t"] = min(card["t"], c["t"])
        card["t_end"] = max(card["t_end"], c.get("t2", c["t"]))
        card["signals"].append({k: v for k, v in c.items()})
    for card in cards:
        del card["_an"]
        # one event can shake the picture several times; the strongest frame tells the story
        jumps = [s for s in card["signals"] if s["kind"] == "jump"]
        if len(jumps) > 1:
            keep = max(jumps, key=lambda s: s["score"])
            card["signals"] = [s for s in card["signals"] if s["kind"] != "jump" or s is keep]
        kinds = {s["kind"] for s in card["signals"]}
        best_jump = max((s["score"] for s in card["signals"] if s["kind"] == "jump"), default=0)
        if len(kinds) >= 2 or "repeat" in kinds:
            card["confidence"] = "high"
        elif any(s["kind"] == "midstart" and not s["weak"] for s in card["signals"]) or best_jump >= 0.1:
            card["confidence"] = "medium"
        else:
            card["confidence"] = "low"
        card["id"] = f"p{round(card['t'] * 10)}"
        card["t"], card["t_end"] = round(card["t"], 2), round(card["t_end"], 2)
        # frames to compare: the jump itself, else the moment the new message starts
        jt = next((s["t"] for s in card["signals"] if s["kind"] == "jump"), None)
        card["frame_t"] = round(jt if jt is not None else card["t_end"], 2)
    return cards
