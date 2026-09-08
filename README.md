# recording-transcriber

One script that turns a meeting recording (Teams/SharePoint link or a local
video/audio file) into a text transcript with speaker labels. Runs fully
locally on your GPU – NVIDIA or AMD – with
[WhisperX](https://github.com/m-bain/whisperX) and
[pyannote](https://github.com/pyannote/pyannote-audio). Nothing is uploaded
anywhere.

Input can be a SharePoint share link, any direct URL, or a local video/audio file:

```bash
# Teams recording shared as an "anyone with the link" SharePoint link
uv run transcribe.py run "https://contoso-my.sharepoint.com/:v:/g/personal/user_contoso_com/IQAbc…?e=xyz" -l cs

# a file you already have (mp4, mkv, mp3, wav… anything ffmpeg reads)
uv run transcribe.py run meeting.mp4 -l cs
```

At the end the script shows you who talked the most, with sample sentences, and
asks you to name them:

```
SPEAKER_03  1.6 min (44%), 0:00 – 3:59
   [0:49] So you just pick it there, I'm not going to…
   [3:38] No, it's fixed, unless the study programme has…
   label for SPEAKER_03 [speaker]: teacher
```

Output: `work/<name>/<name>.txt` (paragraph per speaker turn),
`<name>.srt` (subtitles with speaker prefix) and `<name>.tsv` (one segment per
row: start/end in ms, speaker, text).

---

## Requirements

| | |
|---|---|
| OS | Linux, macOS or Windows |
| Python | 3.10 – 3.12 |
| `ffmpeg` | on `PATH` |
| Disk | ~10 GB: ~5 GB Python env with torch, ~3 GB `large-v3`, plus the recording and its WAV |
| GPU | ~6 GB VRAM for `large-v3` on faster-whisper, ~8 GB on the torch backend; diarization adds ~2 GB. Or none at all with `--device cpu` |

Which GPUs work, and how:

| GPU | Transcription backend | Install |
|---|---|---|
| **NVIDIA** | faster-whisper (CTranslate2) | [one command](#nvidia-or-cpu) |
| **AMD** | `transformers` on ROCm | [a few more](#amd-rocm) |
| **Intel Arc** | `transformers` on XPU (`--device xpu`) | untested, same shape as AMD |
| **Apple Silicon** | `transformers` on Metal (`--device mps`) | untested, same shape as AMD |
| **None** | faster-whisper on CPU (`--device cpu`) | works everywhere, takes hours |

You never pick the backend by hand: `--device auto` is the default and figures
it out. `transcribe.py devices` prints what it found.

---

## Install

### 1. ffmpeg

```bash
sudo apt install ffmpeg        # Debian/Ubuntu
sudo pacman -S ffmpeg          # Arch
brew install ffmpeg            # macOS
winget install Gyan.FFmpeg     # Windows (reopen the terminal afterwards)
```

### 2. Python environment

#### NVIDIA or CPU

[`uv`](https://docs.astral.sh/uv/) reads the dependency header inside
`transcribe.py` and builds the environment on first run. There is nothing else
to install:

```bash
uv run transcribe.py devices
```

If transcription later dies on a missing `libcudnn_ops` / `libcublas`, your CUDA
runtime libraries are older than CTranslate2 expects — it needs **cuDNN 9** for
CUDA 12. Install cuDNN 9 system-wide, or follow
[faster-whisper's pip route](https://github.com/SYSTRAN/faster-whisper#gpu):
`pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"`, which on Linux also
needs those directories added to `LD_LIBRARY_PATH`.

#### AMD (ROCm)

ROCm PyTorch does not come from PyPI, and WhisperX pins `torch~=2.8` while ROCm
ships other versions. `uv run` would therefore pull the CUDA build over your
ROCm one. On AMD, build a venv yourself and call `python transcribe.py`
directly.

**Prerequisites.** A current AMD GPU driver. On Linux your user must be in the
`render` and `video` groups (`sudo usermod -aG render,video $USER`, then log out
and back in). No system-wide ROCm SDK is required — the wheels below carry their
own runtime.

**Find your GPU's architecture** (`rocminfo | grep gfx` on Linux, `hipinfo` on
Windows), or read it off here:

| arch | cards |
|---|---|
| `gfx1201` | RX 9070 / 9070 XT / 9060 XT |
| `gfx1200` | RX 9060 |
| `gfx1151` | Ryzen AI 300 "Strix Halo" |
| `gfx1100` | RX 7900 XTX / XT / GRE |
| `gfx1101` | RX 7800 XT / 7700 XT |
| `gfx1102` | RX 7600 / 7600 XT |
| `gfx1030` | RX 6800 / 6800 XT / 6900 XT |
| `gfx1031` | RX 6700 XT |
| `gfx942` / `gfx90a` | Instinct MI300 / MI200 |

The full list of packages is at <https://repo.amd.com/rocm/whl-multi-arch/>.

**Install.** Substitute your arch for `gfx1201` in both places:

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\Activate.ps1

# 1. ROCm PyTorch — must go in first, and stay
pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
    "torch[device-gfx1201]" "torchvision[device-gfx1201]" torchaudio

# 2. WhisperX without its torch pin, then its remaining dependencies
pip install --no-deps whisperx==3.8.6
pip install faster-whisper ctranslate2 nltk omegaconf pandas \
    "pyannote.audio>=4.0.0" "transformers>=4.48.0" "huggingface_hub<1.0.0"

python transcribe.py devices
```

`--no-deps` on step 2 is the whole trick: without it pip resolves WhisperX's
`torch~=2.8` against PyPI and replaces your ROCm build with a CUDA one. If
`devices` reports a CUDA build, that is what happened — redo step 1.

Expected output on a working setup:

```
  torch: 2.9.1+rocm7.13.0
  torch_build: ROCm 7.13.99004-3309c611
  gpu: AMD Radeon RX 9070 XT
  ctranslate2: 4.8.2, 0 CUDA device(s)
  device: cuda
  vendor: amd
  backend: torch

--device auto picks cuda (AMD Radeon RX 9070 XT) with the torch backend.
```

`0 CUDA device(s)` next to ctranslate2 is correct and expected — that library
has no AMD support, which is exactly why the `torch` backend exists.

On AMD, drop the `uv run` prefix from every command in this README and use
`python transcribe.py …` from the activated venv.

### 3. Speaker labels (optional)

Diarization needs a free HuggingFace account, because pyannote's models are
licence-gated:

1. create a **Read** token at <https://huggingface.co/settings/tokens>
2. click *Agree and access repository* on
   [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) and
   [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   (MIT licensed, the form just asks for name and affiliation)
3. `python -c 'from huggingface_hub import login; login()'` and paste the token

The token only ever proves you accepted those licences and lets you download the
weights, which are then cached and run on your own machine. No audio leaves your
computer at any point. Skip all of it with `--no-diarize`.

The script verifies the token and both licences *before* spending 20 minutes on
transcription.

### 4. Check it

```bash
uv run transcribe.py devices
```

---

## Usage

### One shot

```bash
uv run transcribe.py run INPUT [-l cs] [-m large-v3] [-b 4] [--no-diarize] [--default-label speaker] [-y]
                               [--device auto|cuda|rocm|mps|xpu|cpu] [--backend auto|faster-whisper|torch]
```

Runs every stage and asks you to name the speakers at the end. Everything
lands in `work/<name>/`.

### Step by step

Every stage is its own command. Each one is idempotent (skips work that is
already done), prints what it did and what to run next, and reads/writes fixed
file names in the work dir. This is the mode to use from scripts, or when an AI
agent drives the tool: it can run one command, read the output, decide, and
continue.

```bash
uv run transcribe.py devices                     [--json]    # GPU + ASR backend that will be used
uv run transcribe.py status      -w work/x       [--json]    # what exists, what is next
uv run transcribe.py download    URL          -w work/x      # SharePoint share link or direct URL -> media.*
uv run transcribe.py audio       [MEDIA]      -w work/x      # ffmpeg -> audio.wav
uv run transcribe.py transcribe  -w work/x -l cs [--force]   # WhisperX -> transcript.json
                                 [--device rocm] [--backend torch] [--compute-type float32] [--beam-size 5]
uv run transcribe.py diarize     -w work/x [--force]         # pyannote -> diarized.json
uv run transcribe.py speakers show -w work/x [--top 8] [--samples 3] [--json]
uv run transcribe.py speakers set  -w work/x SPEAKER_03=teacher SPEAKER_07=teacher --default student
uv run transcribe.py speakers ask  -w work/x                 # interactive alternative to `set`
uv run transcribe.py write       -w work/x [--only teacher] [--name out]   # -> transcript.txt / .srt / .tsv
```

`speakers show` lists speakers by talk time with their time range and the
longest sentences they said, which is usually enough to tell who is who.
`speakers set` validates the ids, merges into the existing `speakers.json`
(`--clear` to start over) and two ids may share one label. Labels may use any
letters, digits, `_`, `.` and `-` — including accented ones — but no spaces.
`write --only PREFIX` keeps only labels starting with PREFIX and marks the
omitted stretches with their time range.

Work dir layout:

```
work/x/
  meta.json         source, duration, model, language, backend, device
  media.mp4         downloaded recording (only when `download` was used)
  audio.wav         mono 16 kHz
  transcript.json   WhisperX segments with word timings
  diarized.json     same, with "speaker" on every word/segment
  speakers.json     {"default": "student", "speakers": {"SPEAKER_03": "teacher"}}
  transcript.txt    paragraph per speaker turn
  transcript.srt    subtitles with [label] prefix
  transcript.tsv    start, end (ms), speaker, text – one segment per row
```

All of them are UTF-8. Delete a file to redo that stage, or pass `--force` to
`transcribe` / `diarize`. Redoing `transcribe` removes a stale `diarized.json`
automatically.

### For AI agents

Suggested loop: `status --json` → run the command it names → `speakers show
--json` → pick labels from the samples → `speakers set …` → `write`. All
commands exit non-zero with a one-line `ERROR:` on stderr when a precondition
is missing (no token, no access to a gated model, wrong speaker id, missing
stage), so nothing needs a terminal or stdin except `speakers ask` and
`run` without `-y`.

---

## Models and speed

`-m` takes any Whisper size (`tiny`, `base`, `small`, `medium`, `large-v3`), the
`turbo` shorthand for `large-v3-turbo`, or a full HuggingFace repo id.

Three-hour Czech university info session, **RTX 3060 12 GB**, faster-whisper:

| stage | time |
|---|---|
| WhisperX `large-v3`, batch 4 | ~10 min |
| pyannote diarization | ~5 min |

`--batch-size 16` ran out of GPU memory after ~55 % of the audio; the script
halves the batch size and continues instead of failing. Diarization found 45
voices in a Q&A session; naming the top 3 and labelling the rest `student` was
all that was needed.

Ten minutes of audio, **RX 9070 XT 16 GB**, `torch` backend, transcription plus
word alignment:

| model | time | note |
|---|---|---|
| `large-v3` | ~40 s | peaks ~8 GB VRAM |
| `turbo` | ~26 s | ~1.5x faster, measurably less accurate |

Two knobs look like free speed and are not:

- **Raising `-b` does not help on the `torch` backend.** Beam search already
  multiplies the batch by the beam width, so `-b 8` came out twice as slow as
  `-b 4`, and `-b 16` peaked at 23 GB on a 16 GB card and took five times as
  long. If you do run out of memory the script halves the batch and continues.
- **`--beam-size 1` is about 3x faster and silently loses text.** On clean audio
  it is byte-identical to beam 5, which makes it look free; on a real Czech
  monologue it dropped 11 % of the words and two entire clauses. The default of
  5 is what WhisperX already used — lower it only after checking the result.

---

## How it works

```
media ──ffmpeg──> audio.wav ──VAD──> speech chunks ──Whisper──> text
                                                        │
                                          wav2vec2 word alignment
                                                        │
                                            pyannote diarization ──> .txt / .srt / .tsv
```

Only the Whisper step differs between vendors. WhisperX runs it through
[CTranslate2](https://github.com/OpenNMT/CTranslate2), which ships CUDA and CPU
kernels only — that single step is the entire reason AMD used to be excluded.
The voice-activity detection, the wav2vec2 word alignment and pyannote
diarization are plain PyTorch and run wherever torch runs.

So on a non-CUDA GPU the script keeps all of that and swaps just the Whisper
step for the `transformers` implementation, over the same VAD chunks, producing
the same `transcript.json`. `transformers` is already a WhisperX dependency, so
this costs no extra install.

---

## SharePoint links

Anonymous "anyone with the link" share links (`/:v:/g/personal/...`) are
downloaded directly. Links that require a login are not supported: download the
file in your browser and pass the path instead. Meeting chat is never part of
the recording file, only the Teams conversation has it.

---

## Troubleshooting

**`ffmpeg not found`** — install it as above and open a new terminal so `PATH`
is picked up.

**Out of memory** — the script halves the batch size and retries, but starting
lower (`-b 2`) or smaller (`-m medium`) is faster than letting it back off.

**`no HuggingFace token` / `no access to gated model`** — see
[Speaker labels](#3-speaker-labels-optional). If `login()` fails with a 400, you
probably copied the masked value from the token list; a token's real value is
shown only once, at creation.

**Everything is slow and `devices` says `cpu`** — no GPU was detected. On AMD
that usually means pip replaced the ROCm torch build; see below.

### AMD / ROCm

**`--device rocm` says PyTorch is a CUDA build** — pip replaced the ROCm wheels.
Redo step 1 of the install and keep `--no-deps` on whisperx.

**`has ROCm x.y but sees no GPU`** — on Linux, check `rocm-smi` and that you are
in the `render` and `video` groups. If the card is not officially supported, try
`HSA_OVERRIDE_GFX_VERSION=11.0.0` (RDNA3) or `10.3.0` (RDNA2).

**`cuDNN/MIOpen cannot run normalisation layers here`** — not an error. Some
ROCm builds (the Windows ones as of ROCm 7.13) cannot compile MIOpen's batch
normalisation kernel, so the script falls back to PyTorch's own. Measured cost:
none.

**Empty or garbage transcript** — some ROCm versions are flaky in fp16 on some
cards. `--compute-type float32` costs speed and VRAM but settles it.

**Memory fragmentation** — `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`.

**Already built [`ctranslate2-rocm`](https://github.com/arlo-phoenix/CTranslate2-rocm)?**
Then `--backend faster-whisper --device rocm` will use it.

**Three harmless warnings** show up on a normal AMD run: `torchcodec is not
installed correctly` (pyannote pulls torchcodec in for file decoding and pip
does not always match it to the ROCm torch build — this script hands pyannote
audio that is already in memory, so that decoder is never used), `Using AOTriton
backend for Efficient Attention` (that is ROCm's attention kernel doing its job)
and Lightning offering to upgrade the bundled VAD checkpoint.

---

## Česky

Skript stáhne záznam (SharePoint odkaz nebo soubor), přepíše ho přes WhisperX,
rozpozná mluvčí přes pyannote a na konci se zeptá, kdo je kdo. Výstup je `.txt`
s odstavci podle mluvčího, `.srt` s časy a `.tsv` tabulka (start, end, mluvčí,
text).

Vše běží lokálně na vaší grafické kartě, nic se nikam neodesílá. Funguje na
NVIDII i na AMD — instalace se ale liší, viz [Install](#install). Potřebujete
ffmpeg a pro rozpoznávání mluvčích token z HuggingFace (postup výše); bez něj
běží přepis normálně dál, jen přidejte `--no-diarize`.

---

## License

MIT
