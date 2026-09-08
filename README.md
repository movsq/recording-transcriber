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

## Requirements

- Linux/macOS, Python 3.10–3.12, [`uv`](https://docs.astral.sh/uv/) (handles the Python deps).
  The AMD path also works on Windows, where ROCm ships wheels.
- `ffmpeg` on PATH
- **GPU** with ~6 GB free VRAM for `large-v3` (diarization adds ~2 GB):
  - **NVIDIA** – works out of the box.
  - **AMD** – works through ROCm, with a different transcription backend. See
    [AMD GPUs (ROCm)](#amd-gpus-rocm); the install is not the one-liner above.
  - **Intel Arc / Apple Silicon** – `--device xpu` and `--device mps` take the same
    torch path as AMD. Untested, but nothing in it is AMD-specific.
  - `--device cpu` works everywhere but takes hours for a long recording; a smaller
    model (`-m medium`) helps.
- `uv run transcribe.py devices` prints what was detected and which backend it will use.
- ~10 GB disk: ~5 GB Python env with torch, ~3 GB `large-v3`, plus the recording and its WAV
- For speaker labels: a free HuggingFace account
  1. create a **Read** token at <https://huggingface.co/settings/tokens>
  2. click *Agree and access repository* on
     [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) and
     [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
     (MIT licensed, the form just asks for name and affiliation)
  3. `python -c 'from huggingface_hub import login; login()'` and paste the token

  The script checks all of this *before* spending 20 minutes on transcription.
  Skip speaker labels entirely with `--no-diarize`.

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
uv run transcribe.py devices                   [--json]      # GPU + ASR backend that will be used
uv run transcribe.py status      -w work/x [--json]         # what exists, what is next
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
(`--clear` to start over) and two ids may share one label. `write --only PREFIX`
keeps only labels starting with PREFIX and marks the omitted stretches with
their time range.

Work dir layout:

```
work/x/
  meta.json         source, duration, model, language
  media.mp4         downloaded recording (only when `download` was used)
  audio.wav         mono 16 kHz
  transcript.json   WhisperX segments with word timings
  diarized.json     same, with "speaker" on every word/segment
  speakers.json     {"default": "student", "speakers": {"SPEAKER_03": "teacher"}}
  transcript.txt    paragraph per speaker turn
  transcript.srt    subtitles with [label] prefix
  transcript.tsv    start, end (ms), speaker, text – one segment per row
```

Delete a file to redo that stage, or pass `--force` to `transcribe` / `diarize`.
Redoing `transcribe` removes a stale `diarized.json` automatically.

### For AI agents

Suggested loop: `status --json` → run the command it names → `speakers show
--json` → pick labels from the samples → `speakers set …` → `write`. All
commands exit non-zero with a one-line `ERROR:` on stderr when a precondition
is missing (no token, no access to a gated model, wrong speaker id, missing
stage), so nothing needs a terminal or stdin except `speakers ask` and
`run` without `-y`.

## AMD GPUs (ROCm)

WhisperX transcribes through [CTranslate2](https://github.com/OpenNMT/CTranslate2),
which ships CUDA and CPU kernels only – that one step is the whole reason AMD used
to be out. Everything else in the pipeline (the pyannote VAD, the wav2vec2 word
alignment, pyannote diarization) is plain PyTorch and has always run on ROCm.

So on an AMD card the script keeps all of that and swaps only the transcription
step for the `transformers` implementation of Whisper, which is torch all the way
down. `transformers` is already a WhisperX dependency, so this adds nothing new to
install. Same models, same VAD chunking, same `transcript.json`.

Nothing to configure: `--device auto` (the default) sees the AMD card and picks the
`torch` backend by itself.

```bash
python transcribe.py devices                      # check what was detected
python transcribe.py run meeting.mp4 -l cs
```

Measured on an RX 9070 XT (gfx1201, ROCm 7.13, Windows, `torch` backend): `large-v3`
turns ten minutes of audio into an aligned transcript in about 50 s. `-b 4` and
`-b 8` came out the same, `-b 16` was slower, so the default `-b 4` is a fine
starting point.

### Install

ROCm PyTorch does not come from PyPI, so it goes in first, and WhisperX on top
without letting pip pull the CUDA build back over it. `uv run transcribe.py` builds
its env from PyPI and would do exactly that, so on AMD use a venv and plain
`python transcribe.py`:

```bash
python -m venv .venv
. .venv/bin/activate                              # Windows: .venv\Scripts\activate

# 1. ROCm PyTorch, built for your GPU's arch
pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
    "torch[device-gfx1201]" "torchvision[device-gfx1201]" torchaudio

# 2. WhisperX, minus its torch pin (it pins torch~=2.8; ROCm ships other versions)
pip install --no-deps whisperx==3.8.6
pip install faster-whisper ctranslate2 nltk omegaconf pandas \
    "pyannote.audio>=4.0.0" "transformers>=4.48.0" "huggingface_hub<1.0.0"

python transcribe.py devices
```

Replace `gfx1201` with your card's architecture:

| arch | cards |
|---|---|
| `gfx1201` | RX 9070 / 9070 XT / 9060 XT |
| `gfx1200` | RX 9060 |
| `gfx1100` | RX 7900 XTX / XT / GRE |
| `gfx1101` | RX 7800 XT / 7700 XT |
| `gfx1102` | RX 7600 |
| `gfx1030` | RX 6800 / 6900 XT |
| `gfx1151` | Ryzen AI 300 (Strix Halo) |

`rocminfo` prints yours (`rocminfo | grep gfx` on Linux, `hipinfo` on Windows), and
AMD lists the full set of packages at
<https://repo.amd.com/rocm/whl-multi-arch/>.

### Troubleshooting

- **`--device rocm` says PyTorch is a CUDA build** – pip replaced the ROCm wheels.
  Reinstall step 1 and keep `--no-deps` on whisperx.
- **`has ROCm x.y but sees no GPU`** – on Linux add your user to the `render` and
  `video` groups and log back in. If the card is not officially supported, try
  `HSA_OVERRIDE_GFX_VERSION=11.0.0` (RDNA3) or `10.3.0` (RDNA2).
- **Empty or garbage transcript** – some ROCm versions are flaky in fp16 on some
  cards. `--compute-type float32` costs speed and VRAM but settles it.
- **Out of memory** – the script halves the batch size and retries by itself, but
  starting lower (`-b 2`) or smaller (`-m medium`) is faster than letting it back off.
  `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` helps with fragmentation.
- **First run is slow to start** – MIOpen compiles kernels for your card once and
  caches them; later runs skip it.
- **Already built [`ctranslate2-rocm`](https://github.com/arlo-phoenix/CTranslate2-rocm)?**
  Then `--backend faster-whisper --device rocm` uses it instead.

Three warnings show up on a normal run and none of them mean anything is wrong:
`torchcodec is not installed correctly` (pyannote pulls torchcodec in for file
decoding and pip does not always match it to the ROCm torch build – this script
hands pyannote audio that is already in memory, so that decoder is never used),
`Using AOTriton backend for Efficient Attention` (that is ROCm’s attention kernel
doing its job) and Lightning offering to upgrade the bundled VAD checkpoint.

## SharePoint links

Anonymous "anyone with the link" share links (`/:v:/g/personal/...`) are
downloaded directly. Links that require a login are not supported: download the
file in your browser and pass the path instead. Meeting chat is never part of
the recording file, only the Teams conversation has it.

## Notes from real use

Three-hour Czech university info session, RTX 3060 12 GB:

| stage | time |
|---|---|
| WhisperX large-v3, batch 4 | ~10 min |
| pyannote diarization | ~5 min |

`--batch-size 16` ran out of GPU memory after ~55 % of the audio; the script
now halves the batch size and continues instead of failing. Diarization found
45 voices in a Q&A session; naming the top 3 and labelling the rest `student`
was all that was needed.

## Česky

Skript stáhne záznam (SharePoint odkaz nebo soubor), přepíše ho přes WhisperX,
rozpozná mluvčí přes pyannote a na konci se zeptá, kdo je kdo. Výstup je
`.txt` s odstavci podle mluvčího, `.srt` s časy a `.tsv` tabulka (start, end, mluvčí, text). Vše běží lokálně, potřebuje
GPU (NVIDIA nebo AMD – viz [AMD GPUs](#amd-gpus-rocm)), ffmpeg a pro mluvčí zdarma
HuggingFace token (postup výše).

## License

MIT
