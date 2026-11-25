# Qwen3 Chinese Subtitle Workflow

[简体中文](README.md) | English

Create proofread, time-aligned Chinese SRT subtitles on an Apple Silicon Mac. An agent follows the skill to review the transcript and split it into subtitle cues. The `video-subtitle` CLI handles local model inference, timing, and format validation. Use the skill with a local agent that supports Agent Skills, or run the CLI directly.

## Installation

You need Apple Silicon, macOS 15 or later, `uv`, and FFmpeg with `ffprobe`. Download or clone this repository, then open a terminal in the `qwen3-subtitle-workflow` folder containing `README.md` and `scripts/`. Run the commands below. The setup script uses `uv` to create Python 3.12 environments and installs pinned dependencies in separate environments for MLX transcription and PyTorch alignment.

```sh
brew install uv ffmpeg
bash scripts/setup.sh
```

On first use, model weights are downloaded from Hugging Face and cached. To use local model directories, set `VIDEO_SUBTITLE_ASR_MODEL` and `VIDEO_SUBTITLE_ALIGN_MODEL`. Model inference runs locally. When an agent reviews the transcript, that text becomes part of the agent's conversation context.

- Transcription model — [mlx-community/Qwen3-ASR-0.6B-6bit](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-6bit)
- Alignment model — [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)

## Using an agent

Install the runtime first, then load the skill in your agent. The machine running the agent's commands needs access to the audio, this repository's Python environments, and Apple Metal.

| Agent | Project skill directory | User skill directory | Manual invocation |
| --- | --- | --- | --- |
| Codex | `.agents/skills/` | `~/.agents/skills/` | `$qwen3-subtitle-workflow` |
| Claude Code | `.claude/skills/` | `~/.claude/skills/` | `/qwen3-subtitle-workflow` |
| Cursor | `.agents/skills/` or `.cursor/skills/` | `~/.agents/skills/` or `~/.cursor/skills/` | Select the skill from the `/` menu |

These loading instructions follow each tool's official documentation. End-to-end testing currently covers the local CLI. Invocation from actual Claude Code and Cursor sessions still needs verification. [Codex documentation](https://learn.chatgpt.com/docs/build-skills) · [Claude Code documentation](https://code.claude.com/docs/en/skills) · [Cursor documentation](https://cursor.com/docs/skills)

In the Codex desktop app, set this repository folder as the primary folder of a local project, then start a new chat. With Codex CLI, change into this folder and run `codex`. Cursor users can open this folder directly. Claude Code users can copy the entire `.agents/skills/qwen3-subtitle-workflow` directory into their project's `.claude/skills/` directory.

For example, send this request in chat and provide the path to your audio file.

```text
Use qwen3-subtitle-workflow to create Chinese SRT subtitles for this audio.
```

### Using the skill in other projects

Run the following commands from this repository to copy the skill into your agent's user skill directory. This example uses Codex. For Claude Code, change the destination to `~/.claude/skills/`. For Cursor, you can use `~/.cursor/skills/`. If the destination already exists, review the installed version before updating it.

```sh
mkdir -p ~/.agents/skills
cp -R .agents/skills/qwen3-subtitle-workflow ~/.agents/skills/
export PATH="$PWD/.venv/bin:$PATH"
command -v video-subtitle
```

The `export` applies to the current terminal and processes launched from it. For a desktop agent, you can also provide the absolute path to this repository's `.venv/bin/video-subtitle`. Once the skill and CLI are available, work from your audio project's folder.

The three `.venv*` environments together provide the runtime. Keep them next to each other in the repository. For a custom installation, use `--mlx-python` and `--asr-python` to specify the corresponding model environment's Python executable. If a sandbox reports that Metal is unavailable, use your agent's approval mechanism to request execution with access to the local GPU. GPU access and transcript approval are separate decisions.

## Using the CLI

```sh
mkdir -p out
ffmpeg -nostdin -i examples/window-mint.mp3 -map 0:a:0 -ac 1 -ar 16000 out/audio.wav
.venv/bin/video-subtitle transcribe out/audio.wav out/asr-draft.txt
```

Review the transcript against the audio. Save the proofread text as `proofread.txt`. Create `transcript.txt` containing the full spoken text with Chinese punctuation removed, and `cues.txt` containing one naturally phrased subtitle cue per line. The two alignment inputs must match exactly after whitespace is removed. Each cue currently allows 2–18 visible characters, with meaningful symbols such as decimal points, percent signs, and hyphens within English words preserved. The example includes all three text files.

```sh
.venv/bin/video-subtitle align out/audio.wav \
  examples/window-mint.transcript.txt examples/window-mint.cues.txt out/window-mint.srt \
  --asr-timed out/asr-draft-timed.json
```

For audio longer than the default 150-second alignment limit, `align` requires a `*-timed.json` file to choose chunk boundaries. Even if you already have a reliable transcript, you can first run `transcribe` to obtain these boundaries, then align your own proofread text. Final subtitle timing comes from Qwen forced alignment. The final SRT is written after validation passes, with details recorded in `*.validation.json`.

## Validation

```sh
.venv/bin/python -m unittest discover -s tests
bash scripts/smoke-test.sh
```

The end-to-end check uses the original synthetic audio in `examples/`, then repeats it seven times to test alignment of audio longer than 150 seconds. The script prints its output directory. Unit tests run without models or a GPU. After changing Python source files, update the installed package with `uv pip install --python .venv/bin/python .` before running tests.

The setup script requires macOS 15 or later. Testing has been completed on Apple Silicon with macOS 27.2.1 and Python 3.12.13, using cached models and synthetic audio lasting 26 and 183 seconds. The minimum supported OS version and first-time model download flow still need testing. Recordings with background music, multiple speakers, or specialized terminology require review against the audio.

After editing subtitle start or end times, run the following commands to record fresh validation results. The report generated by `align` describes its original output. Keep new validation reports for subsequent edits.

```sh
.venv/bin/python -m video_subtitle.commands.validate_srt \
  out/window-mint.srt out/audio.wav --report out/window-mint.review-timing.json
.venv/bin/python -m video_subtitle.commands.validate_subtitle_style \
  out/window-mint.srt --report out/window-mint.review-style.json
```

Validation checks subtitle structure, chronological order, duration, and style. Also review the wording and cue boundaries against the audio. When changing text or cue segmentation, update the proofread text, full transcript, and cue file together, then run `align` again. Run jobs sharing an output prefix sequentially. Give concurrent jobs separate output directories.

Project code is licensed under the [MIT License](LICENSE). Refer to the upstream projects for model and third-party dependency licenses.
