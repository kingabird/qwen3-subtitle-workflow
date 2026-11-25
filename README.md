# Qwen3 中文字幕工作流

在 Apple Silicon Mac 上把中文口播校对后对齐为 SRT。Agent 按 skill 负责校对和断句，`video-subtitle` 负责本地模型推理、时间轴和格式校验。可以在支持 Agent Skills 的本地 Agent 中调用 skill，也可以在终端独立运行工具。

## 安装

需要 Apple Silicon、macOS 15 或更新版本，以及 `uv` 和 FFmpeg（含 `ffprobe`）。下载或克隆本仓库后，在终端进入 `qwen3-subtitle-workflow` 文件夹，即包含 `README.md` 和 `scripts/` 的目录，再运行以下命令。安装脚本会用 `uv` 创建 Python 3.12 环境，为 MLX 转写和 PyTorch 对齐分别安装锁定的依赖。

```sh
brew install uv ffmpeg
bash scripts/setup.sh
```

首次运行模型时会从 Hugging Face 下载并缓存权重。使用本地模型目录时，可设置 `VIDEO_SUBTITLE_ASR_MODEL` 和 `VIDEO_SUBTITLE_ALIGN_MODEL`。模型推理在本机运行。使用 Agent 校对时，转写文字会进入该 Agent 的会话上下文。

- 转写模型 [mlx-community/Qwen3-ASR-0.6B-6bit](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-6bit)
- 对齐模型 [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)

## 在 Agent 中使用

先完成上述运行环境安装，再让 Agent 加载技能。Agent 执行命令的机器需要能访问音频、本仓库的 Python 环境和 Apple Metal。

| Agent | 项目内技能目录 | 本机全局技能目录 | 手动调用 |
| --- | --- | --- | --- |
| Codex | `.agents/skills/` | `~/.agents/skills/` | `$qwen3-subtitle-workflow` |
| Claude Code | `.claude/skills/` | `~/.claude/skills/` | `/qwen3-subtitle-workflow` |
| Cursor | `.agents/skills/` 或 `.cursor/skills/` | `~/.agents/skills/` 或 `~/.cursor/skills/` | 在 `/` 菜单选择技能 |

这些加载方式依据各工具官方文档。当前端到端验证覆盖本机命令行，Claude Code 和 Cursor 的实际会话调用仍待验证。[Codex 文档](https://learn.chatgpt.com/docs/build-skills) · [Claude Code 文档](https://code.claude.com/docs/en/skills) · [Cursor 文档](https://cursor.com/docs/skills)

在 Codex 桌面版里，将本仓库文件夹设为本地项目的主文件夹，然后新建聊天。终端版则先进入这个文件夹，再执行 `codex`。Cursor 可直接打开此文件夹。Claude Code 用户可将 `.agents/skills/qwen3-subtitle-workflow` 整个目录复制到项目的 `.claude/skills/` 下。

例如，在聊天里发送以下请求，并附上音频文件路径。

```text
使用 qwen3-subtitle-workflow 为这段音频生成中文字幕 SRT。
```

### 在其他项目里做字幕

在本仓库目录执行以下命令，将技能复制到对应 Agent 的全局目录。下面以 Codex 为例。Claude Code 将目标目录改成 `~/.claude/skills/`，Cursor 可使用 `~/.cursor/skills/`。目标已存在时先检查现有版本，再决定如何更新。

```sh
mkdir -p ~/.agents/skills
cp -R .agents/skills/qwen3-subtitle-workflow ~/.agents/skills/
export PATH="$PWD/.venv/bin:$PATH"
command -v video-subtitle
```

`export` 对当前终端及其启动的进程生效。桌面 Agent 中也可直接告知本仓库 `.venv/bin/video-subtitle` 的绝对路径。技能和命令可用后，在你的音频项目里开展工作即可。

三个 `.venv*` 环境共同提供运行时，保留其相邻目录结构。安装到自定义位置时可用 `--mlx-python` 和 `--asr-python` 指定对应模型环境。受限沙箱遇到 Metal 不可用时，按所用 Agent 的权限机制申请本机 GPU 执行。GPU 授权与文字校对确认各自处理。

## 在终端使用

```sh
mkdir -p out
ffmpeg -nostdin -i examples/window-mint.mp3 -map 0:a:0 -ac 1 -ar 16000 out/audio.wav
.venv/bin/video-subtitle transcribe out/audio.wav out/asr-draft.txt
```

对照音频校对。把正稿保存为 `proofread.txt`，把去掉中文标点的完整口播写入 `transcript.txt`，按语义断句后逐行写入 `cues.txt`。两份对齐用文字去掉空白后应完全一致。当前字幕每条允许 2–18 个可见字符，保留小数点、百分号和英文词内连字符等有语义的符号。仓库样例已提供这三份文字。

```sh
.venv/bin/video-subtitle align out/audio.wav \
  examples/window-mint.transcript.txt examples/window-mint.cues.txt out/window-mint.srt \
  --asr-timed out/asr-draft-timed.json
```

超过默认 150 秒时，`align` 需要 `*-timed.json` 来选分段切点。已有可靠文字稿时，仍可先运行 `transcribe` 取得切点，再用自己的正稿对齐。最终字幕时间来自 Qwen 强制对齐。校验通过后才写入正式 SRT，细节记录在 `*.validation.json`。

## 验证

```sh
.venv/bin/python -m unittest discover -s tests
bash scripts/smoke-test.sh
```

端到端检查使用 `examples/` 中合成的原创短音频，并把它循环七次验证超过 150 秒的分段对齐。脚本打印结果目录。单元测试可独立于模型和 GPU 运行。修改 Python 源码后，先执行 `uv pip install --python .venv/bin/python .` 更新已安装包，再运行测试。

安装脚本要求 macOS 15 或更新版本。已有实测来自 Apple Silicon、macOS 27.2.1、Python 3.12.13，使用已缓存的模型完成 26 秒和 183 秒合成音频验证。最低系统版本和首次下载模型流程仍待实测。一般录音中的背景音乐、多说话人和复杂术语需要按音频复核。

修改 SRT 出入点后，使用下列命令记录新结果。`align` 生成的报告对应当时产物，后续编辑需保留新的复验报告。

```sh
.venv/bin/python -m video_subtitle.commands.validate_srt \
  out/window-mint.srt out/audio.wav --report out/window-mint.review-timing.json
.venv/bin/python -m video_subtitle.commands.validate_subtitle_style \
  out/window-mint.srt --report out/window-mint.review-style.json
```

校验覆盖字幕结构、时间顺序、时长和样式。还需对照音频检查内容与出入点。更改文字或断句时，应同步更新正稿、对齐全文和分句后重新运行 `align`。同一输出前缀的任务依次执行，并发任务使用各自的输出目录。

项目代码采用 [MIT 许可证](LICENSE)。模型和第三方依赖的许可证见各自上游项目。
