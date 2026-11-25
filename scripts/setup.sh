#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
  echo '需要 Apple Silicon Mac，并使用原生 arm64 终端。' >&2
  exit 1
fi

macos_version=$(sw_vers -productVersion)
if (( ${macos_version%%.*} < 15 )); then
  echo '需要 macOS 15 或更新版本。' >&2
  exit 1
fi

for command in uv ffmpeg ffprobe; do
  if ! command -v "$command" >/dev/null; then
    echo "缺少 $command。可运行 brew install uv ffmpeg。" >&2
    exit 1
  fi
done

for environment in .venv .venv-mlx .venv-asr; do
  uv venv --python 3.12 --allow-existing "$environment"
done

uv pip install --python .venv/bin/python .
uv pip sync --python .venv-mlx/bin/python requirements/mlx.txt
uv pip sync --python .venv-asr/bin/python requirements/align.txt

.venv/bin/video-subtitle --help
printf '\n安装完成。命令行工具位于 %s/.venv/bin/video-subtitle\n' "$PWD"
