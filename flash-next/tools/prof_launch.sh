#!/usr/bin/env bash
# Launch the serve script under nsys with the service drop-in's environment.
# Usage: prof_launch.sh OUTDIR
set -euo pipefail
OUT="$1"; mkdir -p "$OUT"
while IFS= read -r line; do
  if [[ "$line" =~ ^Environment=(.*)$ ]]; then kv="${BASH_REMATCH[1]}"; kv="${kv#\"}"; kv="${kv%\"}"; export "$kv"; fi
done < <(cat "$HOME"/.config/systemd/user/vllm-flash-next.service.d/*.conf)
cd /opt/d/vllm-flash-next-0906
exec /usr/local/bin/nsys profile -t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi \
  --capture-range-end=repeat --trace-fork-before-exec=true --sample=none --cpuctxsw=none \
  --force-overwrite=true -o "$OUT/fn" ./serve-qwen38-flash-next.sh --profiler-config '{"profiler":"cuda"}'
