#!/usr/bin/env bash
# Restart vllm-flash-next and block until the API answers (or the unit fails).
# Exit 0 = ready, 1 = failed.
systemctl --user daemon-reload
systemctl --user reset-failed vllm-flash-next 2>/dev/null
systemctl --user restart vllm-flash-next
for i in $(seq 1 360); do
  curl -sf localhost:8000/v1/models >/dev/null && exit 0
  st=$(systemctl --user is-active vllm-flash-next)
  [[ "$st" == "failed" || "$st" == "inactive" ]] && { echo "unit $st"; exit 1; }
  sleep 5
done
echo "timeout waiting for ready"; exit 1
