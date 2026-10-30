#!/usr/bin/env bash
# Add the Docker network-policy dependency to every selected task image.
#
# Harbor's Docker backend applies agent allowlist policies by running iptables
# inside the task image. The OpenSWE Miniconda images do not ship iptables.
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
SUBSET_ROOT="${1:-$REPO_ROOT/data/Lego-RL-subset-64}"
TASKS="$SUBSET_ROOT/task_instances"

[[ -d "$TASKS" ]] || { echo "[iptables ERROR] task root not found: $TASKS" >&2; exit 1; }

marker="# [lego-rl] Harbor Docker network-policy dependency: iptables"
block="$marker
RUN apt-get update && \\
    apt-get install -y --no-install-recommends iptables && \\
    rm -rf /var/lib/apt/lists/*"

updated=0
skipped=0

while IFS= read -r dockerfile; do
    if grep -Fq "$marker" "$dockerfile"; then
        skipped=$((skipped + 1))
        continue
    fi

    if [[ "$(tail -n 1 "$dockerfile")" != "RUN mkdir -p /logs" ]]; then
        echo "[iptables ERROR] unexpected Dockerfile ending: $dockerfile" >&2
        exit 1
    fi

    printf '\n%s\n' "$block" >> "$dockerfile"
    updated=$((updated + 1))
done < <(find "$TASKS" -type f -path '*/environment/Dockerfile' | sort)

printf '[iptables] updated=%d already_present=%d\n' "$updated" "$skipped"
