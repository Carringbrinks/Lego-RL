#!/usr/bin/env bash
# Prepare a small deterministic subset of Lego-RL-2699 without modifying the
# downloaded full dataset.
#
# Produces:
#   data/Lego-RL-subset-<N>/
#     task_instances/<task-id>/...
#     index/instance_ids_<N>.txt
#     index/train_<N>.parquet
#
# Only repositories for the selected task directories are restored because
# prepare_repos.py ignores task ids that do not exist under --harbor-root.
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
FULL_ROOT="${LEGO_2699_ROOT:-$REPO_ROOT/data/Lego-RL-2699}"
COUNT="${1:-64}"
WORKERS="${WORKERS:-8}"
SUBSET_ROOT="${SUBSET_ROOT:-$REPO_ROOT/data/Lego-RL-subset-$COUNT}"

SOURCE_TASKS="$FULL_ROOT/openswe_harbor_2699/task_instances"
SOURCE_IDS="$FULL_ROOT/openswe_harbor_2699/index/instance_ids_2699.txt"
SOURCE_JSONL="$FULL_ROOT/openswe_official_2699/openswe_source_2699.jsonl"
TOOLS="$FULL_ROOT/utils"

die() {
    printf '[prepare-subset ERROR] %s\n' "$*" >&2
    exit 1
}

[[ "$COUNT" =~ ^[1-9][0-9]*$ ]] || die "count must be a positive integer"
[[ -d "$SOURCE_TASKS" ]] || die "task directory not found: $SOURCE_TASKS"
[[ -f "$SOURCE_IDS" ]] || die "instance list not found: $SOURCE_IDS"
[[ -f "$SOURCE_JSONL" ]] || die "source jsonl not found: $SOURCE_JSONL"
[[ -f "$TOOLS/prepare_repos.py" ]] || die "prepare_repos.py not found under $TOOLS"
[[ -f "$TOOLS/make_harbor_index.py" ]] || die "make_harbor_index.py not found under $TOOLS"

mkdir -p "$SUBSET_ROOT/task_instances" "$SUBSET_ROOT/index"

head -n "$COUNT" "$SOURCE_IDS" > "$SUBSET_ROOT/index/instance_ids_$COUNT.txt"
actual="$(wc -l < "$SUBSET_ROOT/index/instance_ids_$COUNT.txt")"
[[ "$actual" -eq "$COUNT" ]] || die "requested $COUNT tasks, selected $actual"

printf '[prepare-subset] copying %d task directories\n' "$COUNT"
while IFS= read -r task_id; do
    [[ -n "$task_id" ]] || continue
    source_task="$SOURCE_TASKS/$task_id"
    [[ -d "$source_task" ]] || die "missing source task: $source_task"
    cp -a "$source_task" "$SUBSET_ROOT/task_instances/"
done < "$SUBSET_ROOT/index/instance_ids_$COUNT.txt"

printf '[prepare-subset] restoring repositories for this subset only\n'
(
    cd "$SUBSET_ROOT"
    python "$TOOLS/prepare_repos.py" \
        --harbor-root "$SUBSET_ROOT/task_instances" \
        --jsonl "$SOURCE_JSONL" \
        --workers "$WORKERS"
)

printf '[prepare-subset] building subset index\n'
python "$TOOLS/make_harbor_index.py" \
    --harbor-root "$SUBSET_ROOT/task_instances" \
    --ids "$SUBSET_ROOT/index/instance_ids_$COUNT.txt" \
    --out "$SUBSET_ROOT/index/train_$COUNT.parquet"

task_count="$(find "$SUBSET_ROOT/task_instances" -mindepth 1 -maxdepth 1 -type d | wc -l)"
repo_count="$(find "$SUBSET_ROOT/task_instances" -type d -path '*/environment/repo' | wc -l)"

printf '[prepare-subset] tasks=%d repos=%d\n' "$task_count" "$repo_count"
printf '[prepare-subset] index=%s\n' "$SUBSET_ROOT/index/train_$COUNT.parquet"
[[ "$task_count" -eq "$COUNT" ]] || die "task count mismatch"
[[ "$repo_count" -eq "$COUNT" ]] || die "repo count mismatch; inspect $SUBSET_ROOT/prepare_repos_failed.txt"
