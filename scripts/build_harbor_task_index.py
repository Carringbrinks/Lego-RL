#!/usr/bin/env python3
"""Build a Lego-RL training index for a directory of Harbor tasks."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
import tomllib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


INDEX_SCHEMA = pa.schema(
    [
        pa.field(
            "prompt",
            pa.list_(
                pa.struct(
                    [
                        pa.field("content", pa.string()),
                        pa.field("role", pa.string()),
                    ]
                )
            ),
        ),
        pa.field(
            "reward_model",
            pa.struct(
                [
                    pa.field("ground_truth", pa.null()),
                    pa.field("style", pa.string()),
                ]
            ),
        ),
        pa.field(
            "extra_info",
            pa.struct(
                [
                    pa.field("data_source", pa.string()),
                    pa.field("harbor_task_path", pa.string()),
                    pa.field("instance_id", pa.string()),
                ]
            ),
        ),
    ]
)


def natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def task_image(task_config: dict) -> str | None:
    environment = task_config.get("environment", {})
    return environment.get("docker_image") if isinstance(environment, dict) else None


def discover_tasks(
    tasks_root: Path,
    *,
    expected_image: str | None,
    expected_count: int | None,
) -> list[tuple[str, Path]]:
    tasks: list[tuple[str, Path]] = []
    for task_dir in sorted(
        (path for path in tasks_root.iterdir() if path.is_dir()),
        key=lambda path: natural_key(path.name),
    ):
        required = [
            task_dir / "task.toml",
            task_dir / "instruction.md",
            task_dir / "tests" / "test.sh",
            task_dir / "solution" / "solve.sh",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        environment_dir = task_dir / "environment"
        if not (
            (environment_dir / "Dockerfile").is_file()
            or (environment_dir / "docker-compose.yaml").is_file()
        ):
            missing.append(str(environment_dir / "Dockerfile"))
        if missing:
            raise ValueError(f"{task_dir.name}: missing required files: {missing}")

        with (task_dir / "task.toml").open("rb") as handle:
            config = tomllib.load(handle)
        config_name = config.get("task", {}).get("name")
        if isinstance(config_name, str) and config_name.rsplit("/", 1)[-1] != task_dir.name:
            raise ValueError(
                f"{task_dir.name}: task.name={config_name!r} does not match directory"
            )
        image = task_image(config)
        if expected_image is not None and image != expected_image:
            raise ValueError(
                f"{task_dir.name}: docker_image={image!r}, expected {expected_image!r}"
            )
        tasks.append((task_dir.name, task_dir.resolve()))

    if not tasks:
        raise ValueError(f"no task directories found under {tasks_root}")
    if expected_count is not None and len(tasks) != expected_count:
        raise ValueError(f"found {len(tasks)} tasks, expected {expected_count}")
    return tasks


def build_rows(tasks: list[tuple[str, Path]]) -> list[dict]:
    rows: list[dict] = []
    for instance_id, task_path in tasks:
        task_path_text = str(task_path)
        rows.append(
            {
                "prompt": [{"content": task_path_text, "role": "user"}],
                "reward_model": {"ground_truth": None, "style": "rule"},
                "extra_info": {
                    "data_source": "harbor",
                    "harbor_task_path": task_path_text,
                    "instance_id": instance_id,
                },
            }
        )
    return rows


def validate_rows(rows: list[dict], expected_count: int | None) -> None:
    instances = [row["extra_info"]["instance_id"] for row in rows]
    paths = [row["extra_info"]["harbor_task_path"] for row in rows]
    if len(instances) != len(set(instances)):
        raise ValueError("duplicate instance_id values")
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate harbor_task_path values")
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"index has {len(rows)} rows, expected {expected_count}")


def write_atomic(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        writer(temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instance-ids", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--expected-image")
    args = parser.parse_args()

    tasks = discover_tasks(
        args.tasks_root.resolve(),
        expected_image=args.expected_image,
        expected_count=args.expected_count,
    )
    rows = build_rows(tasks)
    validate_rows(rows, args.expected_count)
    table = pa.Table.from_pylist(rows, schema=INDEX_SCHEMA)

    def write_parquet(path: Path) -> None:
        pq.write_table(table, path, compression="snappy")

    def write_ids(path: Path) -> None:
        path.write_text("".join(f"{instance_id}\n" for instance_id, _ in tasks))

    write_atomic(args.output.resolve(), write_parquet)
    write_atomic(args.instance_ids.resolve(), write_ids)

    parsed = pq.read_table(args.output.resolve())
    if parsed.schema != INDEX_SCHEMA:
        raise ValueError("written Parquet schema does not match the Lego-RL index schema")
    print(f"wrote {len(rows)} rows to {args.output.resolve()}")
    print(f"wrote {len(tasks)} ids to {args.instance_ids.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
