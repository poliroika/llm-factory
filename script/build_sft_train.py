#!/usr/bin/env python3
"""
Build an SFT training file from agent datasets and task prompts.

For each task in dataset/agent_tasks/*/tasks.json, write a JSONL record with:
- input: task string
- output: selected agent fields from the corresponding dataset.json
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    base_dir = Path("dataset/agent_tasks")
    output_dir = Path("train")
    output_dir.mkdir(exist_ok=True)

    fields = ["agent_id", "display_name", "persona", "description", "tools"]

    for dataset_dir in sorted(base_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_path = dataset_dir / "dataset.json"
        tasks_path = dataset_dir / "tasks.json"
        if not dataset_path.exists() or not tasks_path.exists():
            continue

        dataset_data = json.loads(dataset_path.read_text(encoding="utf-8"))
        tasks_data = json.loads(tasks_path.read_text(encoding="utf-8"))

        agent_map = {agent["agent_id"]: agent for agent in dataset_data["agents"]}

        output_path = output_dir / f"sft_train_{dataset_dir.name}.jsonl"
        with output_path.open("w", encoding="utf-8") as output_file:
            for agent_entry in tasks_data["agents"]:
                agent = agent_map.get(agent_entry["agent_id"])
                if not agent:
                    continue
                agent_output = {key: agent[key] for key in fields}
                for task in agent_entry.get("tasks", []):
                    record = {"input": task, "output": agent_output}
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
