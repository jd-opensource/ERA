"""Visualize sync or async timeline JSONL files in seconds.

The input must be a directory containing ``timeline_rank*.jsonl``.  Selected
steps can be drawn together in one PNG (the default) or separately.  Leaf
operations (rollout inference, environment work, and actor training) are
treated as actual compute.  Only the interval from actor phase end to step end
is known to be phase-B barrier wait.  Remaining gaps are shown as unattributed
overhead rather than being mislabeled as wait.

expected processing directory structure:
    timeline_dir/
        timeline_rank0.jsonl
        timeline_rank1.jsonl
        ...

example usage:
    python tools/visualize_timeline.py \
        --step 1-5 \
        --mode combined \
        --dpi 160 \
        /path/to/timeline_dir
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


NS_PER_SECOND = 1_000_000_000
MAX_COMBINED_STEPS = 10
ROLE_ORDER = ("actor", "rollout_infer", "env")
ROLE_LABELS = {
    "actor": "Actor",
    "rollout_infer": "Rollout infer",
    "env": "Env",
}

COLORS = {
    "env_reset": "#7fc97f",
    "rollout_infer": "#386cb0",
    "env_step": "#f0027f",
    "actor_train": "#fdc086",  # legacy single train block
    "forward": "#fdc086",
    "backward": "#fc8d62",
    "weight_update": "#e6ab02",
    "weight_extract": "#66c2a5",
    "weight_broadcast": "#1b9e77",
    "weight_recv": "#8da0cb",
    "weight_load": "#7570b3",
    "known_wait": "#ffd92f",
    "overhead": "#d9d9d9",
}

# weight_recv is intentionally omitted: its wall clock belongs to the prior
# step's wait and paints over earlier infer gaps on combined plots.
_WEIGHT_EVENT_NAMES = frozenset({
    "weight_extract", "weight_broadcast", "weight_load",
})
_ACTOR_COMPUTE_NAMES = frozenset({
    "forward", "backward", "weight_update", "actor_train",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize timeline_rank*.jsonl files. Times are shown in seconds "
            "relative to the earliest rank start for each step."
        ),
    )
    parser.add_argument(
        "timeline_dir",
        type=Path,
        help="Required directory containing timeline_rank*.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <timeline_dir>/visualization).",
    )
    parser.add_argument(
        "--step",
        default="1-5",
        metavar="START-END",
        help="Step or inclusive step range to render (default: 1-5).",
    )
    parser.add_argument(
        "--mode",
        choices=("combined", "separate"),
        default="combined",
        help=(
            "Draw all selected steps in one image (combined, default) or one "
            "image per step (separate). Combined mode accepts at most 10 steps."
        ),
    )
    parser.add_argument("--dpi", type=int, default=160, help="PNG DPI.")
    return parser.parse_args()


def parse_step_spec(spec: str) -> list[int]:
    """Parse ``N`` or an inclusive ``START-END`` step specification."""
    value = spec.strip()
    try:
        if "-" not in value:
            step = int(value)
            if step < 1:
                raise ValueError
            return [step]
        start_text, end_text = value.split("-", 1)
        start, end = int(start_text), int(end_text)
        if start < 1 or end < start:
            raise ValueError
        return list(range(start, end + 1))
    except ValueError as exc:
        raise ValueError(
            f"Invalid --step value {spec!r}; expected N or START-END, e.g. 3 or 1-5"
        ) from exc


def _merge_timeline_records(
    prev: dict[str, Any], new: dict[str, Any],
) -> dict[str, Any]:
    """Merge two same-step/rank rows (e.g. compute then late weight-sync flush)."""
    out = dict(prev)
    for key, value in new.items():
        if key in ("actor_compute_timeline", "weight_sync_timeline") and isinstance(
            value, list
        ):
            old = out.get(key) if isinstance(out.get(key), list) else []
            out[key] = list(old) + list(value)
        elif key == "step_timeline" and isinstance(value, dict):
            old = out.get("step_timeline") if isinstance(out.get("step_timeline"), dict) else {}
            starts = [
                int(x) for x in (
                    old.get("start_wall_ns"), value.get("start_wall_ns"),
                ) if x is not None
            ]
            ends = [
                int(x) for x in (
                    old.get("end_wall_ns"), value.get("end_wall_ns"),
                ) if x is not None
            ]
            if starts and ends:
                out["step_timeline"] = {
                    "start_wall_ns": min(starts),
                    "end_wall_ns": max(ends),
                }
            else:
                out["step_timeline"] = value
        elif key == "memory" and isinstance(value, dict) and value:
            merged_mem = dict(out.get("memory") or {})
            merged_mem.update(value)
            out["memory"] = merged_mem
        elif value not in (None, [], {}):
            out[key] = value
    return out


def load_timelines(timeline_dir: Path) -> dict[int, dict[int, dict[str, Any]]]:
    if not timeline_dir.is_dir():
        raise ValueError(f"Timeline directory does not exist: {timeline_dir}")

    paths = sorted(timeline_dir.glob("timeline_rank*.jsonl"))
    if not paths:
        raise ValueError(
            f"No timeline_rank*.jsonl files found in: {timeline_dir}"
        )

    by_step: dict[int, dict[int, dict[str, Any]]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    step = int(record["step"])
                    rank = int(record["rank"])
                    _validate_record(record)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    print(
                        f"warning: skipping {path.name}:{line_number}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                # If a resumed run contains a duplicate step, merge leaf timelines
                # (weight sync for the last step is often written in a follow-up row).
                prev = by_step.setdefault(step, {}).get(rank)
                if prev is None:
                    by_step[step][rank] = record
                else:
                    by_step[step][rank] = _merge_timeline_records(prev, record)

    if not by_step:
        raise ValueError(f"No valid timeline records found in: {timeline_dir}")
    return by_step


def _validate_record(record: dict[str, Any]) -> None:
    step_time = record["step_timeline"]
    start = int(step_time["start_wall_ns"])
    end = int(step_time["end_wall_ns"])
    if end < start:
        raise ValueError("step end precedes step start")


def _append_weight_sync_intervals(
    intervals: list[tuple[int, int, str]], record: dict[str, Any],
) -> None:
    events = record.get("weight_sync_timeline")
    if not isinstance(events, list):
        return
    for event in events:
        name = str(event.get("name", "weight_load"))
        # Skip legacy weight_recv (mis-attributed wall clock on next step).
        if name == "weight_recv":
            continue
        if name not in _WEIGHT_EVENT_NAMES:
            name = "weight_load"
        _append_interval(intervals, event, name)


def collect_role_intervals(
    record: dict[str, Any], role: str,
) -> list[tuple[int, int, str]]:
    intervals: list[tuple[int, int, str]] = []
    if role == "actor":
        # Prefer fine-grained fwd/bwd/weight_update (excludes recv/adv waits).
        compute = record.get("actor_compute_timeline")
        if isinstance(compute, list) and compute:
            for event in compute:
                name = str(event.get("name", "actor_train"))
                if name not in _ACTOR_COMPUTE_NAMES:
                    name = "actor_train"
                _append_interval(intervals, event, name)
        else:
            _append_interval(
                intervals, record.get("actor_train_timeline"), "actor_train"
            )
        _append_weight_sync_intervals(intervals, record)
        return intervals

    for epoch in record.get("env_rollout_timeline", []):
        if role == "env":
            reset = epoch.get("env_reset") or {}
            if reset.get("performed"):
                _append_interval(intervals, reset, "env_reset")

        for interaction in epoch.get("interactions", []):
            if role == "rollout_infer":
                _append_interval(
                    intervals,
                    interaction.get("rollout_infer"),
                    "rollout_infer",
                )
            elif role == "env":
                _append_interval(
                    intervals, interaction.get("env_step"), "env_step"
                )

        if role == "rollout_infer":
            _append_interval(
                intervals, epoch.get("bootstrap_infer"), "rollout_infer"
            )

    if role == "rollout_infer":
        _append_weight_sync_intervals(intervals, record)

    return intervals


def _append_interval(
    out: list[tuple[int, int, str]], event: dict[str, Any] | None, name: str,
) -> None:
    if not event:
        return
    start = event.get("start_wall_ns")
    end = event.get("end_wall_ns")
    if start is None or end is None:
        return
    start_ns, end_ns = int(start), int(end)
    if end_ns >= start_ns:
        out.append((start_ns, end_ns, name))


def known_wait_interval(record: dict[str, Any]) -> tuple[int, int] | None:
    """Return the phase-B barrier interval, the only explicitly bounded wait."""
    actor_phase = record.get("actor_phase_timeline") or {}
    step_time = record["step_timeline"]
    start = actor_phase.get("end_wall_ns")
    end = step_time.get("end_wall_ns")
    if start is None or end is None or int(end) < int(start):
        return None
    return int(start), int(end)


def merged_duration_ns(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    merged_total = 0
    current_start, current_end = sorted(intervals)[0]
    for start, end in sorted(intervals)[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged_total += current_end - current_start
            current_start, current_end = start, end
    return merged_total + current_end - current_start


def _step_origin_and_width(records: dict[int, dict[str, Any]]) -> tuple[int, float]:
    origin_ns = min(
        int(record["step_timeline"]["start_wall_ns"])
        for record in records.values()
    )
    end_ns = max(
        int(record["step_timeline"]["end_wall_ns"])
        for record in records.values()
    )
    return origin_ns, max((end_ns - origin_ns) / NS_PER_SECOND, 1e-9)


def _draw_role_rank_timeline(
    ax: Any,
    *,
    role: str,
    step: int,
    rank: int,
    record: dict[str, Any],
    origin_ns: int,
    y: float,
    bar_height: float = 0.62,
) -> dict[str, float | int]:
    step_time = record["step_timeline"]
    step_start = int(step_time["start_wall_ns"])
    step_end = int(step_time["end_wall_ns"])
    step_duration = step_end - step_start

    compute = collect_role_intervals(record, role)
    # Actor fine-grained compute: do not paint a gray "wait" base bar — waits
    # are intentionally omitted from the record.
    paint_overhead = not (
        role == "actor" and isinstance(record.get("actor_compute_timeline"), list)
        and record.get("actor_compute_timeline")
    )
    if paint_overhead:
        ax.broken_barh(
            [(_seconds(step_start, origin_ns), step_duration / NS_PER_SECOND)],
            (y - bar_height / 2, bar_height),
            facecolors=COLORS["overhead"],
            edgecolors="none",
            zorder=1,
        )

    wait = known_wait_interval(record) if role == "actor" else None
    wait_duration = 0
    if paint_overhead and wait is not None:
        wait_duration = wait[1] - wait[0]
        ax.broken_barh(
            [(_seconds(wait[0], origin_ns), wait_duration / NS_PER_SECOND)],
            (y - bar_height / 2, bar_height),
            facecolors=COLORS["known_wait"],
            edgecolors="#8c7800",
            hatch="///",
            linewidth=0.4,
            zorder=2,
        )

    for start, end, name in compute:
        ax.broken_barh(
            [(_seconds(start, origin_ns), (end - start) / NS_PER_SECOND)],
            (y - bar_height / 2, bar_height),
            facecolors=COLORS.get(name, COLORS["actor_train"]),
            edgecolors="none",
            zorder=3,
        )

    compute_duration = merged_duration_ns([(s, e) for s, e, _ in compute])
    overhead_duration = (
        0 if not paint_overhead
        else max(step_duration - compute_duration - wait_duration, 0)
    )
    return {
        "step": step,
        "role": role,
        "rank": rank,
        "step_seconds": step_duration / NS_PER_SECOND,
        "compute_seconds": compute_duration / NS_PER_SECOND,
        "known_wait_seconds": wait_duration / NS_PER_SECOND,
        "overhead_seconds": overhead_duration / NS_PER_SECOND,
        "compute_percent": 100.0 * compute_duration / max(step_duration, 1),
        "known_wait_percent": 100.0 * wait_duration / max(step_duration, 1),
    }


def _role_rank_lanes(
    steps: list[int], by_step: dict[int, dict[int, dict[str, Any]]],
) -> list[tuple[str, int]]:
    lanes: list[tuple[str, int]] = []
    for role in ROLE_ORDER:
        ranks = sorted({
            rank
            for step in steps
            for rank, record in by_step[step].items()
            if collect_role_intervals(record, role)
        })
        lanes.extend((role, rank) for rank in ranks)
    return lanes


def _legend_handles() -> list[Patch]:
    return [
        Patch(facecolor=COLORS["env_reset"], label="Env reset (compute)"),
        Patch(facecolor=COLORS["rollout_infer"], label="Rollout inference (compute)"),
        Patch(facecolor=COLORS["env_step"], label="Env step (compute)"),
        Patch(facecolor=COLORS["forward"], label="Actor forward"),
        Patch(facecolor=COLORS["backward"], label="Actor backward"),
        Patch(facecolor=COLORS["weight_update"], label="Actor weight update"),
        Patch(facecolor=COLORS["weight_extract"], label="Weight extract"),
        Patch(facecolor=COLORS["weight_broadcast"], label="Weight broadcast"),
        Patch(facecolor=COLORS["weight_load"], label="Weight load"),
        Patch(facecolor=COLORS["actor_train"], label="Actor train (legacy)"),
        Patch(
            facecolor=COLORS["known_wait"], edgecolor="#8c7800",
            hatch="///", label="Known phase-B barrier wait",
        ),
        Patch(facecolor=COLORS["overhead"], label="Unattributed overhead / gap"),
    ]


def _finish_axes(ax: Any, width_seconds: float, title: str, xlabel: str) -> None:
    ax.set_xlim(0, width_seconds)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", color="#ffffff", linewidth=0.7, alpha=0.9, zorder=0)
    ax.legend(
        handles=_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
        frameon=False,
    )


def render_step(
    step: int,
    records: dict[int, dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> list[dict[str, float | int]]:
    lanes = _role_rank_lanes([step], {step: records})
    origin_ns, width_seconds = _step_origin_and_width(records)
    fig_height = max(3.2, 0.52 * len(lanes) + 1.8)
    fig, ax = plt.subplots(
        figsize=(max(12, min(26, width_seconds / 6)), fig_height)
    )
    summary_rows = []
    for y, (role, rank) in enumerate(lanes):
        summary_rows.append(
            _draw_role_rank_timeline(
                ax, role=role, step=step, rank=rank, record=records[rank],
                origin_ns=origin_ns, y=y,
            )
        )
        if y + 1 < len(lanes) and lanes[y + 1][0] != role:
            ax.axhline(y + 0.5, color="#777777", linewidth=1.0, alpha=0.8)

    ax.set_yticks(
        range(len(lanes)),
        [f"{ROLE_LABELS[role]} / rank {rank}" for role, rank in lanes],
    )
    ax.set_ylim(-0.8, len(lanes) - 0.2)
    ax.invert_yaxis()
    _finish_axes(
        ax,
        width_seconds,
        f"Component timeline — global step {step}\n"
        "Gray gaps are unattributed overhead; yellow hatch is known phase-B wait",
        "Seconds from earliest rank step start",
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / f"timeline_step_{step:06d}.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)
    return summary_rows


def render_combined(
    selected_steps: list[int],
    by_step: dict[int, dict[int, dict[str, Any]]],
    output_dir: Path,
    dpi: int,
) -> list[dict[str, float | int]]:
    lanes = _role_rank_lanes(selected_steps, by_step)
    fig_height = max(4.0, 0.52 * len(lanes) + 2.2)
    origin_ns = min(
        int(record["step_timeline"]["start_wall_ns"])
        for step in selected_steps
        for record in by_step[step].values()
    )
    end_ns = max(
        int(record["step_timeline"]["end_wall_ns"])
        for step in selected_steps
        for record in by_step[step].values()
    )
    width_seconds = max((end_ns - origin_ns) / NS_PER_SECOND, 1e-9)
    fig, ax = plt.subplots(
        figsize=(max(12, min(26, width_seconds / 6)), fig_height)
    )

    summary_rows: list[dict[str, float | int]] = []
    for y, (role, rank) in enumerate(lanes):
        for step in selected_steps:
            record = by_step[step].get(rank)
            if record is None or not collect_role_intervals(record, role):
                continue
            summary_rows.append(
                _draw_role_rank_timeline(
                    ax, role=role, step=step, rank=rank, record=record,
                    origin_ns=origin_ns, y=y, bar_height=0.58,
                )
            )
        if y + 1 < len(lanes) and lanes[y + 1][0] != role:
            ax.axhline(y + 0.5, color="#777777", linewidth=1.0, alpha=0.8)

    for step in selected_steps:
        step_start_ns = min(
            int(record["step_timeline"]["start_wall_ns"])
            for record in by_step[step].values()
        )
        x = _seconds(step_start_ns, origin_ns)
        ax.axvline(x, color="#555555", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.text(
            x, 1.005, f"step {step}", rotation=90, va="bottom", ha="left",
            fontsize=7, color="#555555", transform=ax.get_xaxis_transform(),
        )

    ax.set_yticks(
        range(len(lanes)),
        [f"{ROLE_LABELS[role]} / rank {rank}" for role, rank in lanes],
    )
    ax.set_ylim(-0.8, len(lanes) - 0.2)
    ax.invert_yaxis()
    step_label = (
        str(selected_steps[0])
        if len(selected_steps) == 1
        else f"{selected_steps[0]}-{selected_steps[-1]}"
    )
    _finish_axes(
        ax,
        width_seconds,
        f"Combined component timeline — steps {step_label}\n"
        "All steps share one wall-clock time axis",
        "Seconds from the earliest selected step start",
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / (
            f"timeline_steps_{selected_steps[0]:06d}-"
            f"{selected_steps[-1]:06d}.png"
        ),
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)
    return summary_rows


def _seconds(timestamp_ns: int, origin_ns: int) -> float:
    return (timestamp_ns - origin_ns) / NS_PER_SECOND


def write_summary(rows: list[dict[str, float | int]], path: Path) -> None:
    fieldnames = [
        "step", "role", "rank", "step_seconds", "compute_seconds",
        "known_wait_seconds", "overhead_seconds", "compute_percent",
        "known_wait_percent",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    try:
        by_step = load_timelines(args.timeline_dir)
        requested_steps = parse_step_spec(args.step)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.mode == "combined" and len(requested_steps) > MAX_COMBINED_STEPS:
        print(
            f"error: combined mode supports at most {MAX_COMBINED_STEPS} steps; "
            f"--step {args.step!r} requests {len(requested_steps)}",
            file=sys.stderr,
        )
        return 2

    selected_steps = [step for step in requested_steps if step in by_step]
    missing = sorted(set(requested_steps) - set(selected_steps))
    if missing:
        print(f"warning: requested steps not found: {missing}", file=sys.stderr)
    if not selected_steps:
        print("error: no steps selected", file=sys.stderr)
        return 2

    output_dir = args.output_dir or args.timeline_dir / "visualization"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float | int]] = []
    if args.mode == "combined":
        summary_rows = render_combined(
            selected_steps, by_step, output_dir, args.dpi,
        )
    else:
        for step in selected_steps:
            summary_rows.extend(
                render_step(step, by_step[step], output_dir, args.dpi)
            )
    summary_path = output_dir / "timeline_summary.csv"
    write_summary(summary_rows, summary_path)

    print(
        f"Rendered {len(selected_steps)} step(s) in {args.mode} mode from "
        f"{sum(len(by_step[s]) for s in selected_steps)} rank-step records."
    )
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
