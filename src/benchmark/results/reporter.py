#!/usr/bin/env python3
"""
Generates formatted benchmark reports from collected results.
Supports: console, markdown, csv, latex output.
"""
from collections import defaultdict
from typing import List
from datetime import datetime
import numpy as np


class Reporter:
    """Generates formatted tables from RunResult data."""

    def __init__(self, runs: List):
        self.runs = runs

    def generate(self, fmt: str = "console", include_failed: bool = True) -> str:
        """Generate the full report."""
        sections = []
        sections.append(self._header())
        sections.append(self._status_summary())
        sections.append(self._main_comparison_table(fmt))

        if include_failed:
            failed_runs = [r for r in self.runs if r.status in ("failed", "crashed")]
            if failed_runs:
                sections.append(self._failure_table(failed_runs, fmt))

        completed = [r for r in self.runs if r.status == "completed"]
        if completed:
            sections.append(self._per_seed_table(completed, fmt))

        return "\n".join(sections)

    def _header(self) -> str:
        return "\n".join([
            "",
            "═" * 80,
            "  BENCHMARK RESULTS REPORT",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Total runs scanned: {len(self.runs)}",
            "═" * 80,
            "",
        ])

    def _status_summary(self) -> str:
        statuses = defaultdict(int)
        for r in self.runs:
            statuses[r.status] += 1

        lines = ["  Status Breakdown:"]
        for status in ["completed", "failed", "crashed", "running", "no_output", "unknown"]:
            if statuses[status] > 0:
                icon = {"completed": "✓", "failed": "✗", "crashed": "⚠", 
                        "running": "→", "no_output": "○"}.get(status, "?")
                lines.append(f"    {icon} {status:12s}: {statuses[status]}")
        lines.append("")
        return "\n".join(lines)

    def _main_comparison_table(self, fmt: str) -> str:
        """Generate main comparison table grouped by experiment."""
        groups = defaultdict(list)
        for run in self.runs:
            groups[run.experiment].append(run)

        headers = [
            "Algorithm", "Arch", "Buffer", "Seeds",
            "Done", "Best SR%", "Best Reward",
            "Avg SR%", "Avg Reward"
        ]

        rows = []
        for exp_name in sorted(groups.keys()):
            exp_runs = groups[exp_name]
            completed = [r for r in exp_runs if r.status == "completed"]
            n_seeds = len(exp_runs)
            n_completed = len(completed)

            if completed:
                sr_values = [r.best_eval_success_rate for r in completed 
                            if r.best_eval_success_rate >= 0]
                rw_values = [r.best_eval_reward for r in completed 
                            if r.best_eval_reward > float('-inf')]

                best_sr = max(sr_values) if sr_values else float('nan')
                best_rw = max(rw_values) if rw_values else float('nan')
                avg_sr = np.mean(sr_values) if sr_values else float('nan')
                avg_rw = np.mean(rw_values) if rw_values else float('nan')
            else:
                best_sr = best_rw = avg_sr = avg_rw = float('nan')

            arch = exp_runs[0].architecture
            buf = exp_runs[0].buffer_type

            rows.append([
                exp_name,
                arch,
                buf,
                str(n_seeds),
                f"{n_completed}/{n_seeds}",
                f"{best_sr:.1f}" if not np.isnan(best_sr) else "N/A",
                f"{best_rw:.1f}" if not np.isnan(best_rw) else "N/A",
                f"{avg_sr:.1f}" if not np.isnan(avg_sr) else "N/A",
                f"{avg_rw:.1f}" if not np.isnan(avg_rw) else "N/A",
            ])

        return self._format_table("Main Comparison", headers, rows, fmt)

    def _failure_table(self, failed_runs: List, fmt: str) -> str:
        """Generate failure analysis table."""
        headers = ["Run ID", "Experiment", "Seed", "Status", "Failure Reason", "Episodes"]
        rows = []
        for r in sorted(failed_runs, key=lambda x: x.experiment):
            # Truncate run_id for readability
            display_id = r.run_id if len(r.run_id) <= 45 else r.run_id[:42] + "..."
            rows.append([
                display_id,
                r.experiment,
                str(r.seed),
                r.status,
                r.failure_reason or "unknown",
                str(r.total_episodes),
            ])

        return self._format_table("Failure Analysis", headers, rows, fmt)

    def _per_seed_table(self, completed: List, fmt: str) -> str:
        """Generate detailed per-seed results for completed runs."""
        headers = [
            "Experiment", "Seed", "Episodes", "Steps",
            "Best SR%", "Best Reward", "Best @ Ep",
            "Final SR%", "Final Reward"
        ]
        rows = []
        for r in sorted(completed, key=lambda x: (x.experiment, x.seed)):
            rows.append([
                r.experiment,
                str(r.seed),
                str(r.total_episodes),
                str(r.total_steps),
                f"{r.best_eval_success_rate:.1f}" if r.best_eval_success_rate >= 0 else "N/A",
                f"{r.best_eval_reward:.1f}" if r.best_eval_reward > float('-inf') else "N/A",
                str(r.best_eval_episode) if r.best_eval_episode > 0 else "N/A",
                f"{r.final_eval_success_rate:.1f}" if r.final_eval_success_rate >= 0 else "N/A",
                f"{r.final_eval_reward:.1f}" if r.final_eval_reward > float('-inf') else "N/A",
            ])

        return self._format_table("Per-Seed Details (Completed Runs)", headers, rows, fmt)

    # ─── Formatting backends ──────────────────────────────────

    def _format_table(self, title: str, headers: List[str], rows: List[List[str]], fmt: str) -> str:
        if fmt == "markdown":
            return self._to_markdown(title, headers, rows)
        elif fmt == "csv":
            return self._to_csv(title, headers, rows)
        elif fmt == "latex":
            return self._to_latex(title, headers, rows)
        else:
            return self._to_console(title, headers, rows)

    def _to_console(self, title: str, headers: List[str], rows: List[List[str]]) -> str:
        if not rows:
            return f"\n  {title}: (no data)\n"

        all_rows = [headers] + rows
        widths = [max(len(str(row[i])) for row in all_rows) for i in range(len(headers))]

        lines = [f"\n┌─ {title} {'─' * max(0, 70 - len(title))}┐\n"]

        # Header
        header_line = " │ ".join(h.ljust(w) for h, w in zip(headers, widths))
        lines.append(f"  {header_line}")
        lines.append("  " + "─┼─".join("─" * w for w in widths))

        # Rows
        for row in rows:
            row_line = " │ ".join(str(v).ljust(w) for v, w in zip(row, widths))
            lines.append(f"  {row_line}")

        lines.append("")
        return "\n".join(lines)

    def _to_markdown(self, title: str, headers: List[str], rows: List[List[str]]) -> str:
        lines = [f"\n## {title}\n"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join("---" for _ in headers) + "|")
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        return "\n".join(lines)

    def _to_csv(self, title: str, headers: List[str], rows: List[List[str]]) -> str:
        lines = [f"# {title}"]
        lines.append(",".join(f'"{h}"' for h in headers))
        for row in rows:
            lines.append(",".join(f'"{v}"' for v in row))
        lines.append("")
        return "\n".join(lines)

    def _to_latex(self, title: str, headers: List[str], rows: List[List[str]]) -> str:
        cols = "l" * len(headers)
        lines = [
            f"\n% {title}",
            f"\\begin{{table}}[h]",
            f"\\centering",
            f"\\caption{{{title}}}",
            f"\\begin{{tabular}}{{{cols}}}",
            "\\toprule",
            " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\",
            "\\midrule",
        ]
        for row in rows:
            lines.append(" & ".join(row) + " \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
        return "\n".join(lines)