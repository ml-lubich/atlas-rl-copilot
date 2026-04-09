"""Lab report: MiniMax (preferred), optional CrewAI, or offline stub."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

from atlas_rl_copilot.env_loader import load_dotenv_if_present, project_root


class MetricsPayload(Protocol):
    env_id: str
    total_timesteps: int
    mean_reward_last_n: float
    instability_index: float


def _stub_advice(payload: MetricsPayload) -> str:
    lines = [
        "## Lab note (offline stub)",
        "",
        f"- **Environment**: `{payload.env_id}`",
        f"- **Timesteps**: {payload.total_timesteps}",
        f"- **Recent mean return** (last window): {payload.mean_reward_last_n:.2f}",
        f"- **Instability index** (spectral): {payload.instability_index:.3f}",
        "",
        "Suggestions:",
    ]
    if payload.instability_index > 0.45:
        lines.append("- High oscillation in returns: try lower learning rate or higher entropy coef.")
    else:
        lines.append("- Learning curve looks relatively smooth; consider longer training or harder env.")
    return "\n".join(lines)


def run_lab_crew(metrics_path: Path, out_md: Path | None = None) -> str:
    """
    Produce a Markdown report from metrics.json.

    Resolution order:
    1. **MiniMax** if `MINIMAX_API_KEY` is set (after loading `.env`).
    2. Else **CrewAI** if `ATLAS_USE_CREW` is truthy and `OPENAI_API_KEY` is set.
    3. Else **offline stub** (CI-friendly).
    """
    load_dotenv_if_present(project_root())
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    out_md = out_md or metrics_path.with_name("lab_report.md")

    has_minimax = bool(os.environ.get("MINIMAX_API_KEY", "").strip())
    use_crew = os.environ.get("ATLAS_USE_CREW", "").lower() in ("1", "true", "yes")
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    if has_minimax:
        try:
            text = _minimax_report(data)
        except Exception as exc:  # noqa: BLE001
            text = _stub_advice(_DictPayload(data)) + f"\n\n_(MiniMax failed: {exc})_"
    elif use_crew and has_openai:
        try:
            text = _crewai_report(data)
        except Exception as exc:  # noqa: BLE001
            text = _stub_advice(_DictPayload(data)) + f"\n\n_(CrewAI failed: {exc})_"
    else:
        text = _stub_advice(_DictPayload(data))

    out_md.write_text(text, encoding="utf-8")
    return str(out_md)


class _DictPayload:
    def __init__(self, d: dict) -> None:
        self.env_id = str(d["env_id"])
        self.total_timesteps = int(d["total_timesteps"])
        self.mean_reward_last_n = float(d["mean_reward_last_n"])
        self.instability_index = float(d["instability_index"])


def _minimax_report(data: dict) -> str:
    from atlas_rl_copilot.minimax_client import minimax_chat

    system = (
        "You are an RL training analyst. Reply in GitHub-flavored Markdown only. "
        "Be concise: a title, 3-5 bullets, one 'Next experiment' line."
    )
    user = (
        "Summarize this RL run metrics and comment on stability using instability_index.\n\n"
        f"```json\n{json.dumps(data, indent=2)}\n```"
    )
    return minimax_chat(system_text=system, user_text=user)


def _crewai_report(data: dict) -> str:
    from crewai import Agent, Crew, Process, Task

    analyst = Agent(
        role="RL metrics analyst",
        goal="Explain training health from numeric summaries in 6 bullet points or fewer.",
        backstory="You specialize in classic control and on-policy PPO runs.",
        verbose=False,
        allow_delegation=False,
    )
    task = Task(
        description=(
            "Given JSON metrics with keys env_id, total_timesteps, mean_reward_last_n, "
            "instability_index (0-1, higher = noisier learning), write concise Markdown: "
            "headline, 3-5 bullets, one 'next experiment' suggestion."
            f"\n\n```json\n{json.dumps(data, indent=2)}\n```"
        ),
        expected_output="Markdown report",
        agent=analyst,
    )
    crew = Crew(agents=[analyst], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    return str(result)
