"""CLI entry: train PPO and optionally generate lab report."""

from __future__ import annotations

import argparse
from pathlib import Path

from atlas_rl_copilot.crew_lab import run_lab_crew
from atlas_rl_copilot.env_loader import load_dotenv_if_present, project_root
from atlas_rl_copilot.training import run_ppo_training


def main() -> None:
    load_dotenv_if_present(project_root())
    parser = argparse.ArgumentParser(description="Atlas RL Copilot — PPO + spectral metrics + lab notes")
    parser.add_argument("--env", default="CartPole-v1", help="Gymnasium env id")
    parser.add_argument("--timesteps", type=int, default=40_000, help="PPO training steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/latest"), help="Output directory")
    parser.add_argument(
        "--lab",
        action="store_true",
        help="Write lab_report.md from metrics (CrewAI if ATLAS_USE_CREW=1 and OPENAI_API_KEY set)",
    )
    args = parser.parse_args()

    summary = run_ppo_training(
        env_id=args.env,
        total_timesteps=args.timesteps,
        seed=args.seed,
        out_dir=args.out,
    )
    print(
        f"Done. mean_reward_last_n={summary.mean_reward_last_n:.2f} "
        f"instability={summary.instability_index:.3f} model={summary.model_path}"
    )
    if args.lab:
        mp = args.out / "metrics.json"
        report = run_lab_crew(mp)
        print(f"Lab report: {report}")


if __name__ == "__main__":
    main()
