from pathlib import Path

from atlas_rl_copilot.training import run_ppo_training


def test_ppo_runs_and_writes_metrics(tmp_path: Path) -> None:
    summary = run_ppo_training(
        env_id="CartPole-v1",
        total_timesteps=2048,
        seed=42,
        out_dir=tmp_path,
    )
    assert summary.mean_reward_last_n >= 0.0
    assert (tmp_path / "metrics.json").is_file()
    assert Path(summary.model_path).is_file()
