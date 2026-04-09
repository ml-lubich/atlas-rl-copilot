import json
from pathlib import Path

from atlas_rl_copilot.crew_lab import run_lab_crew


def test_lab_stub_writes_markdown(tmp_path: Path) -> None:
    metrics = {
        "env_id": "CartPole-v1",
        "total_timesteps": 1000,
        "seed": 0,
        "mean_reward_last_n": 42.0,
        "episode_returns": [10.0, 20.0],
        "instability_index": 0.9,
        "model_path": "x.zip",
    }
    p = tmp_path / "metrics.json"
    p.write_text(json.dumps(metrics), encoding="utf-8")
    out = run_lab_crew(p, out_md=tmp_path / "lab_report.md")
    text = Path(out).read_text(encoding="utf-8")
    assert "Instability" in text or "instability" in text.lower()
