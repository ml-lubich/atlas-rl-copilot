"""PPO training on a Gymnasium environment with metric export."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv


@dataclass(frozen=True)
class TrainingSummary:
    env_id: str
    total_timesteps: int
    seed: int
    mean_reward_last_n: float
    episode_returns: list[float]
    instability_index: float
    model_path: str


class EpisodeReturnCollector(BaseCallback):
    """Collect undiscounted episode returns from the vectorized env."""

    def __init__(self, max_episodes: int = 256) -> None:
        super().__init__()
        self._returns: list[float] = []
        self._max_episodes = max_episodes

    @property
    def returns(self) -> list[float]:
        return self._returns

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        for info in infos:
            if "episode" in info and "r" in info["episode"]:
                self._returns.append(float(info["episode"]["r"]))
                if len(self._returns) >= self._max_episodes:
                    return False
        return True


def _make_env(env_id: str, seed: int):
    def _thunk():
        env = gym.make(env_id)
        env.reset(seed=seed)
        return env

    return _thunk


def run_ppo_training(
    *,
    env_id: str = "CartPole-v1",
    total_timesteps: int = 50_000,
    seed: int = 0,
    out_dir: Path | None = None,
) -> TrainingSummary:
    """Train PPO and write metrics plus optional model checkpoint."""
    out_dir = out_dir or Path("runs/default")
    out_dir.mkdir(parents=True, exist_ok=True)

    vec = DummyVecEnv([_make_env(env_id, seed)])
    model = PPO("MlpPolicy", vec, verbose=0, seed=seed)
    collector = EpisodeReturnCollector()
    model.learn(total_timesteps=total_timesteps, callback=collector)
    model_path = out_dir / "policy.zip"
    model.save(str(model_path.with_suffix("")))

    returns_arr = np.array(collector.returns, dtype=np.float64)
    from atlas_rl_copilot.spectral import instability_index

    inst = instability_index(returns_arr) if returns_arr.size else 0.0
    tail = returns_arr[-32:] if returns_arr.size else np.array([], dtype=np.float64)
    mean_last = float(np.mean(tail)) if tail.size else 0.0

    summary = TrainingSummary(
        env_id=env_id,
        total_timesteps=total_timesteps,
        seed=seed,
        mean_reward_last_n=mean_last,
        episode_returns=[float(x) for x in collector.returns],
        instability_index=inst,
        model_path=str(model_path.with_suffix(".zip")),
    )
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    vec.close()
    return summary
