# Atlas RL Copilot

**Idea:** combine **real on-policy RL** (PPO via [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)) with a **spectral instability score** on episode returns—useful when a learning curve is noisy—and an optional **CrewAI** “lab” pass that turns `metrics.json` into a short Markdown report.

This is a small Python project you can extend (harder envs, sweeps, W&B, etc.), not an empty boilerplate.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
atlas-train --timesteps 8000 --out runs/demo --lab
```

Outputs under `runs/demo/`: `metrics.json`, `policy.zip`, and `lab_report.md` (stub advice unless you enable CrewAI).

## Optional CrewAI (OpenAI API)

```bash
pip install -e ".[crew]"
export OPENAI_API_KEY=sk-...
export ATLAS_USE_CREW=1
atlas-train --timesteps 12000 --out runs/with_crew --lab
```

Without the key, `--lab` still writes a deterministic offline report so CI stays green.

## What the instability index is

A high-frequency energy ratio on the centered FFT of the episode-return sequence (see `spectral.py`). Large values often line up with choppy improvement; use it as a **cheap health signal**, not a theorem.

## License

MIT
