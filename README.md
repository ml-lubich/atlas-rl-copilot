# Atlas RL Copilot

**Idea:** combine **real on-policy RL** (PPO via [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)) with a **spectral instability score** on episode returns—useful when a learning curve is noisy—and an optional **CrewAI** “lab” pass that turns `metrics.json` into a short Markdown report.

This is a small Python project you can extend (harder envs, sweeps, W&B, etc.), not an empty boilerplate.

```mermaid
flowchart LR
    CLI{{"💻 atlas-train CLI"}}
    ENV["🌍 env_loader.py<br/>Gymnasium env"]
    PPO["🎯 PPO trainer<br/>Stable-Baselines3"]
    SPEC["📈 spectral.py<br/>FFT instability score"]
    LAB["🧪 crew_lab.py<br/>+ minimax_client.py"]
    OUT[/"📂 runs/&lt;name&gt;/<br/>metrics.json · policy.zip · lab_report.md"/]
    LLM(("🤖 MiniMax / OpenAI"))

    CLI --> ENV --> PPO --> OUT
    PPO --> SPEC --> OUT
    OUT --> LAB --> LLM
    LAB --> OUT

    classDef io fill:#0e1116,stroke:#2f81f7,stroke-width:1.5px,color:#e6edf3;
    classDef brain fill:#161b22,stroke:#d29922,stroke-width:1.5px,color:#e6edf3;
    classDef tool fill:#161b22,stroke:#3fb950,stroke-width:1.5px,color:#e6edf3;
    classDef out fill:#0e1116,stroke:#a371f7,stroke-width:1.5px,color:#e6edf3;
    class CLI brain;
    class LLM io;
    class ENV,PPO,SPEC,LAB tool;
    class OUT out;
```

## Table of contents

- [Quickstart](#quickstart)
- [Training loop (algorithm)](#training-loop-algorithm)
- [Lab report sequence](#lab-report-sequence)
- [MiniMax lab report](#minimax-lab-report-recommended)
- [Optional CrewAI](#optional-crewai-openai-api)
- [What the instability index is](#what-the-instability-index-is)
- [Project layout](#project-layout)
- [License](#license)
- [🗺️ Repository map](#️-repository-map)
- [📊 Code composition](#-code-composition)

## Training loop (algorithm)

```mermaid
flowchart LR
    A([start])
    B["env_loader<br/>build Gymnasium env"]
    C["PPO.learn(timesteps)"]
    D["collect rollouts<br/>per episode"]
    E["log episode return"]
    F{"timesteps<br/>reached?"}
    G["spectral.py<br/>FFT instability index"]
    H["save policy.zip<br/>+ metrics.json"]
    Z([done])
    A --> B --> C --> D --> E --> F
    F -- "no" --> D
    F -- "yes" --> G --> H --> Z
```

## Lab report sequence

```mermaid
sequenceDiagram
    participant U as user
    participant CLI as atlas-train
    participant T as PPO trainer
    participant S as spectral.py
    participant L as crew_lab.py
    participant LLM as MiniMax / OpenAI

    U->>CLI: --timesteps 12000 --lab
    CLI->>T: learn(env, timesteps)
    T-->>CLI: returns + checkpoints
    CLI->>S: instability(returns)
    S-->>CLI: index
    CLI->>L: write_report(metrics)
    alt MINIMAX_API_KEY set
        L->>LLM: chat(metrics)
        LLM-->>L: advice md
    else no key, ATLAS_USE_CREW=1
        L->>LLM: openai chat
    else offline
        L->>L: deterministic stub
    end
    L-->>U: lab_report.md
```

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
atlas-train --timesteps 8000 --out runs/demo --lab
```

Outputs under `runs/demo/`: `metrics.json`, `policy.zip`, and `lab_report.md` (stub advice unless you enable CrewAI).

## MiniMax lab report (recommended)

1. Copy `.env.example` to `.env` and set `MINIMAX_API_KEY` (keep `.env` local; it is gitignored).
2. Run with `--lab`. The report calls MiniMax `chatcompletion_v2` at `MINIMAX_API_BASE` (default `https://api.minimax.io`).

```bash
cp .env.example .env
# edit .env — add MINIMAX_API_KEY only on your machine
atlas-train --timesteps 12000 --out runs/with_minimax --lab
```

Optional env vars: `MINIMAX_MODEL`, `MINIMAX_API_BASE`.

## Optional CrewAI (OpenAI API)

Used only if **no** `MINIMAX_API_KEY` is set and `ATLAS_USE_CREW=1` with `OPENAI_API_KEY`.

```bash
pip install -e ".[crew]"
export OPENAI_API_KEY=sk-...
export ATLAS_USE_CREW=1
atlas-train --timesteps 12000 --out runs/with_crew --lab
```

Without any LLM key, `--lab` still writes a deterministic offline report so CI stays green.

## What the instability index is

A high-frequency energy ratio on the centered FFT of the episode-return sequence (see `spectral.py`). Large values often line up with choppy improvement; use it as a **cheap health signal**, not a theorem.

## Project layout

```
src/atlas_rl_copilot/
  cli.py             # atlas-train entrypoint
  env_loader.py      # Gymnasium env construction
  spectral.py        # FFT-based instability index
  crew_lab.py        # optional CrewAI advisor
  minimax_client.py  # MiniMax (OpenAI-compatible) client
tests/               # pytest suite (training smoke + unit)
pyproject.toml       # hatchling build, optional [crew] extra
```

## License

MIT


## 🗺️ Repository map

Top-level layout of `atlas-rl-copilot` rendered as a Mermaid mindmap (auto-generated from the on-disk tree).

```mermaid
mindmap
  root((atlas-rl-copilot))
    src/
      atlas_rl_copilot
    tests/
      test_crew_stub.py
      test_minimax_client.py
      test_spectral.py
      test_training_smoke.py
    files
      LICENSE
      README.md
      pyproject.toml
```


## 📊 Code composition

File-type breakdown of source under this repo (skips `.git`, `node_modules`, build caches, lockfiles).

```mermaid
pie showData title File-type composition of atlas-rl-copilot (14 files)
    "Python" : 11
    "Other" : 1
    "TOML" : 1
    "Markdown" : 1
```
