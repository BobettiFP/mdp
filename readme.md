# LLM-Driven Dialogue Annotation & RL Benchmark – Codebase Overview

This repository contains everything required to convert raw task-oriented dialogues into Markov Decision Process (MDP) records, train PPO agents on the resulting environments, and evaluate how annotation methodology (human vs LLM) shapes reinforcement‑learning behaviour.

---

## 1  Project layout

```text
repo_root/
├── annotator/            # ⇢ OpenAI-powered annotation
│   └── annotate_no_ontology_v3.py
├── postprocessing/       # ⇢ slot normalisation + (s, a, s′) extraction
│   └── dynamic_slot_analyzer.py
├── rl_env/               # ⇢ Gymnasium environments, PPO training, log parser
│   ├── improved_env.py
│   ├── improved_train.py
│   └── analyze_results.py
├── rl_tests/             # ⇢ research-grade diagnostics
│   ├── transition_dynamics_pipeline.py
│   ├── state_richness_pipeline.py
│   ├── strategic_complexity_pipeline.py
│   └── reward_efficiency_pipeline.py
└── processed_annotations.json  # generated once, consumed everywhere
```

---

## 2  Quick install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # see sample file below
echo "OPENAI_API_KEY=sk-..." > .env
```

`requirements.txt` (example):

```
openai
python-dotenv
tqdm
gymnasium
stable-baselines3
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## 3  End-to-end pipeline at a glance

```bash
# ❶ LLM annotation  ➜  ConvLab-3 style JSON
python annotator/annotate_no_ontology_v3.py \
       --input data/raw_dialogues.json \
       --output annotator/llm_annotations.json \
       --threads 20 --stats

# ❷ Post-process annotations  ➜  processed_annotations.json
python postprocessing/dynamic_slot_analyzer.py \
       --human annotator/human_annotations.json \
       --llm   annotator/llm_annotations.json \
       --export processed_annotations.json

# ❸ Train PPO on four environments (human/LLM × easy/hard)
python rl_env/improved_train.py \
       --annotations processed_annotations.json \
       --steps 50000 \
       --outdir rl_env/logs

# ❹ Analyse learning curves
python rl_env/analyze_results.py \
       --logdir rl_env/logs \
       --output rl_env/analysis

# ❺ Research metrics (state richness, transition dynamics, …)
python rl_tests/transition_dynamics_pipeline.py \
       --annotations processed_annotations.json \
       --outdir rl_tests/td_results
```

All scripts print a short **cost/time report**, while analysis steps write PNG figures and Markdown summaries to their respective `--outdir`.

---

## 4  Component highlights

| Stage                 | What happens                                                                                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Annotation**        | GPT‑4o‑mini extracts dialogue acts, slot updates, and belief state per turn. Costs and latency are logged automatically.                                      |
| **Post‑processing**   | Slot names are canonicalised, per‑turn `(state_before, action, state_after)` triples are generated, and a global `canonical_map` is emitted for later audits. |
| **Environment build** | `improved_env.py` builds a Gymnasium environment on the fly, with reward shaping and optional stochastic noise that can be toggled via `difficulty` flags.    |
| **PPO training**      | `improved_train.py` uses Stable‑Baselines3. Hyper‑parameters are adapted to the chosen difficulty, and CSV / TensorBoard logs are saved for every run.        |
| **Analysis**          | `analyze_results.py` plots reward and success‑rate curves, ranks environments, and writes a Markdown executive summary.                                       |
| **Research tests**    | Four pipelines in `rl_tests` compute bespoke metrics (e.g., state entropy, planning depth) to quantify how annotation quality affects RL learning dynamics.   |

---

## 5  Reproducibility & tips

* **Determinism:** set `PYTHONHASHSEED`, NumPy and torch seeds; pass `--seed` to training script.
* **Speed:** increase `--threads` during annotation and use GPU‑enabled PyTorch for PPO.
* **Failure rate:** if OpenAI calls occasionally fail, retry logic inside the annotator backs off automatically; for persistent issues lower `temperature` to 0 – 0.2.

---

## 6  Citation

TBA

---

## 7  License

MIT. See `LICENSE` for full terms.

---

Questions or pull requests are welcome. Enjoy experimenting!
