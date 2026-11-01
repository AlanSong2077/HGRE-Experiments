# HGRE: History-Guided Runtime Estimation for Efficient Agent Tool Polling

Code and data accompanying our paper submitted to the **ICSE Agent Workshop**.

> **Status:** under review. This repository is released to support reproducibility;
> it does not imply acceptance.

Authors: **Yinsheng Song**, **Mengshan Li**

## Overview

LLM agents often execute tools asynchronously. When reliable completion events are
unavailable, an agent may poll with shell `sleep` and status checks. Long sleeps delay
completion detection; frequent checks add overhead. **HGRE** predicts tool runtime from
recent same-family executions and uses the estimate to schedule the first status
inspection.

This repository contains the two experiments reported in the paper:

| Paper experiment | Folder | Data | Saved results |
|---|---|---|---|
| LeetCode runtime prediction | `leetcode_runtime/` | 633 problems / 3,042 trials | 2,622-paper subset |
| Controlled local-tool runtime | `local_tool_runtime/` | 4 families x 10 scales | 120 trials |

The LeetCode paper subset is the first 543 question IDs in ascending order:
2,622 trials, 549 cold-start, 2,073 warm-start, 504 full and 39 partial problems.
The raw file keeps all 3,042 trials.

## Repository layout

```text
HGRE-Experiments/
├── hgre_core/                 shared history-based predictor
│   ├── paths.py               absolute project paths
│   ├── config.py              predictor input/output data contract
│   ├── api.py                 history adapter
│   ├── history.py             candidate predictors
│   └── predictor.py           shared prediction entry point
├── leetcode_runtime/          LeetCode experiment
│   ├── data/                  raw problem records
│   ├── results/               trials, policy replay, LLM-prior cache
│   └── *.py                   agent, executor, predictor, runner
├── local_tool_runtime/        controlled local-tool experiment
│   ├── cases.json             40 fixed workloads
│   ├── data/                  generated workload inputs (git-ignored)
│   ├── workloads/             CSV/JSONL/gzip/matrix scripts
│   ├── results/               trials, policy replay, LLM-prior cache
│   └── *.py                   agent, executor, predictor, runner
├── requirements.txt
└── LICENSE
```

## Installation

Python 3.10+ is recommended.

```bash
cd /Users/alansong/Desktop/FORDEEPSEEK/HGRE-Experiments
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the API key in the shell. No `.env` file is included or required:

```bash
export DEEPSEEK_API_KEY=...
# optional:
export DEEPSEEK_MODEL=deepseek-chat
export DEEPSEEK_BASE_URL=https://api.deepseek.com
```

## Run the experiments

### LeetCode

```bash
cd /Users/alansong/Desktop/FORDEEPSEEK/HGRE-Experiments
python -m leetcode_runtime.run_trials --reps 5
```

This runs all 633 problems. Use the first 543 question IDs for the exact paper
subset when reproducing the paper tables.

### Controlled local tools

```bash
cd /Users/alansong/Desktop/FORDEEPSEEK/HGRE-Experiments
python -m local_tool_runtime.run_trials --reps 3
```

The first run calls `local_tool_runtime.generate_data` and creates the large CSV,
JSONL, gzip, and matrix inputs (about 2.4 GB). These generated files are git-ignored.

## Data and results

- `leetcode_runtime/data/leetcode_full.jsonl`: raw LeetCode records used by the
  repeat-execution experiment.
- `leetcode_runtime/results/trials.jsonl`: 3,042 measured trials.
- `local_tool_runtime/results/trials.jsonl`: 120 measured controlled trials.
- `results/strategies.jsonl`: counterfactual replay for every waiting policy.
- `llm_prior_cache.json`: cached LLM priors used by the baselines.

## Absolute paths

All data and result paths are centralized in `hgre_core/paths.py` and point to the
absolute location of this project. If you move the repository, update that file.

## Citation

If you use this code, please cite the paper (details will be updated after review):

```bibtex
@inproceedings{song2026hgre,
  title     = {History-Guided Runtime Estimation for Efficient Agent Tool Polling},
  author    = {Song, Yinsheng and Li, Mengshan},
  booktitle = {ICSE Agent Workshop},
  year      = {2026},
  note      = {Under review}
}
```

## License

MIT License. See `LICENSE`.

## Contact

- Yinsheng Song: yesirsong@gmail.com
- Mengshan Li: minotsos@outlook.com
