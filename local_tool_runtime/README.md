# Controlled local-tool runtime experiment

Paper experiment: local-tool runtime prediction on CSV, JSONL, gzip, and matrix workloads.

- cases.json: 40 fixed workloads (10 scales x 4 families).
- data/: generated on demand by local_tool_runtime/generate_data.py.
- workloads/: four fixed command scripts.
- results/trials.jsonl: 120 measured trials.
- results/strategies.jsonl: counterfactual policy replay.
- results/llm_prior_cache.json: cached LLM priors.

Run:

    cd /Users/alansong/Desktop/FORDEEPSEEK/HGRE-Experiments
    python -m local_tool_runtime.run_trials --reps 3

Requires DEEPSEEK_API_KEY in the environment. No .env file is included.
