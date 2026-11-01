# LeetCode runtime experiment

Paper experiment: LeetCode runtime prediction.

- data/leetcode_full.jsonl: raw problem records.
- results/trials.jsonl: 3,042 measured trials.
- results/strategies.jsonl: counterfactual policy replay.
- results/llm_prior_cache.json: cached LLM priors.

The paper subset is the first 543 question IDs in ascending order:
2,622 trials, 549 cold-start, and 2,073 warm-start records.

Run:

    cd /Users/alansong/Desktop/FORDEEPSEEK/HGRE-Experiments
    python -m leetcode_runtime.run_trials --reps 5

Requires DEEPSEEK_API_KEY in the environment. No .env file is included.
