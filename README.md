# ActuarialBench

**Reproducible Benchmark of Frontier LLM Reliability in Actuarial Science**

ActuarialBench is a small, reproducible empirical study of how black-box language-model endpoints solve and validate actuarial tasks. It prioritizes independently computed ground truth, paired comparisons, preserved raw responses, and explicit uncertainty over a single leaderboard.

## Scope

The benchmark contains a cheap eight-task vertical slice plus a deterministic 48-task bank (six seeded instances per family) across life contingencies, non-life pricing, reserving, risk, survival, statistics, validation, and missing-information/hallucination detection. It includes scalar and component-wise numerical grading, restricted code execution, and exact validation defect scoring.

Configured routes are:

| Label | Provider route | Identity note |
| --- | --- | --- |
| `gpt-5.6-sol` | AgentRouter MCP | externally asserted route identity |
| `deepseek-v4-flash` | AgentRouter MCP | externally asserted route identity |
| `claude-opus-5` | AgentRouter MCP | externally asserted route identity |
| `x0alpha` | OpenRouter `stealth/ox-alpha` | provider catalog ID |

AgentRouter is a tool-routing endpoint, not a documented direct model API. This is recorded as a protocol limitation; the project does not claim that the AgentRouter labels are independently verified model identities.

## Reproduce locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

Copy `.env.example` to `.env` and export the corresponding variables in the shell. Never copy keys from `Desktop\apis.txt` into source or commit history.

Run the bounded smoke benchmark (one repetition, eight tasks, four configured routes; 20-second timeout and no retries):

```powershell
python run_benchmark.py --smoke
```

For the smallest live route probe, use one representative task:

```powershell
python run_benchmark.py --smoke --tasks life_survival_5y
```

Analyze an experiment without overwriting it:

```powershell
python analyze_results.py results/raw/<experiment-id>
```

The full protocol is available but is not launched automatically:

```powershell
python run_benchmark.py --repetitions 3
```

## Methodological controls

- Task text, system prompt, order, temperature, grading, and ground truth are shared.
- Every experiment writes a unique timestamp/UUID directory containing `manifest.json`, `responses.jsonl`, `scores.jsonl`, and later `analysis.json`.
- Task and prompt hashes, seeds, model route metadata, and Git commit are recorded.
- Provider/API failures are separate from mathematical and schema failures.
- Comparisons are paired by task and repetition, with bootstrap intervals, paired sign-permutation tests, McNemar p-values, paired Cohen's d, and Holm adjustment.
- Capability metrics exclude API/timeout failures; route availability is reported separately.
- No LLM-as-judge score is used for the primary metrics.
- Model-generated code is checked with an AST allowlist and executed in an isolated Python subprocess with a short timeout. This is restricted execution, not a hardened security sandbox.

## Major audit risks

1. AgentRouter can change or hide the underlying model route, so model-level attribution is weaker than the OpenRouter catalog route.
2. A combined single-text prompt is used because AgentRouter does not document a system-role equivalent; this is recorded as a lowest-common-denominator limitation.
3. The 48-task bank is an initial portfolio study, not enough evidence for a production model-risk conclusion without review of task quality and a frozen experiment manifest.
4. Cost is reported only when the provider supplies it; unknown pricing is never invented.
5. Restricted subprocess execution reduces risk but is not equivalent to container isolation. Do not run untrusted benchmark code on sensitive hosts without stronger isolation.

## Report

`report/main.tex` is a single research-style report shell. Generated tables and figures are written under `report/generated/` and `figures/` after an experiment is analyzed. The report generator produces overall/domain accuracy, critical-error and hallucination proxies, calibration, paired differences, and accuracy-vs-cost/latency figures when those provider fields are available.

For a deterministic local structural check without API calls, run `pytest`. For the full task bank without a live experiment, use `python run_benchmark.py --repetitions 3 --domains pricing,risk` only after exporting provider keys; no full benchmark is launched by the repository automatically.
