# ActuarialBench

**Reproducible Benchmark of Frontier LLM Reliability in Actuarial Science**

ActuarialBench is a small, reproducible empirical study of how black-box language-model endpoints solve and validate actuarial tasks. It prioritizes independently computed ground truth, paired comparisons, preserved raw responses, and explicit uncertainty over a single leaderboard.

## Scope

The benchmark contains a cheap eight-task vertical slice plus a deterministic 48-task bank (six seeded instances per family) across life contingencies, non-life pricing, reserving, risk, survival, statistics, validation, and missing-information/hallucination detection. It includes scalar and component-wise numerical grading, restricted code execution, and exact validation defect scoring.

Configured routes are:

| Label | Provider route | Identity note |
| --- | --- | --- |
| `gpt-5.6-sol` | AgentRouter OpenAI Responses API | externally asserted route identity |
| `deepseek-v4-flash` | AgentRouter OpenAI Responses API | externally asserted route identity |
| `claude-opus-5` | AgentRouter Anthropic Messages API | externally asserted route identity |
| `glm-5.3` | AgentRouter OpenAI Chat Completions API | externally asserted route identity |

The GPT and DeepSeek routes use AgentRouter's Codex-compatible `/v1/responses` transport, GLM uses `/v1/chat/completions`, and Claude uses the Claude Code-compatible `/v1/messages` transport. The adapter sends transparent compatible client identifiers because AgentRouter rejects the generic benchmark user agent. AgentRouter remains a provider-controlled routing layer, so labels are externally asserted rather than independently verified model identities.

DeepSeek and GLM use low reasoning effort so their internal reasoning does not consume the full 2,000-token response budget before producing a gradeable answer.

## Reproduce locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

Copy `.env.example` to `.env` and export the corresponding variables in the shell. Each route uses its own AgentRouter key with Codex-compatible access. Never copy keys into source or commit history.

Run the bounded smoke benchmark (one repetition, eight tasks, four configured routes; 60-second timeout and one bounded retry):

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

1. AgentRouter can change or hide the underlying model route, so model-level attribution is externally asserted.
2. OpenAI-compatible routes use the same combined text prompt, while Claude preserves separate system and user messages. Provider routing and model availability can change.
3. AgentRouter direct API authentication is provider-controlled; rejected keys are recorded as API failures and excluded from capability scores.
4. The 48-task bank is an initial portfolio study, not enough evidence for a production model-risk conclusion without review of task quality and a frozen experiment manifest.
5. Cost is reported only when the provider supplies it; unknown pricing is never invented.
6. Restricted subprocess execution reduces risk but is not equivalent to container isolation. Do not run untrusted benchmark code on sensitive hosts without stronger isolation.

## Report

`report/main.tex` is a single research-style report shell. Generated tables and figures are written under `report/generated/` and `figures/` after an experiment is analyzed. The report generator produces overall/domain accuracy, critical-error and hallucination proxies, calibration, paired differences, and accuracy-vs-cost/latency figures when those provider fields are available.

For a deterministic local structural check without API calls, run `pytest`. For the full task bank without a live experiment, use `python run_benchmark.py --repetitions 3 --domains pricing,risk` only after exporting provider keys; no full benchmark is launched by the repository automatically.
