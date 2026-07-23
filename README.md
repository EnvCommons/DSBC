# DSBC

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/DSBC)

## Description

DSBC (Data Science task Benchmarking with Context engineering) evaluates language model agents on real-world data science tasks across 11 domains. Agents are given a dataset CSV and a natural language question, then must write and execute Python code to derive the answer. Numeric answers are graded programmatically with a numeric tolerance; non-numeric answers are graded by an **LLM judge** (default model `gpt-5.6-luna`). Based on the [DSBC benchmark](https://arxiv.org/abs/2507.23336) by Kadiyala et al.

## Capabilities

- Exploratory data analysis with pandas
- Statistical computation (correlation, distribution analysis, feature engineering)
- Data parsing and pre-processing
- Writing and executing Python code in a sandboxed environment
- Interpreting natural language questions about tabular data

## Compute Requirements

Agents are given a sandbox with 1GB of RAM and 0.5 CPUs, with pandas pre-installed.

## Tasks

There are 303 tasks in a single training split, spanning 11 datasets:

| Dataset | Tasks |
|---------|-------|
| Stocks | 45 |
| AQI (Air Quality Index) | 36 |
| Sales | 34 |
| COVID | 33 |
| Production | 29 |
| Weather | 25 |
| Inflation | 24 |
| Population | 21 |
| Power | 20 |
| Insurance | 18 |
| Life | 18 |

Tasks cover categories including statistics, correlation analysis, data parsing, feature engineering, data pre-processing, distribution analysis, and data visualization.

## Reward Structure

This is a sparse, verifiable reward environment. Rewards are earned only when the agent submits a final answer:

- **Binary**: 1.0 for correct, 0.0 for incorrect
- **Numeric answers**: the last number in the model's answer is compared with the gold value via `numpy.isclose(rtol=0.01)` (1% relative tolerance)
- **Non-numeric answers** (or answers containing no number): graded by an **LLM judge** — default model **`gpt-5.6-luna`** — which compares the submission to the gold reference, ignoring formatting, phrasing, unit placement, and extra explanation, and returns CORRECT/INCORRECT
- The LLM judge client is built from the `openai_api_key` secret with no explicit `base_url`, so `OPENAI_BASE_URL` can route it to another OpenAI-compatible endpoint. Grader infra failures (auth/network) **propagate** — they are never scored as 0

## Data

Each task is associated with one of 11 CSV datasets covering domains such as stock prices, air quality, insurance, weather, and COVID statistics. The relevant dataset is copied into the agent's working directory at task start.

## Tools

Agents have access to CLI tools for exploring and manipulating files:

- **bash**: Execute shell commands (with pandas available)
- **read**, **write**, **edit**, **multi_edit**: File operations
- **glob**, **grep**, **ls**: File search and directory listing
- **todo_write**: Task planning
- **answer**: Submit final answer (triggers grading)

## Time Horizon

DSBC is a multi-turn environment. Agents typically explore the dataset, write Python analysis code, execute it, and submit an answer.

## Environment Difficulty

Performance varies by task category. Statistical and data parsing tasks tend to be more straightforward, while feature engineering and distribution analysis tasks require deeper reasoning.

## Other Environment Requirements

DSBC requires an OpenReward API key for sandbox provisioning, plus an `openai_api_key` secret for the LLM judge that grades non-numeric answers (default model `gpt-5.6-luna`; point it at any OpenAI-compatible endpoint via `OPENAI_BASE_URL`). Tasks whose answer is numeric grade without it.

## Safety

Agents operate in a sandboxed environment with read-only access to source data. Network access is enabled to allow package installation if needed. The environment does not interact with external systems or real-world data beyond the provided CSV files.

## Citations

```bibtex
@article{kadiyala2025dsbc,
  title={{DSBC}: Data Science task Benchmarking with Context engineering},
  author={Kadiyala, Ram Mohan Rao and Gupta, Siddhant and Purbey, Jebish and Martini, Giulio and Shafique, Ali and Debnath, Suman and Farooq, Hamza},
  journal={arXiv preprint arXiv:2507.23336},
  year={2025},
  url={https://arxiv.org/abs/2507.23336}
}
```
