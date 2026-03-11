# DSBC

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/DSBC)

## Description

DSBC (Data Science task Benchmarking with Context engineering) evaluates language model agents on real-world data science tasks across 11 domains. Agents are given a dataset CSV and a natural language question, then must write and execute Python code to derive the answer. Rewards are programmatically verified via exact match or numeric tolerance. Based on the [DSBC benchmark](https://arxiv.org/abs/2507.23336) by Kadiyala et al.

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

This is a sparse, verifiable reward environment. Rewards are issued only when the agent submits a final answer:

- **Binary**: 1.0 for correct, 0.0 for incorrect
- **Numeric answers**: compared with `numpy.isclose(rtol=0.01)` (1% relative tolerance)
- **String answers**: exact match after normalization (lowercase, strip whitespace, remove `%`, `$`, punctuation)
- No LLM graders are used

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

DSBC requires an OpenReward API key for sandbox provisioning. No other external API keys are needed.

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
