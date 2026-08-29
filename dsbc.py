import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from pydantic import BaseModel

from openreward import AsyncOpenReward, SandboxBucketConfig, SandboxSettings
from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool, Split
from openreward.toolsets import CLIToolset

from prompts import INSTRUCTIONS
import os
import re

import openai


logger = logging.getLogger(__name__)

if os.path.exists('/orwd_data'):
    DATASET_PATH = Path("/orwd_data") / "dataset.csv"
else:
    DATASET_PATH = Path(__file__).parent / "datasets" / "dataset.csv"


# Default LLM judge model for grading non-numeric answers (overridable via env).
DEFAULT_GRADER_MODEL = os.getenv("DSBC_GRADER_MODEL", "gpt-5.6-luna")

_FLOAT_RE = re.compile(r'[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?')

# Reward for a submission made after the task has already been graded. Negative
# so repeat submissions are actively discouraged, not merely left unscored.
REPEAT_SUBMISSION_PENALTY = -0.1



def _to_float(s: str) -> float | None:
    """Parse a scalar answer to a float, tolerating %/$ and thousands separators."""
    s = str(s).replace("%", "").replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _last_float(s: str) -> float | None:
    """Return the last number written anywhere in the model answer. Returns None if
    the answer contains no number at all, so those answers fall through to the LLM
    grader."""
    s = str(s).replace("%", "").replace("$", "")
    matches = _FLOAT_RE.findall(s)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


class TaskSpec(BaseModel, extra="forbid"):
    task_id: int
    dataset: str
    question: str
    answer: str
    max_response_length: int | None = None


class ResponseInput(BaseModel, extra="forbid"):
    answer: str


class DSBC(Environment):
    # 9-tool sandboxed CLI surface provided by the SDK. The class attribute
    # makes the framework auto-instantiate the toolset against self.sandbox.
    toolsets = [CLIToolset]

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.validated = TaskSpec.model_validate(task_spec)

        api_key = secrets.get("api_key")
        if not api_key:
            raise ValueError("OpenReward API key is required")

        # LLM grader client, used only for non-numeric answers. Built with no
        # base_url so an OPENAI_BASE_URL override (e.g. infer.gr.inc) is picked up
        # from the environment. Left None if the secret isn't supplied; numeric
        # tasks still grade without it, and _llm_grade raises if it's actually needed.
        grader_key = secrets.get("openai_api_key")
        self.grader_client = openai.AsyncClient(api_key=grader_key) if grader_key else None

        dataset_short = self.validated.dataset.split(' ')[0]

        self.sandbox_settings = SandboxSettings(
            environment="GeneralReasoning/DSBC",
            image="generalreasoning/dsbc_tools:latest",
            machine_size="0.5:1",
            block_network=False,
            bucket_config=SandboxBucketConfig(
                mount_path="/workspace",
                read_only=True,
                only_dir=f"files/{dataset_short}",
            ),
        )

        or_client = AsyncOpenReward(api_key=api_key)
        self.sandbox = or_client.sandbox(self.sandbox_settings)

        # Answers submitted this session. Only the first is graded and rewarded:
        # `answer` reports Correct!/Incorrect., so an uncapped tool would let the
        # agent probe the verdict repeatedly until it guessed right (and, on the
        # free numeric path, search for the gold value inside the 1% tolerance).
        self.submitted = 0

    async def setup(self) -> None:
        await self.sandbox.start()

    async def teardown(self) -> None:
        await self.sandbox.stop()

    @tool
    async def answer(self, params: ResponseInput) -> ToolOutput:
        """
        Use this tool to provide your final answer to the given question. If the question specified a format of how you
        should format your answer, use that format.
        """
        if self.submitted > 0:
            return ToolOutput(
                blocks=[TextBlock(
                    text="An answer has already been submitted for this task. "
                         "This episode is over: the answer is not re-graded, and repeat submissions are penalised (reward -0.1).",
                )],
                metadata={
                    "already_submitted": True,
                    "submission_count": self.submitted,
                },
                reward=REPEAT_SUBMISSION_PENALTY,
                finished=True,
            )

        gold = self.validated.answer
        gold_num = _to_float(gold)
        model_num = _last_float(params.answer)

        if gold_num is not None and model_num is not None:
            # Numeric task and the model answer contains a number: compare the last
            # float in the answer to gold. Numeric comparison unchanged from the original.
            reward = 1.0 if np.isclose(model_num, gold_num, rtol=0.01) else 0.0
            method = "numeric"
        else:
            # Non-numeric (symbolic/mixed) gold, or no float in the model's answer:
            # defer to an LLM grader with the question + gold + model answer.
            reward = await self._llm_grade(self.validated.question, gold, params.answer)
            method = "llm"

        result_text = "Correct!" if reward == 1.0 else "Incorrect."

        # Incremented only after grading succeeds, so an infra failure in
        # _llm_grade (which deliberately propagates) leaves the attempt retryable.
        self.submitted += 1

        return ToolOutput(
            blocks=[TextBlock(text=result_text)],
            metadata={
                "is_correct": reward,
                "grade_method": method,
                "gold_answer": gold,
                "model_last_float": model_num,
            },
            reward=reward,
            finished=True,
        )

    async def _llm_grade(self, question: str, gold: str, model_answer: str) -> float:
        """LLM grader for non-numeric answers. Deliberately does NOT catch grader
        failures: an infra error (auth/network) propagates so the platform retries
        and, if it still fails, treats the call as terminal with a blank reward —
        rather than fabricating a (wrong) score."""
        if self.grader_client is None:
            raise ValueError(
                "openai_api_key required in secrets to grade non-numeric answers"
            )
        prompt = (
            "You are grading an answer to a data-science question.\n\n"
            f"Question:\n{question}\n\n"
            f"Reference (gold) answer:\n{gold}\n\n"
            f"Model answer:\n{model_answer}\n\n"
            "The model answer is correct if it conveys the same result as the gold "
            "answer, ignoring formatting, phrasing, unit placement, and any extra "
            "explanation.\n\n"
            "Do your reasoning first in <reasoning></reasoning> brackets, and put your "
            "answer (Correct!/Incorrect.) in <answer></answer> tags."
        )
        # NOTE: temperature is intentionally left at the API default. Some grader
        # models (e.g. gpt-5.6-luna) only support the default temperature (1) and
        # reject temperature=0 with a 400, which would fail every non-numeric grade.
        response = await self.grader_client.chat.completions.create(
            model=DEFAULT_GRADER_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content or ""
        # Parse the verdict from inside the <answer></answer> tags; fall back to the
        # raw text if the grader didn't emit the tags.
        m = re.search(r"<answer>(.*?)</answer>", text, re.S | re.I)
        verdict = (m.group(1) if m else text).strip().lower()
        return 1.0 if verdict.startswith("correct") else 0.0

    async def get_prompt(self) -> List[TextBlock]:
        return [TextBlock(text=INSTRUCTIONS + "\n\n" + self.validated.question)]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split != "train":
            return []
        df = pd.read_csv(DATASET_PATH)

        tasks = []
        for idx, row in df.iterrows():
            tasks.append({
                "task_id": idx,
                "dataset": row["Dataset"],
                "question": row["Question_Rewritten"],
                "answer": row["Response_Expected"],
            })
        return tasks

    @classmethod
    def list_splits(cls) -> list[str]:
        return [Split(name="train", type="train")]
