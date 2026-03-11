import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from pydantic import BaseModel

from openreward import AsyncOpenReward, SandboxBucketConfig, SandboxSettings
from openreward.environments import JSONObject, TextBlock, ToolOutput, tool, Split

from cli_environment import CLIEnvironment, BashParams
from prompts import INSTRUCTIONS
import os


logger = logging.getLogger(__name__)

if os.path.exists('/orwd_data'):
    DATASET_PATH = Path("/orwd_data") / "dataset.csv"
else:
    DATASET_PATH = Path(__file__).parent / "dataset.csv"


class TaskSpec(BaseModel, extra="forbid"):
    task_id: int
    dataset: str
    question: str
    answer: str
    max_response_length: int | None = None


class ResponseInput(BaseModel, extra="forbid"):
    answer: str


class DSBC(CLIEnvironment):
    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec, secrets)
        self.validated = TaskSpec.model_validate(task_spec)

        api_key = secrets.get("api_key")
        if not api_key:
            raise ValueError("OpenReward API key is required")

        self.sandbox_settings = SandboxSettings(
            environment="GeneralReasoning/DSBC",
            image="generalreasoning/python-ds:3.12-tools",
            machine_size="0.5:1",
            block_network=False,
            bucket_config=SandboxBucketConfig(
                mount_path="/workspace",
                read_only=True,
                only_dir="files",
            ),
        )

        or_client = AsyncOpenReward(api_key=api_key)
        self.sandbox = or_client.sandbox(self.sandbox_settings)

    async def setup(self) -> None:
        await super().setup()

        # Create working directory and copy the relevant dataset CSV from the mounted bucket
        await self.sandbox.run("mkdir -p /workdir")
        dataset = self.validated.dataset
        dataset_name = f"{dataset.split(' ')[0]}_TRAIN.csv"
        await self.sandbox.run(f"cp /workspace/{dataset_name} /workdir/{dataset_name}")

    @tool
    async def bash(self, params: BashParams) -> ToolOutput:
        """Execute a bash command."""
        try:
            cmd = f"cd /workdir && {params.command.strip()}"
            output, code = await self.sandbox.run(cmd)
            max_len = self.validated.max_response_length

            if isinstance(max_len, int) and len(output) > max_len:
                output = f"...(truncated)\n{output[-max_len:]}"

            return ToolOutput(
                blocks=[TextBlock(text=f"{output}\n\n(exit {code})")],
                metadata={"output": output, "exit_code": code},
                reward=0.0,
                finished=False,
            )
        except Exception as e:
            return ToolOutput(
                metadata={"error": str(e)},
                blocks=[TextBlock(text=f"Error executing command: {str(e)}")],
                finished=False,
            )

    @tool
    async def answer(self, params: ResponseInput) -> ToolOutput:
        """
        Use this tool to provide your final answer to the given question. If the question specified a format of how you
        should format your answer, use that format.
        """
        def _clean_string(s: str) -> str:
            s = s.replace("%", "")
            s = s.replace("$", "")
            s = s.strip(".,!?")
            s = s.lower()
            s = s.replace(" ", "")
            s = s.replace("\n", "")
            s = s.strip()
            return s

        agent_answer = _clean_string(params.answer)
        correct_answer = _clean_string(self.validated.answer)

        try:
            reward = 1.0 if np.isclose(float(agent_answer), float(correct_answer), rtol=0.01) else 0.0
        except ValueError:
            reward = 1.0 if agent_answer == correct_answer else 0.0

        result_text = "Correct!" if reward == 1.0 else "Incorrect."
        return ToolOutput(
            blocks=[TextBlock(text=result_text)],
            metadata={"is_correct": reward, "cleaned_agent_answer": agent_answer, "cleaned_correct_answer": correct_answer},
            reward=reward,
            finished=True,
        )

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
                "answer": row["Answer_Rewritten"],
            })
        return tasks

    @classmethod
    def list_splits(cls) -> list[str]:
        return [Split(name="train", type="train")]
