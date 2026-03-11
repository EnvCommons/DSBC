import json
import asyncio
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward


async def main():
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI()

    MODEL_NAME = "gpt-5.2"
    ENV_NAME = "GeneralReasoning/dsbc"
    SPLIT = "train"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    environment = or_client.environments.get(name=ENV_NAME, base_url="http://localhost:8080")
    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} tasks")

    for task in tasks[:1]:  # Test first task
        async with environment.session(
            task=task,
            secrets={"api_key": os.getenv("OPENREWARD_API_KEY", ""), "openai_api_key": OPENAI_API_KEY},
        ) as session:
            prompt = await session.get_prompt()
            input_list = [{"role": "user", "content": prompt[0].text}]
            finished = False

            while not finished:
                response = await oai_client.responses.create(
                    model=MODEL_NAME,
                    tools=tools,
                    input=input_list,
                )

                input_list += response.output

                for item in response.output:
                    if item.type == "function_call":
                        tool_result = await session.call_tool(
                            item.name, json.loads(str(item.arguments))
                        )

                        reward = tool_result.reward
                        finished = tool_result.finished

                        input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": tool_result.blocks[0].text,
                        })

                        print(f"Tool: {item.name}")
                        print(f"Reward: {reward:.3f}")

                        if tool_result.finished:
                            finished = True
                            print("FINISHED!")
                            break


if __name__ == "__main__":
    asyncio.run(main())
