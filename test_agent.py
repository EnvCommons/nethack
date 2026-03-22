import asyncio
import json
import os

from openai import AsyncOpenAI
from nethack_env import NetHackEnvironment, StepParams, CommandsParams


def get_secrets():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    secrets = {}
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    secrets[key.strip().lower()] = val.strip()
    return secrets


async def run_agent_test(max_turns=50):
    secrets = get_secrets()
    oai_client = AsyncOpenAI(api_key=secrets.get("openai_api_key"))

    tasks = NetHackEnvironment.list_tasks(split="train")

    task = tasks[0]

    print(f"=== Agent Test: NetHack ===")
    print(f"Task: {task['id']}")

    env = NetHackEnvironment(task_spec=task, secrets=secrets)
    await env.setup()
    prompt = await env.get_prompt()

    tools = [
        {
            "type": "function",
            "name": "step",
            "description": (
                "Send a keystroke to NetHack and receive the updated game state. "
                "Use single characters like 'k' (north), 'e' (eat), ',' (pickup). "
                "For control keys use 'ctrl+d' (kick). For meta keys use 'meta+j' (jump). "
                "Use 'enter' for --More-- prompts and 'space' to dismiss menus."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keystroke": {
                        "type": "string",
                        "description": "The keystroke to send to NetHack",
                    }
                },
                "required": ["keystroke"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "commands",
            "description": "Show a reference of all available NetHack keystroke commands.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    ]

    input_list = [{"role": "user", "content": prompt[0].text}]
    finished = False
    turn = 0

    while not finished and turn < max_turns:
        turn += 1
        response = await oai_client.responses.create(
            model="gpt-5.2",
            tools=tools,
            input=input_list,
        )

        input_list += response.output

        for item in response.output:
            if item.type == "function_call":
                args = json.loads(str(item.arguments))

                if item.name == "step":
                    result = await env.step(StepParams(**args))
                elif item.name == "commands":
                    result = await env.commands(CommandsParams())
                else:
                    continue

                finished = result.finished
                reward = result.reward

                input_list.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": result.blocks[0].text,
                })
                print(input_list[-1]['output'])

                print(f"  Turn {turn}: {item.name}({args}) reward={reward:.2f} finished={finished}")

                if finished:
                    score = result.metadata.get("score", "?")
                    print(f"\n=== FINISHED! Final Score: {score} ===")
                    break

    if not finished:
        print(f"\n=== Hit max turns ({max_turns}) without finishing ===")

    await env.teardown()


if __name__ == "__main__":
    asyncio.run(run_agent_test())
