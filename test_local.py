import asyncio

from nethack_env import (
    NetHackEnvironment,
    StepParams,
    CommandsParams,
    KEYSTROKE_TO_INDEX,
    INDEX_TO_INFO,
)


async def smoke_test():
    print("=== NetHack Environment Smoke Test ===\n")

    # 1. Check splits and tasks
    splits = NetHackEnvironment.list_splits()
    print(f"Splits: {splits}")
    assert splits == ["train"], f"Expected ['train'], got {splits}"

    train_tasks = NetHackEnvironment.list_tasks(split="train")
    print(f"Train tasks: {len(train_tasks)}")
    assert len(train_tasks) == 1000, f"Expected 1000 train tasks, got {len(train_tasks)}"

    # 2. Check action mapping coverage
    print(f"\nAction mapping: {len(KEYSTROKE_TO_INDEX)} unique keystrokes for {len(INDEX_TO_INFO)} actions")

    # 3. Create environment and setup
    task = train_tasks[0]
    print(f"\nUsing task: {task}")
    env = NetHackEnvironment(task_spec=task)
    await env.setup()

    # 4. Get prompt
    prompt = await env.get_prompt()
    assert len(prompt) == 1, f"Expected 1 TextBlock, got {len(prompt)}"
    assert "NetHack" in prompt[0].text, "Prompt should mention NetHack"
    assert "Screen:" in prompt[0].text, "Prompt should include screen"
    print(f"Prompt length: {len(prompt[0].text)} chars")
    print("Prompt preview (first 500 chars):")
    print(prompt[0].text[:500])
    print("...\n")

    # 5. Test commands tool
    cmd_result = await env.commands(CommandsParams())
    assert cmd_result.reward == 0.0
    assert cmd_result.finished is False
    assert "MOVEMENT" in cmd_result.blocks[0].text
    print(f"Commands tool: {len(cmd_result.blocks[0].text)} chars, OK")

    # 6. Test stepping with various keystrokes
    test_keys = [
        (".", "wait"),
        ("k", "move north"),
        ("i", "inventory"),
        (":", "look"),
        ("s", "search"),
    ]
    for keystroke, desc in test_keys:
        result = await env.step(StepParams(keystroke=keystroke))
        print(f"Step '{keystroke}' ({desc}): reward={result.reward:.2f}, finished={result.finished}, step={result.metadata.get('step')}")
        assert result.blocks[0].text, "Should have observation text"
        assert "Screen:" in result.blocks[0].text or "The game is already over" in result.blocks[0].text

    # 7. Test invalid keystroke
    result = await env.step(StepParams(keystroke="INVALID"))
    assert result.reward == 0.0
    assert result.finished is False
    assert "Unknown keystroke" in result.blocks[0].text
    print(f"Invalid keystroke: correctly rejected")

    # 8. Test ctrl and meta keystrokes exist in mapping
    assert "ctrl+d" in KEYSTROKE_TO_INDEX, "ctrl+d (kick) should be in mapping"
    assert "meta+j" in KEYSTROKE_TO_INDEX, "meta+j (jump) should be in mapping"
    assert "enter" in KEYSTROKE_TO_INDEX, "enter should be in mapping"
    assert "escape" in KEYSTROKE_TO_INDEX, "escape should be in mapping"
    assert "space" in KEYSTROKE_TO_INDEX, "space should be in mapping"
    print("Special key mappings: OK")

    # 9. Check observation format
    result = await env.step(StepParams(keystroke="."))
    obs_text = result.blocks[0].text
    assert "```" in obs_text, "Should have code fence for screen"
    assert "Stats:" in obs_text, "Should have stats line"
    print("Observation format: OK")

    await env.teardown()
    print("\n=== All smoke tests PASSED ===")


if __name__ == "__main__":
    asyncio.run(smoke_test())
