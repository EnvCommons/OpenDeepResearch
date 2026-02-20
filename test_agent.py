"""
Test agent for OpenDeepResearch environment using modern Responses API.

This script demonstrates how to:
1. Connect to the environment
2. Run a research task with web search and content fetching
3. Submit a comprehensive report
4. Receive ArenaRL 7-dimension grading

Usage:
    export OPENAI_API_KEY="sk-..."
    export TAVILY_API_KEY="tvly-..."
    export MODEL_NAME="gpt-5.2"  # optional, defaults to gpt-5.2
    python test_agent.py
"""

import asyncio
import json
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward


async def main():
    # Configuration
    MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5.2")
    ENV_NAME = "local/OpenDeepResearch"
    BASE_URL = "http://localhost:8080"
    SPLIT = "train"  # Test with train split (open-ended, no references)

    # API keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    if not OPENAI_API_KEY or not TAVILY_API_KEY:
        raise ValueError(
            "Set OPENAI_API_KEY and TAVILY_API_KEY environment variables.\n"
            "Example: export OPENAI_API_KEY='sk-...' TAVILY_API_KEY='tvly-...'"
        )

    # Initialize clients
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Connect to environment
    environment = or_client.environments.get(name=ENV_NAME, base_url=BASE_URL)
    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} tasks in {SPLIT} split")

    # Test first task
    task = tasks[0]
    print(f"\n{'='*80}")
    print(f"{'='*80}\n")

    finished = False

    async with environment.session(
        task=task,
        secrets={
            "openai_api_key": OPENAI_API_KEY,
            "tavily_api_key": TAVILY_API_KEY
        }
    ) as session:
        prompt = await session.get_prompt()
        input_list = [{"role": "user", "content": prompt[0].text}]

        turn = 0
        max_turns = 30  # Allow more turns for research

        while not finished and turn < max_turns:
            turn += 1
            print(f"\n--- Turn {turn} ---")

            # Use modern Responses API
            response = await oai_client.responses.create(
                model=MODEL_NAME,
                tools=tools,
                input=input_list,
            )

            input_list += response.output

            for item in response.output:
                if item.type == "function_call":
                    print(f"🛠️  Tool: {item.name}")

                    if item.name == "web_search":
                        args = json.loads(str(item.arguments))
                        print(f"   Query: {args.get('query', '')}")
                    elif item.name == "fetch_url":
                        args = json.loads(str(item.arguments))
                        print(f"   URL: {args.get('url', '')}")
                    elif item.name == "submit_report":
                        args = json.loads(str(item.arguments))
                        print(f"   Report length: {len(args.get('report', ''))} chars")
                        print(f"   Key findings: {len(args.get('key_findings', []))} items")
                        print(f"   Sources: {len(args.get('sources_cited', []))} URLs")

                    # Call tool through environment session
                    tool_result = await session.call_tool(
                        item.name,
                        json.loads(str(item.arguments))
                    )

                    finished = tool_result.finished

                    # Add tool output to input list
                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": tool_result.blocks[0].text if tool_result.blocks else ""
                    })

                    if item.name == "submit_report":
                        print(f"\n📊 Final Reward: {tool_result.reward:.2f}")
                        if tool_result.blocks:
                            print(f"\nGrading Results:\n{tool_result.blocks[0].text}")
                        if tool_result.metadata:
                            scores = tool_result.metadata.get('dimension_scores', {})
                            print(f"\nDimension Scores:")
                            for dim, score in scores.items():
                                print(f"  {dim}: {score:.2f}")

                    if finished:
                        print(f"\n✅ Task completed!")
                        break

        if turn >= max_turns:
            print(f"\n⚠️  Reached max turns ({max_turns})")

    print(f"\n{'='*80}")
    print("Test completed successfully!")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
