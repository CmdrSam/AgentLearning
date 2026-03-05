from datetime import datetime
import json
import re
import os
from typing import Any, Dict, List, Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def get_llm(name: str):
    if name == "deepseek":
        return ChatOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            temperature=0,
        )
    elif name == "ollama":
        return ChatOllama(model="gemma3:4b", temperature=0)


class SimpleTools:
    def __init__(self):
        pass

    def get_current_time(self):
        current_time = datetime.now().strftime("%H:%M:%S")
        return f"The current time is {current_time}."

    def add_numbers(self, a, b):
        try:
            return f"The sum of {a} and {b} is {float(a) + float(b)}."
        except Exception as e:
            return f"Error: {e}"

    def create_file(self, filename, content):
        try:
            with open(filename, "w") as f:
                f.write(content)
            return f"File '{filename}' created successfully."
        except Exception as e:
            return f"Failed to create file '{filename}': {e}"


ToolNames = Literal["time", "add", "create_file", "exit", "unknown"]


def _extract_json_object(text: str) -> str:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z]*\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)

    match = re.search(r"\{[\s\S]*\}", candidate)
    return match.group(0) if match else candidate


def interpret_user_input(model: ChatOpenAI, user_input: str) -> Dict[str, Any]:
    router_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a command router for a CLI agent.\n"
                "You may plan and call multiple tools in sequence, deciding the best order based on the user request.\n"
                "For each step, choose exactly one tool and its arguments.\n\n"
                "Tools:\n"
                '- time: get current time. args: {{}}\n'
                '- add: add two numbers. args: {{"a": <number>, "b": <number>}}\n'
                '- create_file: create a file. args: {{"filename": <string>, "content": <string>}}\n'
                "- exit: quit the program. args: {{}}\n\n"
                "Return ONLY valid JSON (no markdown, no extra text) with this shape:\n"
                '{{"steps": [{{"tool": "time|add|create_file|exit|unknown", "args": {{}}, "reason": "<short>"}}]}}\n'
                "If the request does not match any tool, return one step with tool=unknown.\n"
                "Use multiple steps when the user asks for several actions or when later tools need outputs from earlier ones.",
            ),
            ("human", "{user_input}"),
        ]
    )

    prompt_value = router_prompt.invoke({"user_input": user_input})
    response = model.invoke(prompt_value)

    print("LLM Response: ", response)

    raw = getattr(response, "content", "") or ""
    try:
        data = json.loads(_extract_json_object(raw))
    except Exception:
        return {
            "steps": [
                {
                    "tool": "unknown",
                    "args": {},
                    "reason": f"Could not parse JSON: {raw[:200]}",
                }
            ]
        }

    
    steps: List[Dict[str, Any]] = []

    if isinstance(data, dict) and "steps" in data and isinstance(data["steps"], list):
        raw_steps = data["steps"]
    else:
        raw_steps = [data]

    for s in raw_steps:
        if not isinstance(s, dict):
            continue
        tool: ToolNames = s.get("tool", "unknown")
        args = s.get("args") if isinstance(s.get("args"), dict) else {}
        reason = s.get("reason", "")
        steps.append({"tool": tool, "args": args, "reason": reason})

    if not steps:
        steps = [{"tool": "unknown", "args": {}, "reason": "No valid steps returned"}]

    return {"steps": steps}


def respond_in_plain_english(
    model: ChatOpenAI,
    user_input: str,
    tools_results: List[Dict[str, Any]],
) -> str:
    response_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful CLI assistant.\n"
                "The system can call tools, potentially multiple times in sequence, and you must explain the combined results to the user in plain English.\n"
                "Return only the final answer text (no JSON, no markdown).\n\n"
                "Tool call details are provided as a JSON list, in execution order. Each item has:\n"
                '- "tool": tool name\n'
                '- "args": arguments passed\n'
                '- "reason": why the tool was chosen\n'
                '- "tool_output": raw output from the tool\n\n'
                "Use this information to summarize what happened and answer the user clearly.",
            ),
            ("human", "{user_input}"),
        ]
    )

    prompt_value = response_prompt.invoke(
        {
            "tools_results": json.dumps(tools_results, ensure_ascii=False),
            "user_input": user_input,
        }
    )
    response = model.invoke(prompt_value)
    return (getattr(response, "content", "") or "").strip()


if __name__ == "__main__":

    model = get_llm("deepseek")
    tools = SimpleTools()

    while True:
        user_input = input("Enter your command (or type 'exit' to quit): ").strip()
        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            break

        tool_plan = interpret_user_input(model, user_input)
        print("Tool plan returned ", tool_plan)

        steps: List[Dict[str, Any]] = tool_plan.get("steps", [])
        tools_results: List[Dict[str, Any]] = []

        exit_requested = False

        for step in steps:
            tool = step.get("tool", "unknown")
            args: Dict[str, Any] = (
                step.get("args") if isinstance(step.get("args"), dict) else {}
            )
            reason = step.get("reason", "")

            if tool == "exit":
                exit_requested = True
                tools_results.append(
                    {
                        "tool": tool,
                        "args": args,
                        "reason": reason,
                        "tool_output": "Exiting as requested.",
                    }
                )
                break

            if tool == "time":
                tool_output = tools.get_current_time()
            elif tool == "add":
                tool_output = tools.add_numbers(args.get("a"), args.get("b"))
            elif tool == "create_file":
                filename = args.get("filename")
                content = args.get("content")
                if not filename or content is None:
                    tool_output = "Invalid arguments. Expected filename and content."
                else:
                    tool_output = tools.create_file(str(filename), str(content))
            else:
                tool_output = (
                    "Unknown command. Available tools: time, add, create_file, exit."
                )

            tools_results.append(
                {
                    "tool": tool,
                    "args": args,
                    "reason": reason,
                    "tool_output": tool_output,
                }
            )

        if exit_requested:
            print(respond_in_plain_english(model, user_input, tools_results))
            break

        print(respond_in_plain_english(model, user_input, tools_results))

