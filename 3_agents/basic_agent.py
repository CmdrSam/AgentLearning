from datetime import datetime
import json
import re
import os
from typing import Any, Dict, Literal

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
                "Pick exactly one tool and extract arguments from the user request.\n\n"
                "Tools:\n"
                '- time: get current time. args: {{}}\n'
                '- add: add two numbers. args: {{"a": <number>, "b": <number>}}\n'
                '- create_file: create a file. args: {{"filename": <string>, "content": <string>}}\n'
                "- exit: quit the program. args: {{}}\n\n"
                "Return ONLY valid JSON (no markdown, no extra text) with this shape:\n"
                '{{"tool": "time|add|create_file|exit|unknown", "args": {{}}, "reason": "<short>"}}\n'
                "If the request does not match a tool, use tool=unknown.",
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
        return {"tool": "unknown", "args": {}, "reason": f"Could not parse JSON: {raw[:200]}"}

    tool: ToolNames = data.get("tool", "unknown")
    args = data.get("args") if isinstance(data.get("args"), dict) else {}
    reason = data.get("reason", "")
    return {"tool": tool, "args": args, "reason": reason}


def respond_in_plain_english(
    model: ChatOpenAI,
    user_input: str,
    tool: ToolNames,
    args: Dict[str, Any],
    tool_output: str,
) -> str:
    response_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful CLI assistant.\n"
                "The system can call tools, and you must explain results to the user in plain English.\n"
                "Return only the final answer text (no JSON, no markdown).\n\n"
                "Tool call details:\n"
                "- tool: {tool}\n"
                "- args: {args}\n"
                "- tool_output: {tool_output}\n",
            ),
            ("human", "{user_input}"),
        ]
    )

    prompt_value = response_prompt.invoke(
        {
            "tool": tool,
            "args": json.dumps(args, ensure_ascii=False),
            "tool_output": tool_output,
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

        tool_call = interpret_user_input(model, user_input)
        print("Tool calls returned ", tool_call)
        tool = tool_call["tool"]
        args: Dict[str, Any] = tool_call["args"]

        if tool == "exit":
            break

        tool_output = ""
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
            tool_output = "Unknown command. Available tools: time, add, create_file, exit."

        print(respond_in_plain_english(model, user_input, tool, args, tool_output))




# def get_llm():
#     provider = "deepseek"
#     if provider == "ollama":
#         from langchain_ollama import ChatOllama
#         return ChatOllama(model="gemma3:4b", temperature=0)
#     else:
#         from langchain_openai import ChatOpenAI
#         return ChatOpenAI(
#             api_key=os.getenv("DEEPSEEK_API_KEY"),
#             base_url="https://api.deepseek.com",
#             model="deepseek-chat",
#             temperature=0,
#         )