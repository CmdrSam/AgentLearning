from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
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

model = get_llm("ollama")

history = []

while True:
    user_input = input("Enter your prompt (type 'exit' or 'bye' to quit): ").strip()
    if user_input.lower() in ["exit", "bye"]:
        break
    history.append({"role": "user", "content": user_input})

    context = ""
    for msg in history:
        context += f"{msg['role']}: {msg['content']}\n"

    # Pass the context to the model
    result = model.invoke(context.strip())

    print("Full result:")
    print(result)
    print("Content only:")
    print(result.content)

    history.append({"role": "assistant", "content": result.content})
print("Full result:")
print(result)
print("Content only:")
print(result.content)