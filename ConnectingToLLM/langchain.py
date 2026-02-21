from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
load_dotenv()

deepseek_API_KEY = os.getenv("DEEPSEEK_API_KEY")

model = ChatOpenAI(api_key=deepseek_API_KEY, base_url="https://api.deepseek.com", model="deepseek-chat")

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