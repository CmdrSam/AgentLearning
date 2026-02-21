import requests

def query_ollama(model: str, prompt: str, stream: bool = False):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response") or data  # show the full JSON if 'response' key is not present
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    model_name = "gemma3:4b"
    prompt_text = ''
    while(prompt_text != "exit" or prompt_text != "bye"):
        if "history" not in locals():
            history = []
        prompt_text = input("Enter a prompt: ").strip()
        history.append({"role": "user", "content": prompt_text})
        context = ""
        for msg in history:
            context += f"{msg['role']}: {msg['content']}\n"
        print(f"Querying local LLM with model '{model_name}' and context:\n{context}")
        output = query_ollama(model_name, context.strip())
        print("Response:")
        print(output)
        history.append({"role": "assistant", "content": str(output)})


