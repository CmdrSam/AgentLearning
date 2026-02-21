import requests
from dotenv import load_dotenv
load_dotenv()

def query_deepseek(model: str, history: list):
    import os

    url = "https://api.deepseek.com/v1/chat/completions"
    api_key = os.getenv("DEEPSEEK_API_KEY") 
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": history,
        "temperature": 0.7
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("response") or data  # show the full JSON if 'response' key is not present
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    model_name = "deepseek-chat"
    
    history = []
    prompt_text = ''
    while(True):        
        prompt_text = input("Enter a prompt: ").strip()
        if (prompt_text == "exit" or prompt_text == "bye"):
            break
        history.append({"role": "user", "content": prompt_text})
        
        context = ""
        for msg in history:
            context += f"{msg['role']}: {msg['content']}\n"
        
        output = query_deepseek(model_name, history)
        print("Response: ", output['choices'][0]['message']['content'])

        history.append({"role": "assistant", "content": str(output['choices'][0]['message']['content'])})

