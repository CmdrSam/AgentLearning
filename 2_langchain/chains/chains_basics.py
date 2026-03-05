from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import os
from langchain_core.runnables import RunnableLambda, RunnableSequence

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
        return ChatOllama(model="gemma3:4b", temperature=0.0)

model = get_llm("ollama")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert about java language. Please give tips for {topic} in one line only."),
        ("human", "Tell me {tip_number} tips."),
    ]
)

chain = prompt_template | model | StrOutputParser()

result = chain.invoke({"topic": "java", "tip_number": 3})

print("Simple chain result: \n", result)

################################################


format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

response = chain.invoke({"topic": "java", "tip_number": 3})

print(response)


################################################

uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

result = chain.invoke({"topic": "java", "tip_number": 3})

print(result)