from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool

load_dotenv()

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together"""
    
    return first_int*second_int

llm = AzureChatOpenAI(
    temperature = 0,
    deployment_name = "iomega-gpt-4",
    max_tokens=4000
)

llm_with_tools = llm.bind_tools([multiply])

message = llm_with_tools.invoke("what's 5 times to forty two")

print(message)
print(message.tool_calls)

print("Creating a chain of tools ...")

chain = llm_with_tools | (lambda x: x.tool_calls[0]["args"]) | multiply

print(
    chain.invoke("what is four times 23")
)