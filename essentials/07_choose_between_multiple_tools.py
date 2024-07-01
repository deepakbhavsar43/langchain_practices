from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

load_dotenv()

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together"""
    
    return first_int*second_int

@tool
def add(first_int: int, second_int: int) -> int:
    """Add two integers"""
    return first_int + second_int

@tool
def exponentize(base: int, exponent: int) -> int:
    """Exponenitize the base to the exponent value"""
    return base ** exponent

llm = AzureChatOpenAI(
    temperature = 0,
    deployment_name = "iomega-gpt-4",
    max_tokens=4000
)

tools = [multiply, add, exponentize]
llm_with_tools = llm.bind_tools(tools)

def call_tools(msg: AIMessage) -> Runnable:
    """
        Simple Sequential Tool Calling Helper Function
    """
    
    tool_map = {tool.name: tool for tool in tools}
    tool_calls = msg.tool_calls.copy()
    for tool_call in tool_calls:
        tool_call["output"] = tool_map[tool_call["name"]].invoke(
            tool_call["args"])
        
    return tool_calls

chain = llm_with_tools | call_tools

# print(
#     chain.invoke("what is 23 times 7")
# )

print(
    chain.invoke("cube thirty seven")
)