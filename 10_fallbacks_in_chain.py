# Fallbacks in Chains

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig, Runnable
load_dotenv()

llm = AzureChatOpenAI(
    temperature=0,
    deployment_name="iomega-gpt-4",
    max_tokens=4000
)


@tool
def complex_tool(arg1: int, arg2: float, dict_args: dict) -> int:
    """
        Dom something complex with a complex tool
    """
    return arg1 * arg2


llm_with_tools = llm.bind_tools([complex_tool])


def try_except_tool(tool_args: dict, config: RunnableConfig) -> Runnable:
    try:
        complex_tool.invoke(tool_args, config=config)
    except Exception as e:
        return f"Calling tool with arguments : \n\n{tool_args}\n\n Raised the following errors {e}"

chain = llm_with_tools | (lambda msg: msg.tool_calls[0]["args"]) | try_except_tool

response = chain.invoke(
    "Use complex tool. the args are 10, 5.6, empty dictionary. do not forget to pass dict_arg"
)

print(response)