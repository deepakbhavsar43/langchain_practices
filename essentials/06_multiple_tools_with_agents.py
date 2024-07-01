from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent

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
prompt = hub.pull("hwchase17/openai-tools-agent")

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent = agent, tools = tools, verbose = True)

print(agent_executor.invoke({
    "input": "Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result"
}
))