from dotenv import load_dotenv
from langchain_openai import AzureOpenAI
from langchain.chains.llm_math.base import LLMMathChain
from langchain_community.utilities import BingSearchAPIWrapper
from langchain.agents import Tool, initialize_agent, tool

load_dotenv

llm = AzureOpenAI(
    temperature=0, deployment_name="gpt-35-turbo-instruct", max_tokens=5000
)

llm_math = LLMMathChain.from_llm(llm)
math_tool = Tool(
    name="Calculator",
    func=llm_math.run,
    description="Useful tool when you need to answer questions about math",
)


@tool("BingSearch")
def search(search_query: str):
    """
    useful to search for any information and
    useful for when you need to search the internet for any kinds of information
    """
    search = BingSearchAPIWrapper()
    return search.run(search_query)


tools = [search, math_tool]

agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=10,
)

agent.handle_parsing_errors = True

response = agent("What's the root over 25? and let me know capital of India?")

print(response)
