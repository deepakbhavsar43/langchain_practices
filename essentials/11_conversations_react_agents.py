# Conversational ReAct Agents

from dotenv import load_dotenv
from langchain.agents import load_tools, initialize_agent
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory

load_dotenv()

llm = AzureChatOpenAI(
    temperature=0,
    deployment_name="iomega-gpt-4",
    max_tokens=4000
)

tools = load_tools(
    [
        "llm-math"
    ],
    llm=llm
)

memory = ConversationBufferMemory(memory_key="chat_history")

conversational_agent = initialize_agent(
    agent="conversational-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    memory=memory,
)

output1 = conversational_agent.run("Add 7 to 9 and tell me the result")

print(output1)

output2 = conversational_agent.run("add 5")

print(output2)
