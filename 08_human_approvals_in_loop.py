from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import tool

load_dotenv()

llm = AzureChatOpenAI(
    temperature = 0,
    deployment_name = "iomega-gpt-4",
    max_tokens=4000
)

@tool
def count_emails(last_number_of_days: int) -> int:
    """
        Counting Emails from Mailbox, based on the last or previous number of days. 
    """
    
    return last_number_of_days * 2

@tool
def send_email(message: str, recipient: str) -> str:
    """
        Sending Emails to a recipient.
    """
    
    return f"Successfully sent a mail to {recipient}"

tools = [count_emails, send_email]
llm_with_tools = llm.bind_tools(tools)

import json

def human_approval(message: AIMessage)-> Runnable:
    tool_strings = "\n\n".join(
        json.dumps(tool_call, indent = 2) for tool_call in message.tool_calls)
    
    input_message = f"Do you want to approvate of the following tool invocation {tool_strings} \n\n" + \
        "Please enter yes / Y, anything else shall reject execution of the tool!"
    
    response = input(input_message)
    
    if(response.lower() not in ("yes", "y")):
        raise ValueError("Tool Invocations Not Approved!")

    return message

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

chain = llm_with_tools | human_approval | call_tools

print(
    chain.invoke("how many emails i do have in mailbox for the last 10 days?")
)