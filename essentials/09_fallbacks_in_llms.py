from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAI
from langchain.output_parsers import DatetimeOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm_3 = AzureOpenAI(
    deployment_name = "gpt-35-turbo-instruct"
) | DatetimeOutputParser()

llm_4 = AzureChatOpenAI(
    temperature = 0,
    deployment_name = "iomega-gpt-4",
    max_tokens=4000
) | DatetimeOutputParser()

prompt = ChatPromptTemplate.from_template(
    """
        what time was {event} in (%Y-%m-%dT%H:%M:%S.%fZ format - only return this value not anything else.)
    """
)

only_35 = prompt | llm_3

# try:
#     print(only_35.invoke({
#         "event": "The superbowl in 1994"
#     }))
# except Exception as e:
#     print(f"Error Occurred, Details : {e}")
    
fallback_4 = prompt | llm_3.with_fallbacks([llm_4])

try:
    print(
        fallback_4.invoke({
            "event": "The superbowl in 1994"
        })
    )
except Exception as e:
    print(f"Error Occurred in Fallback Also!, {e}")