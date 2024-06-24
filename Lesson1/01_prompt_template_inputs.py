from langchain_openai import AzureOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain

from dotenv import load_dotenv

llm = AzureOpenAI(azure_deployment="gpt-35-turbo-instruct")
prompt_template = PromptTemplate(
    input_variables=["topic1", "topic2"],
    template="Give me a tweet idea on {topic1} and {topic2}",
)

prompt = prompt_template.format(topic1="AI", topic2="NLP")

response = llm.invoke(prompt)

print(response)