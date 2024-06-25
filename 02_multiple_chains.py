from langchain_openai import AzureOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain

from dotenv import load_dotenv

llm = AzureOpenAI(azure_deployment="gpt-35-turbo-instruct")

template1 = """
    write a blog outline given a topic.
    
    Topic: {topic}
"""

prompt_template1 = PromptTemplate(input_variables=["topic"], template=template1)

outline_chain = LLMChain(llm=llm, prompt=prompt_template1, output_key="outline")

template2 = """
    write a blog article based on the below outline.
    
    Outline: {outline}
"""

prompt_template2 = PromptTemplate(input_variables=["outline"], template=template2)
article_chain = LLMChain(llm=llm, prompt=prompt_template2, output_key="article")

overall_chain = SequentialChain(
    chains=[outline_chain, article_chain],
    input_variables=["topic"],
    output_variables=["outline", "article"],
    verbose=True,
)

response = overall_chain({"topic": "Deep Learning"})

print(response)
