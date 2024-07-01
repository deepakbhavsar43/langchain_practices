import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureOpenAI


def main():
    load_dotenv()
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]

    print("Azure OpenAI Endpoint : " + endpoint)

    prompt = ChatPromptTemplate.from_template("Tell me key achievements of {name} in 6 bulleted points")
    llm = AzureOpenAI(deployment_name="gpt-35-turbo-instruct", max_tokens=1000)
    chain = prompt | llm

    response = chain.invoke({"name": "Mahatma Gandhi"})

    print(response)


if __name__ == "__main__":
    main()
