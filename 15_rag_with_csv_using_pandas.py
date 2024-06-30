import pandas as pd

from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import AzureChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint

import streamlit as st
import os


def main():
    load_dotenv()

    st.set_page_config(
        page_title="CSV Document - Chatbot",
        page_icon=":books:"
    )

    st.title("HR - Attrition Analysis Chatbot")
    st.subheader("Helps to uncover insights from HR Attrition Data!")

    st.markdown(
        """
            This chatbot is created to demonstrate and answer questions from a set of attributes
            data from your CSV File, that was curated by your organization data engineering team.

            This is designed to analyze your questions, and execute data frame pandas to answer your questions.
        """
    )

    user_question = st.text_input(
        "Ask your questions about HR Attrition Data ...")

    csv_path = "./hr-employees-attritions-internet.csv"
    df = pd.read_csv(csv_path)

    llm = AzureChatOpenAI(
        deployment_name="training-gpt-4",
        temperature=0,
        max_tokens=4000
    )

    # repo_id = "databricks/dolly-v2-3b"

    # llm = HuggingFaceEndpoint(
    #     repo_id=repo_id,
    #     temperature=0.5,
    #     huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    # )

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True
    )

    agent.handle_parsing_errors = True

    answer = agent.invoke(user_question)

    st.write(answer["output"])


if __name__ == "__main__":
    main()
