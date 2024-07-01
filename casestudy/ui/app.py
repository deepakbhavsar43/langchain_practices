import streamlit as st
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains.summarize import load_summarize_chain


def create_embeddings():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    return embeddings


def search_similar_documents(query, no_of_documents, index_name, embeddings):
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    similar_documents = vector_store.similarity_search(
        query, k=no_of_documents)

    return similar_documents


def get_summary_from_llm(current_document):
    llm = OpenAI(
        temperature=0,
        max_tokens=2000
    )

    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_document])

    return summary


def main():
    try:
        load_dotenv()

        index_name = "zensar-case-study-demo"

        st.set_page_config(page_title="Resume Screening Assistance")
        st.title("Resume Screening AI Assistant")
        st.subheader(
            "This AI assistant would help you to screen available resumes, that are submitted to the organization")

        job_description = st.text_area(
            "Please paste the 'JOB DESCRIPTION' here ...",
            key="1",
            height=200
        )

        document_count = st.text_input("No. of 'RESUME(s)' to return", key="2")

        submit = st.button("Analyze")

        if submit:
            embeddings = create_embeddings()
            relevant_docs = search_similar_documents(
                job_description, int(document_count), index_name=index_name, embeddings=embeddings)

            for document_index in range(len(relevant_docs)):
                st.subheader("👉 " + str(document_index+1))

                file_name = "** FILE **" + \
                    relevant_docs[document_index].metadata["source"]

                st.write(file_name)

                with st.expander("Show Me Summary ... 👀"):
                    summary = get_summary_from_llm(
                        relevant_docs[document_index])

                    st.write(" *** SUMMARY *** " + summary)

    except Exception as error:
        print(f"Error Occurred, Details : {error}")


if __name__ == "__main__":
    main()
