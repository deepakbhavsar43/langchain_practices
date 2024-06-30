from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureOpenAIEmbeddings


def create_embeddings():
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        deployment="text-embedding-ada-002",
        chunk_size=1
    )

    return embeddings


def search_similar_documents(query, no_of_documents, index_name, embeddings):
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    similar_documents = vector_store.similarity_search(
        query, k=no_of_documents)

    return similar_documents


def main():
    try:
        load_dotenv()

        index_name = "zensar-case-study-demo"
        query = """
            Experienced Candidates with Embedded Systems
            
            Requirements:
            Bachelors degree in Computer Science
            At least five years of experience working in Embedded Systems
            Understanding Computer Architecture, Programming Languages and Interfacing Technologies
        """

        embeddings = create_embeddings()
        no_of_documents = 2
        relevant_documents = search_similar_documents(
            query, no_of_documents, index_name, embeddings)

        for doc_index in range(len(relevant_documents)):
            document = relevant_documents[doc_index]
            print(document.metadata["source"])
            print(document.page_content)
            print("\n")
    except Exception as e:
        print(f"Exception Occurred, Details : {e}")


if __name__ == "__main__":
    main()
