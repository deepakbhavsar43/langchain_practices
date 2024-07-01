import os

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv

load_dotenv()

endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
search_key = os.environ["AZURE_SEARCH_ADMIN_KEY"]
index_name = "employee-benefits"

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = AzureSearch(
    azure_search_endpoint=endpoint,
    azure_search_key=search_key,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
    semantic_configuration_name="default"
)

directory = os.path.join("data", "Documents")

files = [
    "Benefit_Options.pdf",
    "employee_handbook.pdf",
    "Northwind_Health_Plus_Benefits_Details.pdf",
    "Northwind_Standard_Benefits_Details.pdf",
    "PerksPlus.pdf",
    "role_library.pdf"
]

total_chunks = 0
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

for file in files:
    loader = PyPDFLoader(os.path.join(directory, file))
    file_chunks = loader.load_and_split(splitter)
    results = vector_store.add_documents(documents=file_chunks)
    total_chunks += len(results)

    print(f"Indexed The File ... {file}")

print(f"Totally Indexed ... {total_chunks} Chunk(s)")

relevant_documents = vector_store.similarity_search(
    "What is included in the organization Northwind Health Plus Plan that is not in standard?",
    k=50,
    search_type="similarity"
)

top3_documents = relevant_documents[:3]

for doc in top3_documents:
    print("-" * 80)
    print(f" Source: {doc.metadata['source']}")
    print(f"Chunk Content : {doc.page_content}")
