import os
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

loader = PyPDFLoader("C:\\Users\\Shaina Hussain\\OneDrive\\Desktop\\MatterS\\rag-learning-log\\Shaina_Hussain_Resume (2).pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap = 50)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db",
)

query = "What are Shaina's skills?"
results = vectorstore.similarity_search(query, k=3)

llm = ChatGroq(model_name="llama-3.1-8b-instant")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k":3}),
    return_source_documents=True,
)

query = "What are Shaina's technical skills?"
response = qa_chain.invoke({"query": query})

print("\nAnswer:")
print(response["result"])
print("\nSources used:")
for doc in response["source_documents"]:
    print("—", doc.page_content[:100])

# print(f"Total chunks: {len(chunks)}")
# print("\nfirst chunk:")
# print(chunks[0].page_content)

# print(f"\nTop 3 results for:'{query}'")
# for i,doc in enumerate(results):
#     print(f"\nResults {i+1}:")
#     print(doc.page_content)
