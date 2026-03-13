from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
import os

# Add your Hugging Face API key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_HF_API_KEY"

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector database
db = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

# Load LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.3,
    max_new_tokens=512
)

print("📘 SOP Assistant Ready (type exit to stop)")

while True:

    query = input("\n❓ Ask a question: ")

    if query.lower() == "exit":
        break

    docs = db.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an AI assistant answering questions from a company's SOP document.

Only answer from the provided context.
If the answer is not in the document say:
"I could not find this information in the SOP."

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    print("\n💡 Answer:\n")
    print(response)