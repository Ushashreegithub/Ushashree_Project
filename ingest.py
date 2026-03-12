from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma

# Load PDF
loader = PyPDFLoader("data/policy.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages")

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

embeddings = FakeEmbeddings(size=384)

vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db"
)

vectorstore.persist()

print("Documents stored locally!")

# Test query
query = "What is this policy about?"

results = vectorstore.similarity_search(query, k=5)

print("\nTop Results:\n")

for i, r in enumerate(results):
    print(f"Result {i+1}:")
    print(r.page_content[:300])
    print("-" * 50)
