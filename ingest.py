from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load PDF
loader = PyPDFLoader("data/sop.pdf")
documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

texts = text_splitter.split_documents(documents)

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector database
db = Chroma.from_documents(
    texts,
    embeddings,
    persist_directory="db"
)

db.persist()

print("✅ SOP document embedded successfully!")