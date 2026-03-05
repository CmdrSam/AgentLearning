
import hashlib
from pathlib import Path
import json
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    DirectoryLoader,
)

# Use a path relative to this script file so it works
# no matter where you run the script from.
DOC_DIR = Path(__file__).parent / "books"
METADATA_FILE = DOC_DIR / ".index_metadata.json"
VECTOR_INDEX_PATH = DOC_DIR / "vector_store"

DOC_DIR.mkdir(exist_ok=True)

def calculate_file_hash(path):
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_file_metadata(directory):
    metadata = {}
    for file in directory.glob("**/*"):
        if file.is_file() and not file.name.startswith("."):
            metadata[str(file)] = calculate_file_hash(file)
    return metadata

def load_index_metadata():
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_index_metadata(metadata):
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)

def load_and_index_documents_if_changed():
    current_meta = get_file_metadata(DOC_DIR)
    previous_meta = load_index_metadata()

    if current_meta != previous_meta or not VECTOR_INDEX_PATH.exists():
        print("Changes detected. Re-indexing documents...")
        loaders = [
            DirectoryLoader(
                str(DOC_DIR),
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"autodetect_encoding": True},
            ),
            DirectoryLoader(
                str(DOC_DIR),
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs={"autodetect_encoding": True},
            ),
            DirectoryLoader(str(DOC_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader),
            DirectoryLoader(
                str(DOC_DIR),
                glob="**/*.docx",
                loader_cls=UnstructuredWordDocumentLoader,
            ),
        ]
        all_docs = []
        for loader in loaders:
            all_docs.extend(loader.load())

        if not all_docs:
            raise ValueError(
                f"No documents found in {DOC_DIR}. "
                "Ensure there are .txt, .md, .pdf, or .docx files present."
            )

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(all_docs)

        if not docs:
            raise ValueError(
                "No document chunks were created. "
                "Check that your source documents are non-empty text."
            )

        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(str(VECTOR_INDEX_PATH))
        save_index_metadata(current_meta)
    else:
        print("No changes detected. Loading existing index...")
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.load_local(str(VECTOR_INDEX_PATH), embeddings, allow_dangerous_deserialization=True)

    return vector_store


def get_relevant_context(query, vector_store, k=3):
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    # In newer LangChain versions, retrievers are Runnable objects.
    # Use .invoke(query) instead of .get_relevant_documents(query).
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)


vector_store = load_and_index_documents_if_changed()

question = input("Ask a question about your documents:")

context = get_relevant_context(question, vector_store)
print(context)