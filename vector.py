from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("amazon_review.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"

vector_store = Chroma(
    collection_name="amazon_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

existing_count = vector_store._collection.count()
add_documents = existing_count == 0

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        title = str(row.get("summary", "")).strip()
        review_text = str(row.get("reviewText", "")).strip()
        content = " ".join(part for part in [title, review_text] if part)
        document = Document(
            page_content=content,
            metadata={
                "rating": row.get("overall"),
                "date": row.get("reviewTime"),
                "asin": row.get("asin"),
                "reviewer": row.get("reviewerName"),
            },
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
