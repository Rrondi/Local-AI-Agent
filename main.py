from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about a product based on customer reviews.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


def format_reviews(docs):
    lines = []
    for doc in docs:
        rating = doc.metadata.get("rating", "N/A")
        date = doc.metadata.get("date", "N/A")
        lines.append(f"- ({rating} stars, {date}) {doc.page_content}")
    return "\n".join(lines)


while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break

    reviews = retriever.invoke(question)
    reviews_text = format_reviews(reviews)
    result = chain.invoke({"reviews": reviews_text, "question": question})
    print(result)
