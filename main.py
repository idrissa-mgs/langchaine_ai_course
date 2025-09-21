import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore


from langchain import hub

from langchain_groq import ChatGroq

load_dotenv()

file_path = "apache-iceberg.pdf"


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

def split_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    texts = text_splitter.split_documents(docs)
    return texts



def main():
    book = load_pdf(file_path)
    texts = split_docs(book)

    llm = ChatOpenAI()

    embeddings = OpenAIEmbeddings()

    index_name = os.getenv("INDEX_NAME")

    query = "what is Pinecone in machine learning?"
    chain = PromptTemplate.from_template(template=query) | llm



    vectorstore = PineconeVectorStore.from_documents(texts, embeddings, index_name=index_name)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrival_chain.invoke(input={"input": query})

    print(result)



if __name__ == "__main__":
    main()
