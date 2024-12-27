from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from src.prompt import *


# load env file and openai key
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]


def file_processing(file_path):

    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    # combining texts from all pages becasue some pages may have less data
    # so the chunks wouldn't get created as per the size defined
    question_gen = ""
    for page in data:
        question_gen = question_gen + " \n" + page.page_content

    # testing a smaple split
    splitter_ques_gen = TokenTextSplitter(
        model_name="gpt-4o-mini",
        chunk_size=10000,
        chunk_overlap=200,
    )
    chunk_ques_gen = splitter_ques_gen.split_text(question_gen)

    # Earlier chunking was just an example. Below is the actual chunking for
    # the project. Also we will use Document instead of passing a string as it
    # is the preferred method.

    document_ques_gen = [Document(page_content=t) for t in chunk_ques_gen]
    # If you wish to skip the first split, then go for the document_ques_gen below:
    # document_ques_gen = [Document(page_content=question_gen)]

    splitter_ans_gen = TokenTextSplitter(
        model_name="gpt-4o-mini",
        chunk_size=1000,
        chunk_overlap=100,
    )
    document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)

    return document_ques_gen, document_answer_gen


def llm_pipeline(file_path):

    document_ques_gen, document_answer_gen = file_processing(file_path)

    llm_ques_gen_pipeline = ChatOpenAI(
        temperature=0.3,
        model="gpt-4o-mini",
    )

    PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["text"],
        template=prompt_template,
    )

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    ques_gen_chain = load_summarize_chain(
        llm=llm_ques_gen_pipeline,
        chain_type="refine",
        verbose=True,
        question_prompt=PROMPT_QUESTIONS,
        refine_prompt=REFINE_PROMPT_QUESTIONS,
    )

    # generating questions
    ques = ques_gen_chain.run(document_ques_gen)
    # print(ques)

    # Saving document_answer_gen in vector DB and perform similarity search
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    # defining llm model for answer generations
    llm_answer_gen = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")

    ques_list = ques.split("\n\n")
    print(ques_list)
    filtered_ques_list = [
        element
        for element in ques_list
        if element.endswith("?") or element.endswith(".")
    ]
    print(filtered_ques_list)

    answer_generation_chain = RetrievalQA.from_chain_type(
        llm=llm_answer_gen,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
    )

    return answer_generation_chain, filtered_ques_list
