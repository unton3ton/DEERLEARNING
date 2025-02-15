# pip install beautifulsoup4 langchain_community langchain streamlit ollama faiss-gpu pypdf

# https://anakin.ai/blog/llama-3-rag-locally/

import os
os.environ['USER_AGENT'] = 'myagent'
import random
from glob import glob
import streamlit as st
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (WebBaseLoader,
                                                  TextLoader,
                                                  PyPDFLoader,
                                                  MergedDataLoader,
                                                  SeleniumURLLoader,
                                                  WikipediaLoader,
                                                 )
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model=llmodel, 
                           messages=[{'role': 'user', 'content': formatted_prompt}],
                           options={
                                    'temperature': temperatuur, # значение от 0,0 до 0,9 (или 1) определяет уровень креативности модели или ее неожиданных ответов.
                                    #'top_p': 0.9, #  от 0,1 до 0,9 определяет, какой набор токенов выбрать, исходя из их совокупной вероятности.
                                    #'top_k': 90, # от 1 до 100 определяет, из скольких лексем (например, слов в предложении) модель должна выбрать, чтобы выдать ответ.
                                    #'num_ctx': 500_000, # устанавливает максимальное используемое контекстное окно, которое является своего рода областью внимания модели.
                                    'num_predict': predict_msg, # задает максимальное количество генерируемых токенов в ответах для рассмотрения (100 tokens ~ 75 words).
                                    },
                        )
    return response['message']['content']

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

st.title("LLM's Chat with 🌐 Web & 📝 Docs")
st.caption("This app allows you to chat with your webpages and docs using local LLM&RAG") # deepseek-r1:7b (arch:qwen2)

llmodel = st.selectbox(
                'Choose your LLM',
                    ('deepseek-r1:32b',
                     'deepseek-r1:7b',
                     'qwen2.5-coder:32b', 
                     'llama3.2', 
                     'llama3.3', 
                     'mistral-small', 
                     'lakomoor/vikhr-llama-3.2-1b-instruct:1b', 
                     'rscr/vikhr_nemo_12b', 
                     'hf.co/yasserrmd/Qwen2.5-7B-Instruct-1M-gguf:Q4_K_M')
)

temperatuur = st.slider("Select a LLM's creative coefficient", 0.0, 2.0, step=0.1, value=0.1)

predict_msg = st.slider("Select a lenght output message (tokens): 100 tokens = 75 words in English",
                         0, 
                         100_000, 
                         step=100, 
                         value=3000,
                        )

# Get the webpage URL from the user
webpage_url = st.text_area("Enter Webpage URLs", help='enter your urls with "Enter"')

document_files_path = st.file_uploader("Choose a txt or pdf file", 
                                          type=["txt", "pdf"],
                                          help="Upload a PDF or TXT file",
                                          key="file_uploader",
                                          accept_multiple_files=True)
unic_key = 0
prompt = st.text_input("Ask any question about the webpages or docs", key=f'{unic_key}')


web_loader, txt_loader, pdf_loader = None, None, None
document_file_path = []

if prompt:
    print(webpage_url, web_loader)
    # 1. Load the data
    if document_files_path:

        for document_file_path in document_files_path:         

            if document_file_path.name[-3:] == 'txt':
                temp_file_txt = f"./temp_{random.randint(1, 1_000_000)}.txt"
                with open(temp_file_txt, "ab") as file:
                    file.write(document_file_path.getvalue())
                txt_loader = TextLoader(temp_file_txt)

            if document_file_path.name[-3:] == 'pdf':
                temp_file_pdf = f"./temp_{random.randint(1, 1_000_000)}.pdf"
                with open(temp_file_pdf, "ab") as file:
                    file.write(document_file_path.getvalue())
                pdf_loader = PyPDFLoader(temp_file_pdf, extract_images = False)

    if webpage_url == '':
        webpage_url = 'http://localhost:8501'
        web_loader = WebBaseLoader(webpage_url)
    else:
        web_loader = WebBaseLoader(webpage_url,encoding='utf-8')
        st.success(f"Loaded {webpage_url} successfully!")
    print(webpage_url, web_loader)

    # Объединение загрузчиков
    if (txt_loader and pdf_loader) is not None:
        merged_loader = MergedDataLoader(loaders=[web_loader, txt_loader, pdf_loader])
        print('merged txt, pdf')
    elif txt_loader is not None:
        merged_loader = MergedDataLoader(loaders=[web_loader, txt_loader])
        print('merged txt')
    elif pdf_loader is not None:
        merged_loader = MergedDataLoader(loaders=[web_loader, pdf_loader])
        print('merged pdf')
    else:
        merged_loader = MergedDataLoader(loaders=[web_loader])
        print('merged web')

    # Загрузка всех документов
    docs = merged_loader.load()

    print(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    splits = text_splitter.split_documents(docs)
    
    # 2. Create Ollama embeddings and vector store
    embeddings = OllamaEmbeddings(model=llmodel)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    # 4. RAG Setup
    retriever = vectorstore.as_retriever()

    # Ask a question about the webpage
    # Chat with the webpage
    if prompt:
        result = rag_chain(prompt)
        st.write(result)
        unic_key += 1

        print(os.getcwd())

        print(glob(f"{os.getcwd()}/*.txt"))
        print(glob(f"{os.getcwd()}/*.pdf"))

        # удаление временных файлов
        try:
            for file in glob(f"{os.getcwd()}/*.txt"):
                os.remove(file)
        except OSError:
            pass

        try:
            for file in glob(f"{os.getcwd()}/*.pdf"):
                os.remove(file)
        except OSError:
            pass

        print('hier')


# streamlit run app.py
