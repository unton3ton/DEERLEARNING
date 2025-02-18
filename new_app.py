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
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

def ollama_llm(question, context):
    formatted_prompt = f'''Context: {context}
        Based on the context, answer the following question:
        Question: {question}\n
        {userpart_system_promt}''' # f"Question: {question}\n\nContext: {context}"
    
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

# def rag_chain(question):
#     search = vectorstore.similarity_search(question, k=5)
#     temp_prompt = PromptTemplate(input_variables=["context", "question"], template=template)
#     final_prompt = prompt.format(question=query, context=search)
#     retrieved_docs = retriever.invoke(question)
#     formatted_context = combine_docs(retrieved_docs)
#     return ollama_llm(question, formatted_context)


st.title("LLM's Chat: 🌐 Web & 📝 Docs")
st.caption("Интеллектуальный поиск по веб-страницам и документам, использущий RAG-метод и локальные LLM.") # deepseek-r1:7b (arch:qwen2)

llmodel = st.selectbox(
                'Выберите LLM:',
                    ('deepseek-r1:32b',
                     'deepseek-r1:7b',
                     #'qwen2.5-coder:32b', 
                     'llama3.3', 
                     'llama3.2', 
                     'mistral-small', 
                     #'lakomoor/vikhr-llama-3.2-1b-instruct:1b', 
                     #'rscr/vikhr_nemo_12b', 
                     #'hf.co/yasserrmd/Qwen2.5-7B-Instruct-1M-gguf:Q4_K_M'
                    )
)

system_promt = st.text_input("Введите ваш системный промт 👇 (при необходимости)", 
                              help='''Это просто текстовая инструкция для объяснения модели, что от нее требуется.
                              Например, самый просто вариант: отвечай только по контексту''')

if system_promt:
    userpart_system_promt = system_promt
else:
    userpart_system_promt = '''Provide an answer based only on the context provided, without using general knowledge. The answer should be taken directly from the context provided.
        Please correct grammatical errors to improve readability.
        If there is not enough information in the context to answer the question, indicate that the answer is missing in this context.
        Please include the source of the information as a link explaining how you came to your answer.'''

temperatuur = st.slider("Выберите коэффициент креативности языковой модели (LLM)", 0.0, 2.0, step=0.1, value=0.1)

predict_msg = st.slider("Выберите длину выходного окна печати для модели (tokens): 100 tokens = 75 words in English",
                         0, 
                         100_000, 
                         step=100, 
                         value=3000,
                        )
chunk_size = st.slider("Максимальный размер текста в символах для каждого куска (tokens)", 100, 2000, step=100, value=1200)
chunk_overlap = st.slider("Количество символов, которые будут пересекаться между соседними кусками (это может быть полезно для обеспечения контекста)", 0, 1000, step=25, value=300)

# Get the webpage URL from the user
webpage_url = st.text_area("Добавте ссылки на интересующие Web-страницы (URLs)", help='Можно ввести несколько ссылок через "Enter"')

document_files_path = st.file_uploader("Выберите txt или pdf файлы", 
                                          type=["txt", "pdf"],
                                          help="Upload a PDF or TXT file",
                                          key="file_uploader",
                                          accept_multiple_files=True)
unic_key = 0
question = st.text_input("Задайте вопрос по содержимому web-страниц(ы) и/или загруженным файлам", key=f'{unic_key}', help='Для старта назмите "Enter"')

# Шаблон для вопросно-ответного запроса
template = "Вопрос: {question} \n\nОтвет:"


web_loader, txt_loader, pdf_loader = None, None, None
document_file_path = []

if question:
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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap, is_separator_regex = True)
    splits = text_splitter.split_documents(docs)
    
    # 2. Create Ollama embeddings and vector store
    embeddings = OllamaEmbeddings(model=llmodel)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    vectorstore.save_local("faiss_AiDocs")

    vectorstore = FAISS.load_local("faiss_AiDocs", embeddings, allow_dangerous_deserialization=True)

    # 4. RAG Setup
    retriever = vectorstore.as_retriever()

    # Ask a question about the webpage
    # Chat with the webpage
    if question:
        # result = rag_chain(prompt)

        search = vectorstore.similarity_search(question, k=5)
        prompt = PromptTemplate(input_variables=["context", "question"], template=template)

        final_prompt = prompt.format(question=question, context=search)
        retrieved_docs = retriever.invoke(final_prompt)
        formatted_context = combine_docs(retrieved_docs)

        result = ollama_llm(question, formatted_context)
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
