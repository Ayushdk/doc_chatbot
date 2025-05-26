# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader
# from langchain.indexes import VectorstoreIndexCreator
# from langchain.chains import RetrievalQA


# st.title("RAG Chatbot")

# # holding old messages with session state
# if 'messages' not in st.session_state:
#     st.session_state.messages = []

# # displaying history of messages
# for message in st.session_state.messages:
#     st.chat_message(message['role']).markdown(message['content'])



# @st.cache_resource
# def get_vectorstore():
#     pdf_name = "./doc.pdf"
#     loaders = [PyPDFLoader(pdf_name)]
#     # chunks
#     index = VectorstoreIndexCreator(
#         embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
#         text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     ).from_loaders(loaders)
#     return index.vectorstore



# prompt = st.chat_input("Enter your prompt here")


# if prompt : 
#     st.chat_message("user").markdown(prompt)
#     st.session_state.messages.append({'role':'user','content':prompt})

#     groq_sys_prompt = ChatPromptTemplate.from_template("""You are very smart at everything, you always give the best, 
#                                             the most accurate and most precise answers. Answer the following Question: {user_prompt}.
#                                             Start the answer directly. No small talk please""")
    
#     model="llama3-8b-8192"
#     groq_chat = ChatGroq(
#         model = model,
#         groq_api_key = "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"
#     )


#     # parser = StrOutputParser()
#     # chain = groq_sys_prompt | groq_chat | parser
#     # response = chain.invoke({"user_prompt":st.session_state.messages})

#     # st.chat_message("assistant").markdown(response)
#     # st.session_state.messages.append({'role':'assistant','content':response})


#     try:
#         vectorstore = get_vectorstore()
#         if vectorstore is None:
#             st.error("Failed to load document")
      
#         chain = RetrievalQA.from_chain_type(
#             llm=groq_chat,
#             chain_type='stuff',
#             retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
#             return_source_documents=True)
       
#         result = chain({"query": prompt})
#         response = result["result"]  # Extract just the answer
#         #response = get_response_from_groq(prompt)
#         st.chat_message('assistant').markdown(response)
#         st.session_state.messages.append(
#             {'role':'assistant', 'content':response})
#     except Exception as e:
#         st.error(f"Error: {str(e)}")
    






from dotenv import load_dotenv
import streamlit as st
import os
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

st.markdown("<h1 style='text-align: center;'>RAG Chatbot</h1>", unsafe_allow_html=True)

# Session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# File uploader
uploader_placeholder = st.empty()
file = uploader_placeholder.file_uploader("Upload a PDF file", type=["pdf"])

# temp directory exists
temp_dir = "temp_files"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

@st.cache_resource
def get_vectorstore(uploaded_file):
    """Processes the uploaded PDF and creates a vector database for retrieval."""
    if uploaded_file is None:
        return None
    
    # Sanitize filename to avoid path issues
    safe_filename = re.sub(r'[\\/:*?"<>|]', '_', uploaded_file.name)  
    temp_file_path = os.path.join(temp_dir, safe_filename)
    
    # saving the uploaded file 
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # main work for pdf processing
    loaders = [PyPDFLoader(temp_file_path)]
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders(loaders)

    return index.vectorstore

# Hide file uploader after a successful upload
if file:
    succ = st.success(f"File '{file.name}' uploaded successfully! ✅")
    uploader_placeholder.empty() 

prompt = st.chat_input("Enter your prompt here")

if prompt: 
    succ.empty()
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    groq_sys_prompt = ChatPromptTemplate.from_template("""You are very smart at everything, you always give the best, 
                                            the most accurate and most precise answers. Answer the following Question: {user_prompt}.
                                            Start the answer directly. No small talk please""")
    
    model = "llama3-8b-8192"
    groq_chat = ChatGroq(
        model=model,
        groq_api_key="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"
        # os.getenv("GROQ_API_KEY")
    )
    
    try:
        vectorstore = get_vectorstore(file)  
        if vectorstore is None:
            st.error("Failed to load document")
        else:
            chain = RetrievalQA.from_chain_type(
                llm=groq_chat,
                chain_type='stuff',
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True
            )
        
            result = chain({"query": prompt})  
            
            if result["result"]:
                response = result["result"]
            else:
                response = groq_chat.invoke({"user_prompt": st.session_state.messages})

            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})
    
    except Exception as e:
        st.error(f"Error: {str(e)}")