import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="무엇이든 물어보세요", page_icon="💬", layout="wide")

st.markdown("""
    <style>
    .chat-container { display: flex; flex-direction: column; gap: 15px; padding: 10px; }
    .chat-row { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 10px; }
    .row-user { flex-direction: row-reverse; }
    
    /* 프로필 이미지 */
    .avatar { width: 40px; height: 40px; border-radius: 50%; object-fit: cover; }
    
    /* 말풍선 스타일 */
    .bubble { padding: 12px 18px; border-radius: 15px; position: relative; max-width: 60%; line-height: 1.5; font-size: 14px; }
    .user-bubble { background-color: #FF8A65; color: white; } /* 강조색 (오렌지 계열) */
    .ai-bubble { background-color: #F0F0F0; color: black; } /* 연한 회색 */
    </style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state: st.session_state.messages = []
if "vector_db" not in st.session_state: st.session_state.vector_db = None

st.title("💬 무엇이든 물어보세요")

sidebar, main = st.columns([1, 2])

with sidebar:
    st.subheader("1. 문서 업로드")
    uploaded_files = st.file_uploader("PDF 파일 선택", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files and st.session_state.vector_db is None:
        if st.button("지식 베이스 구축 시작"):
            with st.spinner("구축 중..."):
                all_docs = []
                for f in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.getvalue())
                        path = tmp.name
                    all_docs.extend(PyPDFLoader(path).load_and_split())
                    os.remove(path)
                texts = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(all_docs)
                st.session_state.vector_db = Chroma.from_documents(texts, OpenAIEmbeddings())
                st.rerun()
        
    st.info("상태: " + ("✅ 준비 완료" if st.session_state.vector_db else "⏳ 대기 중"))
    st.markdown("---")

    st.subheader("2. 질문 하기")
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("메시지 입력", disabled=st.session_state.vector_db is None)
        if st.form_submit_button("전송"):
            if user_input and st.session_state.vector_db:
                st.session_state.messages.append({"role": "user", "content": user_input})
                llm = ChatOpenAI(model="gpt-4o-mini")
                retriever = st.session_state.vector_db.as_retriever()
                rag_chain = (
                    {"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), "input": RunnablePassthrough()}
                    | ChatPromptTemplate.from_template("Context: {context}\n\n질문: {input}\n\n답변:")
                    | llm | StrOutputParser()
                )
                response = rag_chain.invoke(user_input)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

with main:
    st.subheader("3. 대화 창")
    with st.container(height=500, border=True):
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            is_user = msg["role"] == "user"
            cls = "user-bubble" if is_user else "ai-bubble"
            row_cls = "row-user" if is_user else ""
            avatar = "https://cdn-icons-png.flaticon.com/512/3135/3135715.png" if not is_user else "https://cdn-icons-png.flaticon.com/512/3135/3135768.png"
            
            st.markdown(f'''
                <div class="chat-row {row_cls}">
                    <img src="{avatar}" class="avatar">
                    <div class="bubble {cls}">{msg["content"]}</div>
                </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)