import streamlit as st
import os

# Streamlit Secrets에서 API 키를 가져와 환경변수에 설정합니다.
# 로컬 환경: 프로젝트 폴더 내 .streamlit/secrets.toml 파일 사용
# 배포 환경: Streamlit Cloud > App Settings > Secrets 사용
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("⚠️ OpenAI API 키가 설정되지 않았습니다. Streamlit Secrets에 'OPENAI_API_KEY'를 설정해주세요.")
    st.stop() # 키가 없으면 이후 코드를 실행하지 않고 중단합니다.

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 웹 페이지 기본 설정
st.set_page_config(page_title="PDF RAG 챗봇", page_icon="🤖")
st.title("🤖 PDF 문서 기반 챗봇 (unsu.pdf)")
st.caption("질문을 입력하면 AI가 PDF 문서를 검색하여 답변합니다.")

# 1. PDF 로드 및 벡터 DB 구축 (캐싱 처리하여 1회만 실행되게 함)
@st.cache_resource
def load_and_process_pdf():
    try:
        # 파일이 존재하는지 먼저 확인
        if not os.path.exists("unsu.pdf"):
            st.error("현재 디렉토리에 'unsu.pdf' 파일이 없습니다. 파일을 준비해 주세요.")
            return None
            
        loader = PyPDFLoader("unsu.pdf")
        pages = loader.load_and_split()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.split_documents(pages)
        
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma.from_documents(texts, embeddings_model)
        return db
    except Exception as e:
        st.error(f"데이터베이스 구축 중 오류 발생: {e}")
        return None

# 2. RAG 체인 생성 (캐싱 처리)
@st.cache_resource
def create_chain(_db):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # MultiQueryRetriever 설정
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=_db.as_retriever(), 
        llm=llm
    )

    system_prompt = (
        "너는 질문-답변을 돕는 유능한 비서야. "
        "아래 제공된 맥락(context)만을 사용하여 질문에 답해줘. "
        "답을 모르면 모른다고 하고, 절대 답변을 지어내지 마.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever_from_llm, question_answer_chain)
    return rag_chain

# --- 메인 로직 실행 ---

with st.spinner("문서 분석 및 데이터베이스를 준비 중입니다. 잠시만 기다려주세요..."):
    db = load_and_process_pdf()

if db:
    rag_chain = create_chain(db)
    
    # 대화 기록을 저장할 세션 상태(Session State) 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 이전 대화 내용 화면에 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 UI
    if prompt_text := st.chat_input("문서 내용에 대해 질문해 보세요 (예: 아내가 사달라고 한 음식은?)"):
        
        # 1. 사용자의 질문을 화면에 표시하고 기록에 추가
        with st.chat_message("user"):
            st.markdown(prompt_text)
        st.session_state.messages.append({"role": "user", "content": prompt_text})

        # 2. AI의 답변 생성 및 표시
        with st.chat_message("assistant"):
            with st.spinner("문서를 검색하여 답변을 생성하고 있습니다..."):
                try:
                    # 체인 실행
                    response = rag_chain.invoke({"input": prompt_text})
                    answer = response['answer']
                    
                    # 답변 표시
                    st.markdown(answer)
                    
                    # 대화 기록에 추가
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"답변 생성 중 오류가 발생했습니다: {e}")