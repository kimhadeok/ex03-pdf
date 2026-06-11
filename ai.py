import streamlit as st
import time
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -------------------------------------------------------------------
# 필수 라이브러리 설치 안내
# -------------------------------------------------------------------
# 이 코드를 실행하기 전에 아래 라이브러리들을 설치해야 합니다.
# pip install streamlit langchain langchain-openai langchain-community faiss-cpu beautifulsoup4 tiktoken
# -------------------------------------------------------------------

# 1. 페이지 및 레이아웃 설정
st.set_page_config(page_title="AI 지식 비서 Q&A", page_icon="🤖", layout="wide")

# 2. 커스텀 CSS (아바타 애니메이션 효과)
css = """
<style>
/* 말하는 애니메이션: 얼굴 끄덕임과 몸의 약간의 움직임을 표현 */
@keyframes talk {
  0% { transform: translateY(0) rotate(0deg) scale(1); }
  25% { transform: translateY(-5px) rotate(-2deg) scale(1.02); }
  50% { transform: translateY(0) rotate(0deg) scale(1); }
  75% { transform: translateY(-3px) rotate(2deg) scale(1.01); }
  100% { transform: translateY(0) rotate(0deg) scale(1); }
}
.avatar-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 400px;
    background-color: #f0f2f6;
    border-radius: 15px;
    margin-top: 20px;
}
.avatar-talking {
  width: 250px;
  height: 250px;
  /* 몸 전체가 나오는 로봇/AI 이미지로 교체 */
  background-image: url('https://cdn-icons-png.flaticon.com/512/1698/1698535.png');
  background-size: contain;
  background-repeat: no-repeat;
  background-position: center;
  animation: talk 1.2s infinite ease-in-out; /* 애니메이션 시간 및 속도 조절 */
  filter: drop-shadow(0 10px 15px rgba(0,0,0,0.2)); /* box-shadow 대신 drop-shadow 사용 */
}
.avatar-idle {
  width: 250px;
  height: 250px;
  /* 몸 전체가 나오는 로봇/AI 이미지로 교체 */
  background-image: url('https://cdn-icons-png.flaticon.com/512/1698/1698535.png');
  background-size: contain;
  background-repeat: no-repeat;
  background-position: center;
  filter: drop-shadow(0 5px 10px rgba(0,0,0,0.1));
  transition: all 0.3s ease;
}
/* 컨테이너 스크롤바 숨기기 (선택적) */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-thumb {
    background-color: #c1c1c1;
    border-radius: 10px;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# 3. 데이터 로드 및 Vector DB 구축 (캐싱하여 반복 연산 방지)
@st.cache_resource(show_spinner=False)
def init_rag(api_key):
    # 실제 환경에서는 책 전체를 크롤링하지만, 예시로 주요 챕터 URL을 로드합니다.
    urls = [
        "https://wikidocs.net/book/5942", # 책 메인
        "https://wikidocs.net/52460",     # 머신러닝의 개념
        "https://wikidocs.net/52846",     # 파이토치 텐서 선언
        "https://wikidocs.net/53560"      # 딥러닝 개요
    ]
    loader = WebBaseLoader(urls)
    docs = loader.load()
    
    # 텍스트 스플리팅
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    
    # 임베딩 및 FAISS 벡터 스토어 생성
    vectorstore = FAISS.from_documents(
        documents=splits, 
        embedding=OpenAIEmbeddings(openai_api_key=api_key)
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. LangChain RAG 체인 설정 (retriever 의존성 제거)
def get_rag_chain(api_key):
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, openai_api_key=api_key)
    
    # 시스템 프롬프트 구성
    template = """너는 Internal Knowledge Base AI비서이자 유능한 AI 전문가야.
누구나, 특히 중학생도 완벽히 이해할 수 있는 수준으로 비유를 들어서 아주 친절하고 쉽게 설명해.
필요하다면 마크다운 테이블, 가상의 텍스트 그래프 기호, 예제 코드 등을 동원해서 시각적으로 쉽게 설명해.

제공된 [컨텍스트(전자책 데이터)]만을 모두 읽고 해당 콘텐츠에 대해서만 [주 답변]을 작성해.
[주 답변]에는 외부 지식을 섞지 말고, 컨텍스트의 내용에만 집중해.

컨텍스트의 내용 외에, 중학생의 이해를 돕기 위한 너의 배경 지식, 비유, 추가 설명은 반드시 [추가 보조 답변]에 작성해.
[추가 보조 답변]에는 추가적인 배경 지식과 비유만 작성해. 참고 링크나 '더 알고 싶다면~' 같은 안내 문구는 시스템이 자동으로 추가하므로 너는 절대로 작성하지 마.

AI 및 머신러닝, 딥러닝과 전혀 관련 없는 질문에는 "해당 질문은 제 지식 범위를 벗어납니다. AI에 관련된 질문을 해주세요."라고만 답해.

반드시 아래 지정된 포맷을 엄격하게 지켜서 출력해:

[주 답변]
(컨텍스트 기반의 아주 쉬운 설명)

[추가 보조 답변]
(추가적인 배경 지식, 비유. 링크나 안내 문구 절대 금지)

[컨텍스트]
{context}

[질문]
{question}
"""
    prompt = PromptTemplate.from_template(template)
    
    # 체인 구성: 컨텍스트와 질문을 직접 입력받도록 변경
    chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    return chain

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_thinking" not in st.session_state:
    st.session_state.is_thinking = False

# =====================================================================
# UI 레이아웃 구성
# =====================================================================

# 사이드바 (API 키 설정)
with st.sidebar:
    st.header("⚙️ 시스템 설정")
    openai_api_key = st.text_input("OpenAI API Key를 입력하세요", type="password")
    st.markdown("---")
    st.markdown("**사용 기술 스택**")
    st.markdown("- LangChain\n- Streamlit\n- FAISS (Vector DB)\n- OpenAI Embeddings")
    st.markdown("**학습 데이터 출처**")
    st.markdown("[PyTorch로 시작하는 딥 러닝 입문](https://wikidocs.net/book/5942)")

# (1) 상단 영역: 제목, 설명, 질문 폼
st.title("🤖 AI에 대한 궁금한 사항을 무엇이든 물어보세요")
st.markdown("전자책 데이터를 기반으로 동작하는 AI 지식 비서입니다. 어려운 AI 기술도 중학생이 이해할 수 있을 만큼 쉽고 친절하게 답변해 드립니다.")

with st.form(key="qa_form", clear_on_submit=False): # clear_on_submit=False로 변경하여 엔터키 이슈 방지
    col_input, col_btn = st.columns([8, 1])
    with col_input:
        user_input = st.text_input("질문 입력", placeholder="예: 머신러닝과 딥러닝의 차이가 뭐야?", label_visibility="collapsed")
    with col_btn:
        submit_btn = st.form_submit_button("질문하기", use_container_width=True)

# 질문 버튼 클릭 이벤트 처리 전처리
if submit_btn and user_input:
    if not openai_api_key:
        st.warning("👈 사이드바에 OpenAI API Key를 먼저 입력해주세요.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.is_thinking = True

st.markdown("---")

# (2) 하단 레이어 분할 (왼쪽 아바타 영역 축소, 오른쪽 답변 영역 확대)
col_left, col_right = st.columns([1, 2.5])

# (3) 왼쪽 영역: 아바타 표시
with col_left:
    st.subheader("💡 AI 비서")
    avatar_html = '<div class="avatar-container"><div class="{0}"></div></div>'
    avatar_placeholder = st.empty()
    
    # 답변 생성 상태에 따른 애니메이션 토글
    if st.session_state.is_thinking:
        avatar_placeholder.markdown(avatar_html.format("avatar-talking"), unsafe_allow_html=True)
    else:
        avatar_placeholder.markdown(avatar_html.format("avatar-idle"), unsafe_allow_html=True)

# (4) 오른쪽 영역: 답변 히스토리 (스크롤 고정)
with col_right:
    st.subheader("📜 답변 히스토리")
    # 높이를 600px로 고정하여 내부 스크롤 활성화
    history_container = st.container(height=500)
    
    with history_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="🧑‍💻"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    # (5) 주 답변, 추가 보조 답변 구분 표시
                    if "main_ans" in msg and msg["main_ans"]:
                        st.info(f"**[ 📖 데이터(전자책) 주 답변 ]**\n\n{msg['main_ans']}")
                    
                    if "sub_ans" in msg and msg["sub_ans"]:
                        # (6) 보조 답변은 참조 정보 표시 및 링크 적용 컨셉 (초록색 상자)
                        st.success(f"**[ 🔗 추가 보조 답변 및 참고 자료 ]**\n\n{msg['sub_ans']}")
                    
                    if "raw" in msg and not msg["main_ans"] and not msg["sub_ans"]:
                        st.write(msg["raw"])

# 실제 LLM 처리 과정 (렌더링 흐름상 화면 업데이트 후 백그라운드 처리처럼 동작)
if submit_btn and user_input and openai_api_key:
    with col_left:
        with st.spinner("전자책을 뒤져서 가장 쉬운 설명을 작성하고 있어요..."):
            try:
                # 1. RAG Retriever 초기화 및 로드
                retriever = init_rag(openai_api_key)
                
                # 2. 관련 문서 검색 및 소스 URL 추출
                relevant_docs = retriever.invoke(user_input)
                source_urls = list(set([doc.metadata.get("source", "") for doc in relevant_docs if doc.metadata.get("source", "")]))
                
                # 3. 컨텍스트 포맷팅 및 체인 실행
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)
                
                chain = get_rag_chain(openai_api_key)
                response = chain.invoke({
                    "context": format_docs(relevant_docs),
                    "question": user_input
                })
                
                # 4. 응답 문자열 파싱 (주 답변과 보조 답변 분리)
                main_ans, sub_ans = "", ""
                if "[주 답변]" in response and "[추가 보조 답변]" in response:
                    parts = response.split("[추가 보조 답변]")
                    main_ans = parts[0].replace("[주 답변]", "").strip()
                    sub_ans = parts[1].strip()
                else:
                    # 프롬프트를 무시하고 한 번에 나온 경우
                    main_ans = response
                
                # 5. 보조 답변에 실제 참조 링크 추가
                if source_urls:
                    links_md = "\n\n**[실제 참고한 문서 링크]**\n"
                    for i, url in enumerate(source_urls, 1):
                        links_md += f"- [참고 문서 {i}]({url})\n"
                    sub_ans += links_md

                # 6. 세션에 답변 저장
                st.session_state.messages.append({
                    "role": "assistant",
                    "main_ans": main_ans,
                    "sub_ans": sub_ans,
                    "raw": response
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "main_ans": "데이터를 처리하는 중 오류가 발생했습니다. API Key나 네트워크 상태를 확인해주세요.",
                    "sub_ans": str(e),
                    "raw": ""
                })
                
    # 처리 완료 후 애니메이션 중지 및 화면 갱신
    st.session_state.is_thinking = False
    st.rerun()