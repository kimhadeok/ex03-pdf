# 과제]

# 1. 유튜브 링크만 넣으면 끝! 영상 요약 및 번역기

# 2. 나만의 맞춤형 문서(PDF/TXT) 요정 (RAG 챗봇)
# Streamlit의 파일 업로드 기능(st.file_uploader)으로 화면을 구현

# 3. 메일·SNS 채널별 맞춤형 마케팅 글 생성기


# '메일·SNS 채널별 맞춤형 마케팅 글 생성기' 컨셉을 가지고 python 언어로 코드 만들어줘. 당연 실행이 가능해야해.

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# 환경변수 로드 (.env 파일에서 OPENAI_API_KEY 가져오기)
load_dotenv()

# 페이지 설정
st.set_page_config(page_title="맞춤형 마케팅 글 생성기", page_icon="✍️", layout="wide")

# --- 프롬프트 템플릿 정의 ---
# 각 채널별로 최적화된 글쓰기 가이드를 프롬프트에 포함합니다.
MARKETING_PROMPT = """
당신은 10년 차 탑티어 마케팅 카피라이터입니다.
주어진 상품/서비스 정보와 타겟 고객을 바탕으로, 지정된 '마케팅 채널'의 특성에 완벽하게 맞는 마케팅 문구를 작성해 주세요.

[상품 및 마케팅 정보]
- 상품/서비스명: {product_name}
- 핵심 특징/장점: {features}
- 타겟 고객: {target_audience}
- 글의 톤앤매너: {tone}

[채널별 작성 가이드: {channel}]
{channel_guidelines}

위 정보를 바탕으로 가장 매력적이고 클릭을 유도할 수 있는 마케팅 카피를 작성해 주세요.
결과물은 반드시 한국어로 작성해야 합니다.

[작성된 마케팅 카피]:
"""

# 채널별 세부 가이드라인 딕셔너리
CHANNEL_GUIDES = {
    "이메일 (뉴스레터/콜드메일)": """
    - 시선을 끄는 매력적인 [제목]을 2~3가지 제안해 주세요.
    - 본문은 친근하면서도 전문적인 인사말로 시작하세요.
    - 내용 전개: 문제 공감 -> 상품(해결책) 제시 -> 핵심 장점 요약 -> 명확한 Call to Action(클릭/구매 유도 버튼 문구).
    - 모바일에서도 읽기 쉽게 문단을 짧게 나누어 주세요.
    """,
    "인스타그램 (피드/릴스)": """
    - 첫 줄에서 스크롤을 멈추게 할 강력한 후킹(Hook) 문장을 작성하세요.
    - 텍스트 중간중간 시각적으로 돋보이는 이모지(🌟, 🔥, 👀 등)를 적극적으로 활용하세요.
    - 친근하고 소통하는 듯한 말투를 사용하세요.
    - 글 마지막에 어울리는 해시태그(#) 5~8개를 추천해 주세요.
    """,
    "링크드인 (B2B/전문가 네트워크)": """
    - 지나치게 상업적인 느낌보다는, 비즈니스적 가치와 인사이트를 제공하는 형태로 작성하세요.
    - 이모지는 과하지 않게 핵심 포인트에만 사용하세요.
    - 논리적이고 신뢰감을 주는 전문가의 톤을 유지하세요.
    - 마지막에는 네트워크의 의견을 묻거나 소통을 유도하는 질문으로 끝맺으세요.
    """,
    "페이스북 (광고/페이지 게시물)": """
    - 친구에게 이야기하듯 자연스럽고 공감가는 스토리텔링으로 시작하세요.
    - 너무 길지 않게 핵심 혜택을 명확히 전달하세요.
    - 즉각적인 행동(링크 클릭, 댓글 달기 등)을 유도하는 멘트를 포함하세요.
    """,
    "블로그 (SEO 홍보글 요약)": """
    - 도입부(문제 공감) - 본론(특징 설명) - 결론(추천 및 CTA) 구조로 소제목을 나누어 작성해 주세요.
    - 검색 엔진 최적화(SEO)를 고려하여 타겟 고객이 검색할 만한 키워드를 자연스럽게 녹여내세요.
    - 정보성과 홍보성이 적절히 섞인 톤을 유지하세요.
    """
}

# --- UI 레이아웃 시작 ---
st.title("✍️ AI 마케팅 카피라이터")
st.markdown("상품 정보만 입력하면 인스타그램, 이메일, 링크드인 등 **채널에 딱 맞는 마케팅 문구**를 자동으로 생성해 줍니다.")

# 메인 화면을 좌/우 2단으로 분할
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 마케팅 정보 입력")
    
    # 1. API 키 설정 (사이드바 대신 메인 상단에 배치)
    api_key = st.text_input("OpenAI API Key (또는 .env에 설정)", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    
    # 2. 필수 입력 정보
    product_name = st.text_input("상품/서비스명 *", placeholder="예: 무중력 메모리폼 베개")
    target_audience = st.text_input("타겟 고객 *", placeholder="예: 목 디스크로 고생하는 3040 직장인")
    features = st.text_area("핵심 특징 및 장점 *", placeholder="- 누운 지 3분 만에 꿀잠 보장\n- 세탁기 통째 빨래 가능\n- 런칭 기념 50% 할인", height=150)
    
    # 3. 채널 및 톤앤매너 선택 (가로로 나란히 배치)
    col_a, col_b = st.columns(2)
    with col_a:
        channel = st.selectbox("📌 타겟 채널", list(CHANNEL_GUIDES.keys()))
    with col_b:
        tone = st.selectbox("🗣️ 톤앤매너", ["친근하고 재치있는", "전문적이고 신뢰감 주는", "감성적이고 따뜻한", "긴급하고 혜택을 강조하는 (홈쇼핑톤)", "트렌디하고 힙한"])

    submit_btn = st.button("✨ 마케팅 카피 생성하기", type="primary", use_container_width=True)

with col2:
    st.subheader("💡 생성된 마케팅 카피")
    
    # 결과 출력 영역
    if submit_btn:
        if not api_key:
            st.error("OpenAI API Key를 입력해주세요.")
        elif not product_name or not target_audience or not features:
            st.warning("상품명, 타겟 고객, 핵심 특징을 모두 입력해주세요.")
        else:
            with st.spinner(f"'{channel}'에 최적화된 마케팅 문구를 고민 중입니다... ⏳"):
                try:
                    os.environ["OPENAI_API_KEY"] = api_key
                    
                    # LLM 모델 초기화
                    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
                    
                    # 프롬프트 템플릿 설정
                    prompt = PromptTemplate(
                        input_variables=["product_name", "features", "target_audience", "tone", "channel", "channel_guidelines"],
                        template=MARKETING_PROMPT
                    )
                    
                    # 최종 프롬프트 생성
                    final_prompt = prompt.format(
                        product_name=product_name,
                        features=features,
                        target_audience=target_audience,
                        tone=tone,
                        channel=channel,
                        channel_guidelines=CHANNEL_GUIDES[channel]
                    )
                    
                    # LLM 호출
                    response = llm.invoke(final_prompt)
                    
                    # 결과 출력 (성공)
                    st.success("✅ 생성이 완료되었습니다! 내용을 복사하여 바로 사용해 보세요.")
                    
                    # 수정 및 복사가 쉽도록 text_area에 출력
                    st.text_area(
                        label="결과물 수정/복사", 
                        value=response.content, 
                        height=500,
                        label_visibility="collapsed"
                    )
                    
                except Exception as e:
                    st.error(f"오류가 발생했습니다: {str(e)}")
    else:
        st.info("좌측에 정보를 입력하고 '생성하기' 버튼을 누르면 이곳에 마케팅 카피가 나타납니다.")