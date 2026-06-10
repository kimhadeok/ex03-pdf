# 과제]
# 1. 유튜브 링크만 넣으면 끝! 영상 요약 및 번역기

import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
import yt_dlp
from openai import OpenAI

# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

# 페이지 기본 설정 (오른쪽 창을 위해 레이아웃을 wide로 변경)
st.set_page_config(page_title="유튜브 AI 요약기", page_icon="📺", layout="wide")

# --- 세션 상태(Session State) 초기화 ---
# 1. 성공한 URL 기록을 저장할 리스트
if "history" not in st.session_state:
    st.session_state.history = []
    
# 2. 텍스트 입력창의 값을 관리할 변수
if "input_url" not in st.session_state:
    st.session_state.input_url = ""

# 히스토리 버튼 클릭 시 입력창 URL을 변경하는 콜백 함수
def set_input_url(url):
    st.session_state.input_url = url

# --- 사이드바 설정 ---
with st.sidebar:
    st.header("⚙️ 설정")
    default_api_key = os.getenv("OPENAI_API_KEY", "")
    api_key = st.text_input("OpenAI API Key를 입력하세요", type="password", value=default_api_key)
    
    st.markdown("---")
    st.markdown("### 지원 언어")
    language = st.selectbox(
        "요약본을 번역할 언어를 선택하세요:",
        ("한국어", "English", "日本語", "中文", "Español", "Français")
    )

    st.markdown("---")
    st.markdown("### 추출 방식")
    extraction_method = st.radio(
        "자막 추출이 차단될 경우 음성 추출을 사용하세요:",
        ("일반 자막 추출 (빠름)", "음성 인식(Whisper) 추출 (IP차단 우회)")
    )

# --- 메인 화면 레이아웃 분할 (좌측 7 : 우측 3) ---
col_main, col_right = st.columns([7, 3])

# --- 좌측 메인 창 (입력 및 결과) ---
with col_main:
    st.title("📺 유튜브 영상 AI 요약 & 번역")
    st.markdown("유튜브 링크를 입력하면 AI가 영상의 핵심 내용을 요약하고 원하는 언어로 번역해 줍니다.")

    # URL 입력창 (key를 'input_url'로 지정하여 세션 상태와 연동)
    youtube_url = st.text_input(
        "유튜브 영상 링크(URL)를 붙여넣으세요:", 
        placeholder="https://www.youtube.com/watch?v=...",
        key="input_url"
    )

    prompt_template = """
    당신은 전문적인 영상 콘텐츠 요약가이자 번역가입니다.
    아래 제공된 유튜브 영상의 자막(스크립트)을 바탕으로 다음 지시사항을 수행하세요:

    1. 영상의 핵심 내용을 3~5개의 글머리 기호(Bullet points)로 명확하게 요약할 것.
    2. 전체적인 결론이나 시사점을 한 문단으로 덧붙일 것.
    3. 작성된 모든 내용을 반드시 '{target_language}'(으)로 번역하여 출력할 것.

    [영상 자막]
    {transcript}

    [요약 및 번역 결과]:
    """
    prompt = PromptTemplate(
        input_variables=["target_language", "transcript"],
        template=prompt_template
    )

    if st.button("🚀 요약 및 번역 시작"):
        if not api_key:
            st.warning("⚠️ 좌측 사이드바에 OpenAI API Key를 입력해주세요. (또는 .env 파일에 설정해주세요)")
        elif not youtube_url:
            st.warning("⚠️ 유튜브 링크를 입력해주세요.")
        else:
            try:
                with st.spinner("영상 텍스트를 분석하고 요약 중입니다. 잠시만 기다려주세요... ⏳"):
                    os.environ["OPENAI_API_KEY"] = api_key
                    video_title = "알 수 없는 제목" # 히스토리에 표시할 제목 초기화
                    
                    if extraction_method == "일반 자막 추출 (빠름)":
                        loader = YoutubeLoader.from_youtube_url(
                            youtube_url, 
                            add_video_info=False,
                            # 주요 국가의 원본 자막을 모두 허용하도록 확장하고, 에러를 유발하는 translation="ko"는 제거합니다.
                            language=["ko", "en", "en-US", "ja", "zh-CN", "zh-TW", "es", "fr", "de", "ru", "pt", "it", "vi", "th", "id"] 
                        )
                        docs = loader.load()
                        
                        if not docs:
                            st.error("❌ 이 영상에서는 자막을 추출할 수 없습니다. (자막이 비활성화된 영상일 수 있습니다.)")
                            st.stop()
                        else:
                            st.success("✅ 자막 추출 완료!")
                            transcript_text = docs[0].page_content
                            
                            # 빠르고 안전하게 영상 제목만 가져오기
                            try:
                                with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                                    info = ydl.extract_info(youtube_url, download=False)
                                    video_title = info.get('title', youtube_url)
                            except:
                                video_title = youtube_url
                    
                    else:
                        st.info("🎵 유튜브 음성을 다운로드하여 텍스트로 변환 중입니다. (영상 길이에 따라 1~3분 소요)")
                        ydl_opts = {
                            'format': 'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio',
                            'outtmpl': 'temp_audio.%(ext)s',
                            'noplaylist': True,
                            'quiet': True
                        }
                        
                        audio_filename = None
                        try:
                            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                info = ydl.extract_info(youtube_url, download=True)
                                audio_filename = f"temp_audio.{info.get('ext', 'webm')}"
                                video_title = info.get('title', youtube_url) # 오디오 추출 시 제목 획득
                                
                            client = OpenAI(api_key=api_key)
                            with open(audio_filename, "rb") as audio_file:
                                transcript_response = client.audio.transcriptions.create(
                                    model="whisper-1",
                                    file=audio_file
                                )
                            transcript_text = transcript_response.text
                            st.success("✅ 음성 인식(Whisper STT) 추출 완료!")
                            
                        finally:
                            if audio_filename and os.path.exists(audio_filename):
                                os.remove(audio_filename)

                    # LLM 설정 및 요약 실행
                    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
                    final_prompt = prompt.format(target_language=language, transcript=transcript_text)
                    response = llm.invoke(final_prompt)
                    
                    st.markdown("### 📝 요약 결과")
                    st.info(response.content)
                    
                    # --- 처리가 모두 성공하면 히스토리에 추가 ---
                    # 기존에 같은 URL이 있다면 제거하고 최상단에 새롭게 추가
                    st.session_state.history = [item for item in st.session_state.history if item['url'] != youtube_url]
                    st.session_state.history.insert(0, {"title": video_title, "url": youtube_url})
                        
            except Exception as e:
                error_msg = str(e)
                if "YouTube is blocking requests from your IP" in error_msg:
                    st.error("🚨 YouTube에서 자막 추출 접속을 차단했습니다 (IP 차단).")
                    st.info("좌측 사이드바에서 '음성 인식(Whisper) 추출' 방식으로 변경하여 시도해 보세요.")
                else:
                    st.error(f"오류가 발생했습니다: {error_msg}")

# --- 우측 창 (히스토리 기록) ---
# 메인 창에서 처리가 끝난 후 렌더링 되므로 업데이트 된 리스트가 즉시 반영됨
with col_right:
    st.header("🕒 최근 성공 기록")
    st.markdown("성공적으로 요약된 영상 목록입니다. 클릭하면 링크가 다시 입력됩니다.")
    
    if not st.session_state.history:
        st.write("아직 기록이 없습니다.")
    else:
        for item in st.session_state.history:
            # 클릭 시 set_input_url 함수를 호출하여 텍스트창을 채움
            st.button(
                label=f"🎬 {item['title']}", 
                key=f"hist_{item['url']}", 
                on_click=set_input_url, 
                args=(item['url'],),
                use_container_width=True,
                help=item['url'] # 마우스를 올리면 URL 원본이 보임
            )