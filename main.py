from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("unsu.pdf")
pages = loader.load_and_split()

# PDF 파일에서 페이지 단위로 로드된 문서 객체 출력
# 텍스트 정크 (Chunk) 단위로 쪼개기
# LLM이 처리하기 좋게 문서를 더 작은 단위(chunk)로 잘게 쪼갠다.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,           # 각 텍스트 조각의 최대 길이 (문자 수)
    chunk_overlap = 20,         # 텍스트 조각 간의 겹치는 길이 (문자 수). 문맥이 끊기는 것을 방지하기 위해 보통 10~20% 정도 겹치게 설정함
    length_function = len,      # 텍스트 길이를 계산하는 함수 (기본값은 len)
    is_separator_regex = False, # 구분자(separator)가 정규 표현식인지 여부 (기본값은 False)
)

# PDF 페이지 단위로 로드된 문서 객체를 텍스트 조각(Chunk) 단위로 분할 (300자)
texts = text_splitter.split_documents(pages)

if texts:
    print("--- [첫 번째 텍스트 조각(Chunk) 객체 출력] ---")
    print(texts[0])
    
    print("\n--- [첫 번째 조각의 실제 텍스트 내용만 출력] ---")
    print(texts[0].page_content)
else:
    print("분할된 텍스트 조각이 없습니다. PDF 파일 내용을 확인해 주세요.")
