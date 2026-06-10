# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

def show_metadata(docs):
    if docs:
        print("[metadata]")
        print(list(docs[0].metadata.keys()))
        print("\n[examples]")
        max_key_length = max(len(k) for k in docs[0].metadata.keys())
        for k, v in docs[0].metadata.items():
            print(f"{k:<{max_key_length}} : {v}")

from langchain_community.document_loaders import PyPDFLoader

# 파일 경로 설정
FILE_PATH = "unsu.pdf"
loader = PyPDFLoader(FILE_PATH)

# PDF 로더 초기화
# docs = loader.load()  # PDF를 페이지 단위로 분할하여 로드
docs = loader.load_and_split()  # PDF를 일정한 단위로 분할하여 로드

# 문서의 내용 출력
# print(docs)
# print(docs[0])
# print(docs[0].page_content[:100])

if len(docs) > 0:
    print("=============", end="\n")
    print(docs[1])
    print("=============", end="\n")
    print(docs[1].page_content)
else:
    print(f"총 페이지 수: {len(docs)}")


# 메타데이터 출력
# show_metadata(docs)
