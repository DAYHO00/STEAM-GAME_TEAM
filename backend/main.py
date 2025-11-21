from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # 👈 CORS 미들웨어 임포트
from recommend import recommend_by_user, recommend_by_item

app = FastAPI()

# ----------------------------------------------------
# 🌟 CORS 설정 추가 🌟
# ----------------------------------------------------
origins = [
    "http://localhost:3000",  # 👈 React 앱의 주소
    # "http://127.0.0.1:3000", # 필요한 경우 추가
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # 허용할 Origin 목록
    allow_credentials=True,         # 쿠키 등 자격 증명 허용 여부
    allow_methods=["*"],            # 모든 HTTP 메소드 허용 (GET, POST 등)
    allow_headers=["*"],            # 모든 헤더 허용
)
# ----------------------------------------------------

@app.get("/recommend/user/{user_id}")
def get_user_based_recommendation(user_id: int):
    return recommend_by_user(user_id)

@app.get("/recommend/item/{app_id}")
def get_item_based_recommendation(app_id: int):
    return recommend_by_item(app_id)

@app.get("/recommend/model/{user_id}")
def get_model_based_recommendation(user_id: int):
    return recommend_by_model(user_id)