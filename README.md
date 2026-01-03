🎮 Steam Game Recommendation System

사용자 행동 데이터를 기반으로 다양한 추천 알고리즘을 적용한
게임 추천 웹 애플리케이션

목차

주요 기능

기술 스택

프로젝트 구조

시작하기

핵심 기능

데이터 흐름

🎯 주요 기능
다양한 추천 알고리즘 제공

Item-based Collaborative Filtering

User-based Collaborative Filtering

Jaccard Similarity 기반 개선 알고리즘

BPR-MF (Bayesian Personalized Ranking – Matrix Factorization) 모델

모델 학습 및 추천

사용자–아이템 상호작용 데이터 기반 학습

PyTorch 기반 BPR-MF 모델 구현

학습 결과 모델 저장 및 재사용

성능 평가

F1-score, Recall 등 추천 성능 지표 측정

알고리즘별 성능 비교 스크립트 제공

추천 결과 시각화

React 기반 프론트엔드

사용자별 추천 게임 목록 확인

🛠 기술 스택
Frontend
Framework: React (CRA)
Language: JavaScript
UI: HTML, CSS

Backend
Framework: FastAPI
Language: Python
Server: Uvicorn

Machine Learning
Libraries: NumPy, Pandas, SciPy
Deep Learning: PyTorch
Model: BPR-MF

Environment
OS: Windows
Virtual Env: Python venv
Dataset: Kaggle Steam Recommendation Dataset

📁 프로젝트 구조
steam/
│
├── backend/
│   ├── data/                     # 원본 데이터 및 학습 결과
│   │   ├── games.csv
│   │   ├── recommendations.csv
│   │   ├── users.csv
│   │   └── model/
│   │       ├── bpr_model.pt
│   │       └── bpr_meta.pkl
│   │
│   ├── processed/                # train / valid / test 데이터
│   ├── evaluation/               # 성능 검증 스크립트
│   ├── recommend/                # 추천 알고리즘 구현
│   │   ├── item_based.py
│   │   ├── item_based_advanced.py
│   │   ├── user_based.py
│   │   ├── user_based_advanced.py
│   │   └── model_based.py
│   │
│   ├── preprocess.py             # 데이터 전처리
│   ├── model.py                  # BPR-MF 모델 학습
│   ├── main.py                   # FastAPI 실행 진입점
│   └── .venv/                    # 가상환경
│
├── frontend/
│   ├── public/
│   ├── src/
│   └── README.md                 # CRA 기본 README
│
└── README.md                     # 전체 프로젝트 설명서

🚀 시작하기
사전 요구사항

Python 3.9+

Node.js 16+

npm

Kaggle Steam Dataset

📊 데이터 다운로드

Steam 추천 데이터는 Kaggle에서 제공합니다.

https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam

아래 파일을 다운로드하여 /backend/data 폴더에 위치시켜 주세요.

games.csv

recommendations.csv

users.csv

⚙️ Backend 실행
cd backend

# 가상환경 활성화
.venv\Scripts\Activate.ps1

# 서버 실행
uvicorn main:app --reload

데이터 전처리
python preprocess.py


→ train / valid / test 데이터 자동 생성

모델 학습
python model.py


→ /backend/data/model에 학습 결과 저장

🖥 Frontend 실행
cd frontend
npm install
npm start


기본 주소: http://localhost:3000

💡 핵심 기능
1. 데이터 전처리
python preprocess.py


사용자–게임 상호작용 데이터 정제

학습/검증/테스트 데이터 분리

2. 추천 알고리즘
Item-based

아이템 간 유사도 기반 추천

User-based

사용자 간 유사도 기반 추천

Advanced Version

Jaccard Similarity 적용

희소성 문제 완화

Model-based

BPR-MF

implicit feedback 기반 랭킹 최적화

3. 성능 평가
python item_based_test.py
python user_based_test.py
python model_based_test.py


F1-score

Recall

알고리즘별 성능 비교

🔄 데이터 흐름
Kaggle Dataset
      ↓
데이터 전처리 (preprocess.py)
      ↓
Train / Valid / Test 분리
      ↓
추천 알고리즘 학습
      ↓
추천 결과 생성
      ↓
성능 평가
      ↓
Frontend 시각화
