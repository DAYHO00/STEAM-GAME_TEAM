🎮 Steam Game Recommendation System

- backend는 Python 기반으로 모델 학습 및 추천 로직을 수행.
- frontend는 React 기반으로 결과를 시각화.

📂 프로젝트 구조
steam/
│
├─ backend/ # 추천 알고리즘 및 데이터 처리
│ ├─ data/ # 원본 데이터 및 학습 결과
│ │ ├─ games.csv
│ │ ├─ recommendations.csv
│ │ ├─ users.csv
│ │ └─ model/
│ │ ├─ bpr_model.pt
│ │ └─ bpr_meta.pkl
│ │
│ ├─ evaluation/ # 검증용 스크립트 (F1-score, Recall 등)
│ ├─ processed/ # test, train, valid data
│ ├─ recommend/ # 추천 알고리즘 구현부
│ ├─ main.py # 백엔드 실행 진입점 (FastAPI)
│ ├─ model.py # BPR-MF 모델 학습
│ ├─ preprocess.py # 데이터 전처리
│ └─ .venv/ # 가상환경
│
├─ frontend/ # React 프론트엔드
│ ├─ public/
│ ├─ src/
│ └─ README.md # CRA 기본 설명서 (자동 생성)
│
└─ README.md # 전체 프로젝트 설명서

📊 1. 데이터 다운로드

Steam 게임 추천 데이터는 Kaggle에서 제공합니다:
(https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam?select=recommendations.csv)
아래 세 개의 파일을 다운로드하여 /backend/data 폴더에 넣으세요:

a. games.csv
b. recommendations.csv
c. users.csv

⚙️ 2. 실행 방법

Backend (Python)

1. 가상환경 활성화
   cd backend
   ..venv\Scripts\Activate.ps1

2. 백엔드 서버 실행
   uvicorn main:app --reload

3. 데이터 전처리
   python preprocess.py
   → train, valid, test 데이터가 자동으로 분리됩니다.

4. 모델 학습
   python model.py
   → 학습 완료 후 /backend/data/model에
   bpr_meta.pkl과 bpr_model.pt가 생성됩니다.

Frontend (React)

1. 프론트엔드 실행
   cd frontend
   npm install
   npm start
   → 기본 포트(localhost:3000)에서 실행됩니다.

🧾 3. 검증 방법

/backend/evaluation 폴더에는 다양한 추천 알고리즘의 성능을 측정하는 스크립트가 있습니다.
다음 명령어로 검증 지표(F1-Score, Recall 등)를 확인할 수 있습니다.

python item_based_test.py
python item_based_test_advanced.py
python user_based_test.py
python user_based_test_advanced.py
python model_based_test.py

🧾 4. 핵심 알고리즘 위치

모든 추천 알고리즘은 /backend/recommend 폴더에 구현되어 있습니다.

item_based.py — 아이템 기반 협업 필터링

item_based_advanced.py — Jaccard 유사도 기반 개선 버전

user_based.py — 사용자 기반 협업 필터링

user_based_advanced.py — 사용자 Jaccard 개선 버전

model_based.py — BPR-MF 모델 기반 추천

🧾 5. 기술 스택

Frontend : React.js, HTML, CSS, JavaScript
Backend : Python, FastAPI, Uvicorn
Machine : Learning NumPy, Pandas, SciPy, PyTorch
Dataset : Kaggle Steam Recommendation Dataset
Environment : Windows PowerShell + venv
test2
