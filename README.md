# 🎮 Steam Game Recommendation System

사용자 행동 데이터를 기반으로 다양한 추천 알고리즘을 적용한 게임 추천 웹 애플리케이션

## 목차

- [주요 기능](#-주요-기능)
- [기술 스택](#-기술-스택)
- [프로젝트 구조](#-프로젝트-구조)
- [시작하기](#-시작하기)
- [핵심 기능](#-핵심-기능)

---

## 🎯 주요 기능

### 실시간 자세 분석
- MediaPipe Pose Landmarker를 활용한 33개 관절 포인트 추적
- 3D 공간에서의 정확한 관절 각도 계산
- 코사인 유사도 실시간 유사도 측정 (0-100점)

### 다양한 운동 방식
- **샘플 영상**: 추천 요가 동작 영상 제공
- **화면 공유**: 유튜브 영상 활용 운동
- **웹캠 연동**: 실시간 자세 비교 및 피드백

### 자세 분류 시스템
- 40가지 요가 자세 자동 인식 (Plank, Warrior, Tree, Bridge 등)
- 좌우 반전 자동 대응
- 벡터화된 자세 데이터 기반 분류 (임계값 90점)

### 운동 기록 관리
- 타임라인별 자세 분석 및 점수 기록
- 필터링 및 검색 기능
- 상세 운동 내역 조회


---

## 🛠 기술 스택

### Frontend
```
Framework: React (CRA)
Language: JavaScript
UI: HTML, CSS
```

### Backend
```
Framework: FastAPI
Language: Python
Server: Uvicorn
```

### Machine Learning
```
Libraries: NumPy, Pandas, SciPy
Deep Learning: PyTorch
Model: BPR-MF
```
### Environment
```
OS: Windows
Virtual Env: Python venv
Dataset: Kaggle Steam Recommendation Dataset

---

## 📁 프로젝트 구조

```
steam/
│
├── backend/
│ ├── data/
│ │ ├── games.csv
│ │ ├── recommendations.csv
│ │ ├── users.csv
│ │ └── model/
│ │ ├── bpr_model.pt
│ │ └── bpr_meta.pkl
│ │
│ ├── processed/
│ ├── evaluation/
│ ├── recommend/
│ │ ├── item_based.py
│ │ ├── item_based_advanced.py
│ │ ├── user_based.py
│ │ ├── user_based_advanced.py
│ │ └── model_based.py
│ │
│ ├── preprocess.py
│ ├── model.py
│ ├── main.py
│ └── .venv/
│
├── frontend/
│ ├── public/
│ ├── src/
│ └── README.md
│
└── README.md
```

---

## 🚀 시작하기

### 사전 요구사항
- Python 3.9+
- Node.js 16+
- npm
- Kaggle Steam Dataset


### 📊 데이터 다운로드
```bash
게임 추천 데이터는 Kaggle에서 제공합니다.
https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam
아래 파일을 다운로드하여 `/backend/data` 폴더에 위치시켜 주세요.

- games.csv
- recommendations.csv
- users.csv
```

# ⚙️ Backend 실행

```
cd backend
.venv\Scripts\Activate.ps1
uvicorn main:app --reload
```

### 데이터 전처리
```
python preprocess.py
```

### 모델 학습
```
python model.py
```

### 🖥 Frontend 실행
```
cd frontend
npm install
npm start
```
---

## 💡 핵심 기능

### 1. 데이터 전처리

```typescript
- 사용자–게임 상호작용 데이터 정제
- Train / Valid / Test 분리
```

### 2. 추천 알고리즘

```
- Item-based / User-based 협업 필터링
- Jaccard Similarity 기반 개선 버전
- BPR-MF 모델 기반 추천
```

### 3. 성능 평가

```
- F1-score
- Recall
- 알고리즘별 성능 비교
```

---

## 🔄 데이터 흐름

```
Kaggle Dataset
    ↓
데이터 전처리
    ↓
추천 모델 학습
    ↓
추천 결과 생성
    ↓
성능 평가
    ↓
Frontend 시각화
```
---
