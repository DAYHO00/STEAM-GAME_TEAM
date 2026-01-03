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

### 다양한 추천 알고리즘
- Item-based Collaborative Filtering
- User-based Collaborative Filtering
- Jaccard Similarity 기반 개선 알고리즘
- BPR-MF (Bayesian Personalized Ranking – Matrix Factorization)

### 모델 학습 및 추천
- 사용자–아이템 상호작용 데이터 기반 학습
- PyTorch 기반 BPR-MF 모델 구현
- 학습 결과 모델 저장 및 재사용

### 성능 평가
- F1-score, Recall 기반 추천 성능 측정
- 알고리즘별 성능 비교 스크립트 제공

### 추천 결과 시각화
- React 기반 프론트엔드
- 사용자별 추천 게임 목록 제공

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
samadhi/
├── frontend/
│   ├── app/
│   │   ├── (with-navbar)/
│   │   │   ├── home/              # 메인 페이지
│   │   │   ├── ready/             # 운동 준비 (4단계)
│   │   │   └── record/            # 운동 기록
│   │   └── (without-navbar)/
│   │       ├── auth/              # 로그인/회원가입
│   │       └── workout/           # 실시간 운동
│   ├── components/
│   │   ├── ready/                 # 운동 준비 UI
│   │   ├── workout/               # 운동 중 UI
│   │   ├── video/                 # 비디오 재생
│   │   ├── webcam/                # 웹캠 처리
│   │   └── timeline/              # 타임라인 클리퍼
│   ├── lib/
│   │   ├── mediapipe/
│   │   │   └── angle-calculator.ts    # 관절 각도 계산
│   │   └── poseClassifier/
│   │       └── pose-classifier-with-vectorized.ts
│   └── store/                     # Zustand 상태 관리
│
└── backend/
    └── src/main/java/com/capstone/samadhi/
        ├── config/                # JWT, Security, S3, CORS
        ├── security/              # 인증/인가
        │   ├── jwt/              # JWT 필터 및 유틸
        │   └── service/          # UserDetailsService
        ├── record/               # 운동 기록
        │   ├── entity/           # Record, TimeLine
        │   └── service/          # 기록 저장/조회
        ├── video/                # 샘플 영상
        └── common/               # 공통 유틸 (S3, ResponseDto)
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
Steam 게임 추천 데이터는 Kaggle에서 제공합니다.
https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam
아래 파일을 다운로드하여 /backend/data 폴더에 위치시켜 주세요.
- games.csv
- recommendations.csv
- users.csv
```

### ⚙️ Backend 실행

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
