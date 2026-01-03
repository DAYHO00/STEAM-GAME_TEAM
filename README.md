# 🎮 Steam Game Recommendation System

사용자 행동 데이터를 기반으로 다양한 추천 알고리즘을 적용한 게임 추천 웹 애플리케이션

## 목차

- [주요 기능](#-주요-기능)
- [기술 스택](#-기술-스택)
- [프로젝트 구조](#-프로젝트-구조)
- [시작하기](#-시작하기)
- [핵심 기능](#-핵심-기능)
- [데이터 흐름](#-데이터-흐름)
- 
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
OS: Windows
Virtual Env: Python venv
Dataset: Kaggle Steam Recommendation Dataset

---

## 📁 프로젝트 구조

```
├── steam/
├── ├── backend/
├── │   ├── data/
├── │   │   ├── games.csv
├── │   │   ├── recommendations.csv
├── │   │   ├── users.csv
├── │   │   └── model/
├── │   │       ├── bpr_model.pt
├── │   │       └── bpr_meta.pkl
├── │   │
├── │   ├── processed/
├── │   ├── evaluation/
├── │   ├── recommend/
├── │   │   ├── item_based.py
├── │   │   ├── item_based_advanced.py
├── │   │   ├── user_based.py
├── │   │   ├── user_based_advanced.py
├── │   │   └── model_based.py
├── │   │
├── │   ├── preprocess.py
├── │   ├── model.py
├── │   ├── main.py
├── │   └── .venv/
├── │
├── ├── frontend/
├── │   ├── public/
├── │   ├── src/
├── │   └── README.md
├── │
└── └── README.md
```

---

## 🚀 시작하기

### 사전 요구사항
- Python 3.9+
- Node.js 16+
- npm
- Kaggle Steam Recommendation Dataset

### 데이터 다운로드

```
Steam 게임 추천 데이터는 Kaggle에서 제공합니다.
https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam

아래 파일을 다운로드하여 /backend/data 폴더에 위치시켜 주세요.
- games.csv
- recommendations.csv
- users.csv
```

### Backend 실행

```bash
cd backend
.venv\Scripts\Activate.ps1
uvicorn main:app --reload

# 데이터 전처리
python preprocess.py
```

### Docker 배포

```bash
# 백엔드 이미지 빌드
docker build -t samadhi-api ./backend

# Docker Compose 실행
cd backend
docker-compose up -d
```

---

## 💡 핵심 기능

### 1. 자세 추적 및 각도 계산

```typescript
// 33개 관절 포인트에서 주요 각도 계산
calculateAllAngles(landmarks: Landmark[]): JointAngles
```

**계산 각도**
- 팔: 팔꿈치, 어깨 (좌/우)
- 다리: 무릎, 엉덩이 (좌/우)
- 몸통: 척추, 정렬
- 손목, 발목, 목

**특징**
- 3D 공간 벡터 기반 계산
- Dead Zone 필터 (±2도 떨림 방지)
- Visibility 필터링 (임계값 0.5)

### 2. 유사도 측정

```typescript
CalculateSimilarity(P1: number[], P2: number[], lambda: 1.0): number
```

- **코사인 유사도**: 자세 방향성 비교
- **결과**: 0-100점 범위

### 3. 자세 분류

```typescript
classifyPoseWithVectorized(vectorized: number[]): string
```

### 4. 타임라인 기록

```typescript
type Timeline = {
  pose: string;
  startTime: number;
  endTime: number;
  similarity: number;
};
```

운동 중 자세별 구간을 자동 기록하고 평균 유사도를 계산합니다.

---

## 🔄 데이터 흐름

```
웹캠/비디오 입력
    ↓
MediaPipe Pose Landmarker
    ↓
관절 좌표 추출 (33개)
    ↓
벡터화 및 정규화
    ↓
자세 분류 + 유사도 계산
    ↓
실시간 피드백
    ↓
타임라인 기록
    ↓
서버 저장 (MySQL + S3)
```


---


## 🚀 배포

### GitHub Actions CI/CD

```yaml
# main 브랜치 push 시 자동 배포
- Docker 이미지 빌드
- AWS ECR 푸시
- EC2 SSH 접속
- Docker Compose 재시작
```

### 환경 설정

**Production**
- `DEPLOY=prod` 환경변수 설정
- SameSite=None, Secure Cookie 사용
- AWS RDS MySQL
- AWS S3 파일 저장



