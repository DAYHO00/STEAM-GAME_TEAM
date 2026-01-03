# Samadhi - Personal Yoga Care Partner

MediaPipe 기반의 실시간 자세 추적 및 분석 시스템을 활용한 요가 운동 보조 웹 애플리케이션

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
Framework: Next.js 15 (App Router)
Language: TypeScript
UI: Tailwind CSS, Ant Design, Framer Motion
State: Zustand
AI/ML: MediaPipe Tasks Vision
```

### Backend
```
Framework: Spring Boot 3.3.2
Language: Java 17
Database: MySQL
Authentication: JWT
ORM: JPA/Hibernate
```

### Infrastructure
```
Containerization: Docker
Cloud: AWS (ECR, EC2, S3)
CI/CD: GitHub Actions
```

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

### Backend 설정

```bash
cd backend

# 환경 변수 설정 (.env)
DB_HOST=jdbc:mysql://localhost:3306/samadhi
DB_USERNAME=root
DB_PASSWORD=your_password
SECURITY_SECRET_KEY=your_secret_key
S3_ACCESS_KEY=your_access_key
S3_SECRET_KEY=your_secret_key
DEPLOY=local

./gradlew bootRun
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



