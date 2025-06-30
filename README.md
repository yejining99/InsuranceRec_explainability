# 🏥 보험 추천 시스템 (Insurance Recommendation System)

딥러닝 기반 보험 상품 추천 시스템으로, 고객의 프로필과 선호도를 분석하여 최적의 보험 상품을 추천합니다.

## 📋 프로젝트 개요

이 프로젝트는 자연어 처리(NLP)와 임베딩 기술을 활용하여 고객의 요구사항과 보험 상품 간의 유사도를 계산하고, 가장 적합한 상품을 추천하는 시스템입니다.

### 🎯 주요 기능

- **데이터 전처리**: 보험 데이터 정제 및 변환
- **Query-Value 페어 생성**: 고객 프로필과 상품 정보를 자연어로 변환
- **임베딩 모델 학습**: Sentence Transformer 기반 맞춤형 모델 학습
- **추천 시스템**: 코사인 유사도 기반 상품 추천
- **어텐션 분석**: 모델의 의사결정 과정 시각화 및 해석

## 📁 프로젝트 구조

```
InsuranceRec_explainability/
├── config.py                  # 설정 파일 (모델 경로, 하이퍼파라미터 등)
├── data_preprocessing.py      # 데이터 전처리 모듈
├── data_converter.py          # Query-Value 페어 변환 모듈
├── model_trainer.py           # 모델 학습 및 평가 모듈
├── attention_analyzer.py      # 어텐션 분석 모듈
├── main.py                    # 메인 실행 파일
├── requirements.txt           # 패키지 의존성
├── README.md                  # 프로젝트 문서
└── initialize.ipynb           # 원본 Jupyter 노트북
```

## 🛠️ 설치 방법

### 1. 환경 설정

```bash
# Python 3.8 이상 권장
python -m venv insurance_rec_env
source insurance_rec_env/bin/activate  # Windows: insurance_rec_env\Scripts\activate
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. 데이터 준비

- 보험 데이터 파일을 `/workspace/recomAI/aggregated_df_250625_info.parquet` 경로에 배치
- 또는 `config.py`에서 데이터 경로 수정

## 🚀 사용법

### 메인 파이프라인 실행

```bash
python main.py
```

실행 시 다음 옵션 중 선택:
1. **전체 파이프라인**: 데이터 전처리 → 모델 학습 → 평가 → 어텐션 분석
2. **빠른 테스트**: 소량 데이터로 빠른 테스트
3. **학습된 모델 테스트**: 기존 모델로 추천 테스트

### 개별 모듈 사용

#### 데이터 전처리
```python
from data_preprocessing import preprocess_insurance_data

# 데이터 전처리
df = preprocess_insurance_data('path/to/your/data.parquet')
```

#### Query-Value 페어 변환
```python
from data_converter import InsuranceDataConverter

converter = InsuranceDataConverter()
result_df = converter.convert_dataframe(df)
```

#### 모델 학습
```python
from model_trainer import InsuranceEmbeddingTrainer

trainer = InsuranceEmbeddingTrainer()
trainer.train_model_with_safe_evaluator(train_df, eval_df)
```

#### 어텐션 분석
```python
from attention_analyzer import AttentionAnalyzer

analyzer = AttentionAnalyzer()
analyzer.visualize_token_importance("고객 쿼리 텍스트")
```

## ⚙️ 설정 변경

`config.py` 파일에서 다음 설정들을 변경할 수 있습니다:

```python
# 모델 경로
MODEL_PATHS = {
    'pretrained_model': './MiniLM-L12-v2/0_Transformer',
    'trained_model': './trained_insurance_model_retoken'
}

# 학습 하이퍼파라미터
TRAINING_CONFIG = {
    'epochs': 4,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'max_length': 512
}

# 추천 설정
RECOMMENDATION_CONFIG = {
    'top_k': 5,
    'similarity_threshold': 0.5
}
```

## 📊 모델 성능 지표

시스템은 다음 지표들로 성능을 평가합니다:
- **정확도 (Accuracy)**
- **정밀도 (Precision)**
- **재현율 (Recall)**
- **F1 Score**
- **추천 적중률**

## 🔍 어텐션 분석 기능

모델의 의사결정 과정을 시각화하여 해석 가능한 AI를 구현합니다:

- **토큰 중요도 시각화**: 어떤 단어가 추천에 중요한지 확인
- **어텐션 히트맵**: 문장 내 토큰 간 관계 시각화
- **Query-Value 어텐션 분석**: 고객 요구사항과 상품 정보 간 매칭 과정 분석

## 📈 사용 예시

### 추천 결과 예시

```
👤 고객 프로필: 35세 남성 고객으로 직업등급 1급 사무직, 상해등급 1급 매우낮음에 해당합니다. 
   희망하는 보험료는 5~10만원 이하이고 암, 사망 테마에 특별한 관심이 있습니다.

📋 추천 결과:
1. 유사도: 0.8234 - 무배당 종합보험 상품으로 월 보험료 8만 5천원입니다.
2. 유사도: 0.7891 - 암보험 프리미엄 상품으로 월 보험료 6만원입니다.
3. 유사도: 0.7456 - 건강보험 플러스 상품으로 월 보험료 12만원입니다.
```

## 🚨 주의사항

1. **GPU 메모리**: 대용량 데이터 처리 시 GPU 메모리 부족 가능
2. **데이터 경로**: `config.py`에서 데이터 파일 경로 확인 필요
3. **모델 저장**: 학습된 모델은 `./trained_insurance_model_retoken` 경로에 저장

## 🔧 문제 해결

### 공통 오류
- **모델 로드 실패**: 사전 학습된 모델 경로 확인
- **메모리 부족**: 배치 크기 줄이기 또는 샘플 크기 제한
- **데이터 파일 없음**: 데이터 파일 경로 및 존재 여부 확인

### 성능 최적화
- **GPU 사용**: CUDA 설치 후 GPU 사용 활성화
- **배치 크기 조정**: 메모리에 맞게 배치 크기 조정
- **모델 크기**: 더 작은 모델 사용 시 속도 향상

## 📚 참고 자료

- [Sentence Transformers](https://www.sbert.net/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)

## 📧 문의

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.

---

**개발자**: Insurance Recommendation Team  
**업데이트**: 2024년 현재 