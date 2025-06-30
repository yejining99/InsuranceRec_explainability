"""
보험 추천 시스템 Baseline 모델들
다양한 baseline 모델을 구현하여 현재 모델과 성능 비교
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.models import Word2Vec
from typing import List, Tuple, Dict
import logging
import warnings
warnings.filterwarnings('ignore')

from config import TRAINING_CONFIG


class BaselineModel:
    """Baseline 모델의 기본 클래스"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
    
    def fit(self, train_df: pd.DataFrame):
        """모델 학습"""
        raise NotImplementedError
    
    def get_similarities(self, queries: List[str], values: List[str]) -> List[float]:
        """쿼리와 값 간의 유사도 계산"""
        raise NotImplementedError
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict:
        """모델 성능 평가"""
        queries = test_df['query'].tolist()
        values = test_df['value'].tolist()
        labels = test_df['label'].tolist()
        
        similarities = self.get_similarities(queries, values)
        
        # 유사도를 이진 분류로 변환 (임계값 0.5)
        predictions = [1 if sim > 0.5 else 0 for sim in similarities]
        
        metrics = {
            'model_name': self.name,
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1_score': f1_score(labels, predictions, zero_division=0),
            'avg_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities)
        }
        
        return metrics


class RandomBaseline(BaselineModel):
    """랜덤 베이스라인 - 무작위 유사도 반환"""
    
    def __init__(self):
        super().__init__("Random Baseline")
        np.random.seed(TRAINING_CONFIG['random_state'])
    
    def fit(self, train_df: pd.DataFrame):
        """학습 불필요"""
        self.is_trained = True
        logging.info(f"✅ {self.name} 준비 완료")
    
    def get_similarities(self, queries: List[str], values: List[str]) -> List[float]:
        """무작위 유사도 반환 (0~1 사이)"""
        return np.random.uniform(0, 1, len(queries)).tolist()


class TfidfBaseline(BaselineModel):
    """TF-IDF 기반 베이스라인"""
    
    def __init__(self, max_features: int = 5000):
        super().__init__("TF-IDF Baseline")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words=None,  # 한국어는 별도 처리 필요
            lowercase=True
        )
    
    def fit(self, train_df: pd.DataFrame):
        """TF-IDF 벡터라이저 학습"""
        all_texts = train_df['query'].tolist() + train_df['value'].tolist()
        self.vectorizer.fit(all_texts)
        self.is_trained = True
        logging.info(f"✅ {self.name} 학습 완료 (특성 수: {len(self.vectorizer.vocabulary_)})")
    
    def get_similarities(self, queries: List[str], values: List[str]) -> List[float]:
        """TF-IDF 벡터 간 코사인 유사도 계산"""
        query_vectors = self.vectorizer.transform(queries)
        value_vectors = self.vectorizer.transform(values)
        
        similarities = []
        for i in range(len(queries)):
            sim = cosine_similarity(query_vectors[i], value_vectors[i])[0][0]
            similarities.append(sim)
        
        return similarities


class CountVectorizerBaseline(BaselineModel):
    """Count Vectorizer 기반 베이스라인"""
    
    def __init__(self, max_features: int = 5000):
        super().__init__("Count Vectorizer Baseline")
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            lowercase=True
        )
    
    def fit(self, train_df: pd.DataFrame):
        """Count Vectorizer 학습"""
        all_texts = train_df['query'].tolist() + train_df['value'].tolist()
        self.vectorizer.fit(all_texts)
        self.is_trained = True
        logging.info(f"✅ {self.name} 학습 완료 (특성 수: {len(self.vectorizer.vocabulary_)})")
    
    def get_similarities(self, queries: List[str], values: List[str]) -> List[float]:
        """Count 벡터 간 코사인 유사도 계산"""
        query_vectors = self.vectorizer.transform(queries)
        value_vectors = self.vectorizer.transform(values)
        
        similarities = []
        for i in range(len(queries)):
            sim = cosine_similarity(query_vectors[i], value_vectors[i])[0][0]
            similarities.append(sim)
        
        return similarities


class Word2VecBaseline(BaselineModel):
    """Word2Vec 기반 베이스라인"""
    
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1):
        super().__init__("Word2Vec Baseline")
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None
    
    def _preprocess_text(self, text: str) -> List[str]:
        """텍스트 전처리 (간단한 공백 기반 토큰화)"""
        return text.lower().split()
    
    def _get_sentence_vector(self, sentence: str) -> np.ndarray:
        """문장의 Word2Vec 평균 벡터 계산"""
        words = self._preprocess_text(sentence)
        vectors = []
        
        for word in words:
            if word in self.model.wv:
                vectors.append(self.model.wv[word])
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.vector_size)
    
    def fit(self, train_df: pd.DataFrame):
        """Word2Vec 모델 학습"""
        all_texts = train_df['query'].tolist() + train_df['value'].tolist()
        sentences = [self._preprocess_text(text) for text in all_texts]
        
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            seed=TRAINING_CONFIG['random_state']
        )
        
        self.is_trained = True
        vocab_size = len(self.model.wv.key_to_index)
        logging.info(f"✅ {self.name} 학습 완료 (어휘 수: {vocab_size})")
    
    def get_similarities(self, queries: List[str], values: List[str]) -> List[float]:
        """Word2Vec 벡터 간 코사인 유사도 계산"""
        similarities = []
        
        for query, value in zip(queries, values):
            query_vec = self._get_sentence_vector(query)
            value_vec = self._get_sentence_vector(value)
            
            # 코사인 유사도 계산
            if np.linalg.norm(query_vec) > 0 and np.linalg.norm(value_vec) > 0:
                sim = cosine_similarity([query_vec], [value_vec])[0][0]
            else:
                sim = 0.0
            
            similarities.append(sim)
        
        return similarities


class PretrainedSentenceTransformerBaseline(BaselineModel):
    """사전 학습된 SentenceTransformer 베이스라인 (fine-tuning 없음)"""
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        super().__init__(f"Pretrained {model_name}")
        self.model_name = model_name
        self.model = None
    
    def fit(self, train_df: pd.DataFrame):
        """사전 학습된 모델 로드 (추가 학습 없음)"""
        self.model = SentenceTransformer(self.model_name)
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
        
        self.is_trained = True
        logging.info(f"✅ {self.name} 로드 완료")
    
    def get_similarities(self, queries: List[str], values: List[str]) -> List[float]:
        """사전 학습된 모델로 임베딩 후 코사인 유사도 계산"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        query_embeddings = self.model.encode(queries, device=device)
        value_embeddings = self.model.encode(values, device=device)
        
        similarities = []
        for q_emb, v_emb in zip(query_embeddings, value_embeddings):
            sim = cosine_similarity([q_emb], [v_emb])[0][0]
            similarities.append(sim)
        
        return similarities


class HuggingFaceBaseline(BaselineModel):
    """HuggingFace Transformers 기반 베이스라인"""
    
    def __init__(self, model_name: str = 'klue/bert-base'):
        super().__init__(f"HuggingFace {model_name}")
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def fit(self, train_df: pd.DataFrame):
        """HuggingFace 모델 로드"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.is_trained = True
        logging.info(f"✅ {self.name} 로드 완료")
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """텍스트 리스트를 임베딩으로 변환"""
        embeddings = []
        
        for text in texts:
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, 
                truncation=True, max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                embeddings_tensor = outputs.last_hidden_state
                pooled = (embeddings_tensor * attention_mask).sum(1) / attention_mask.sum(1)
                embeddings.append(pooled.cpu().numpy()[0])
        
        return np.array(embeddings)
    
    def get_similarities(self, queries: List[str], values: List[str]) -> List[float]:
        """HuggingFace 모델 임베딩 간 코사인 유사도 계산"""
        similarities = []
        
        # 배치 처리를 위해 작은 청크로 나누어 처리
        chunk_size = 8
        
        for i in range(0, len(queries), chunk_size):
            end_idx = min(i + chunk_size, len(queries))
            
            query_chunk = queries[i:end_idx]
            value_chunk = values[i:end_idx]
            
            query_embeddings = self._get_embeddings(query_chunk)
            value_embeddings = self._get_embeddings(value_chunk)
            
            for q_emb, v_emb in zip(query_embeddings, value_embeddings):
                sim = cosine_similarity([q_emb], [v_emb])[0][0]
                similarities.append(sim)
        
        return similarities


def get_all_baseline_models() -> List[BaselineModel]:
    """모든 baseline 모델 리스트 반환"""
    return [
        RandomBaseline(),
        TfidfBaseline(max_features=3000),
        CountVectorizerBaseline(max_features=3000),
        Word2VecBaseline(vector_size=100),
        PretrainedSentenceTransformerBaseline('paraphrase-multilingual-MiniLM-L12-v2'),
        PretrainedSentenceTransformerBaseline('sentence-transformers/paraphrase-multilingual-mpnet-base-v2'),
        HuggingFaceBaseline('klue/bert-base')
    ]


def run_baseline_comparison(train_df: pd.DataFrame, 
                           test_df: pd.DataFrame,
                           selected_models: List[str] = None) -> pd.DataFrame:
    """
    모든 baseline 모델들을 학습하고 성능 비교
    
    Args:
        train_df: 학습 데이터
        test_df: 테스트 데이터
        selected_models: 선택할 모델 이름 리스트 (None이면 모든 모델)
    
    Returns:
        성능 비교 결과 DataFrame
    """
    logging.info("🏁 Baseline 모델 성능 비교 시작")
    logging.info("=" * 60)
    
    all_models = get_all_baseline_models()
    
    # 선택된 모델만 필터링
    if selected_models:
        all_models = [model for model in all_models if model.name in selected_models]
    
    results = []
    
    for i, model in enumerate(all_models):
        logging.info(f"🔄 {i+1}/{len(all_models)}: {model.name} 평가 중...")
        
        try:
            # 모델 학습/로드
            model.fit(train_df)
            
            # 성능 평가
            metrics = model.evaluate(test_df)
            results.append(metrics)
            
            logging.info(f"✅ {model.name} - F1: {metrics['f1_score']:.4f}, Acc: {metrics['accuracy']:.4f}")
            
        except Exception as e:
            logging.error(f"❌ {model.name} 평가 실패: {e}")
            # 실패한 경우 기본값 추가
            results.append({
                'model_name': model.name,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'avg_similarity': 0.0,
                'std_similarity': 0.0,
                'error': str(e)
            })
    
    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1_score', ascending=False)
    
    logging.info("=" * 60)
    logging.info("📊 Baseline 모델 성능 비교 완료")
    logging.info("=" * 60)
    
    return results_df


if __name__ == "__main__":
    # 테스트 실행
    from data_preprocessing import preprocess_insurance_data
    from data_converter import InsuranceDataConverter
    from model_trainer import split_data_by_date
    
    print("🧪 Baseline 모델 테스트 시작")
    
    # 데이터 로드 및 전처리
    df = preprocess_insurance_data()
    sample_df = df.head(1000)  # 샘플 1000개만
    
    # 데이터 변환
    converter = InsuranceDataConverter()
    result_df = converter.convert_dataframe(sample_df)
    
    # 데이터 분할
    train_df, eval_df, test_df = split_data_by_date(result_df)
    
    # 빠른 테스트를 위해 일부 모델만 선택
    selected_models = [
        "Random Baseline",
        "TF-IDF Baseline",
        "Pretrained paraphrase-multilingual-MiniLM-L12-v2"
    ]
    
    # Baseline 모델 비교 실행
    results_df = run_baseline_comparison(
        train_df=train_df.head(100),  # 빠른 테스트
        test_df=test_df.head(50),
        selected_models=selected_models
    )
    
    print("\n" + "=" * 60)
    print("📊 Baseline 모델 성능 비교 결과")
    print("=" * 60)
    print(results_df[['model_name', 'f1_score', 'accuracy', 'avg_similarity']].to_string(index=False))
    
    print("🎯 Baseline 모델 테스트 완료") 