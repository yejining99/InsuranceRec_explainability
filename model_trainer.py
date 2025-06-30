"""
보험 추천 모델 학습 모듈
InsuranceEmbeddingTrainer 클래스 - 임베딩 모델 학습 및 평가
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import random
from typing import List, Tuple, Dict
import os
from datetime import datetime

from config import MODEL_PATHS, TRAINING_CONFIG

# 로깅 설정
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


class InsuranceEmbeddingTrainer:
    """보험 추천을 위한 Embedding 학습기"""
    
    def __init__(self, 
                 model_path: str = None,
                 output_path: str = None):
        """
        Args:
            model_path: 사전 학습된 모델 경로
            output_path: 학습된 모델 저장 경로
        """
        self.model_path = model_path or MODEL_PATHS['pretrained_model']
        self.output_path = output_path or MODEL_PATHS['trained_model']
        self.tokenizer = None
        self.model = None
        self.sentence_transformer = None
        
    def load_pretrained_model(self):
        """사전 학습된 모델 로드"""
        logging.info(f"Loading pretrained model from: {self.model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            logging.info("✅ Pretrained model loaded successfully")
        except Exception as e:
            logging.error(f"❌ Error loading pretrained model: {e}")
            # Fallback to online model
            logging.info("Falling back to online model...")
            self.sentence_transformer = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    def get_embeddings_manual(self, sentences: List[str]) -> torch.Tensor:
        """수동으로 임베딩 추출 (사전 학습된 모델 사용)"""
        if self.tokenizer is None or self.model is None:
            raise ValueError("Pretrained model not loaded. Call load_pretrained_model() first.")
        
        inputs = self.tokenizer(
            sentences, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=TRAINING_CONFIG['max_length']
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            # Mean pooling
            pooled = (embeddings * attention_mask).sum(1) / attention_mask.sum(1)
        
        return pooled
    
    def prepare_sentence_transformer(self):
        """Sentence Transformer 모델 준비"""
        if self.sentence_transformer is None:
            # 로컬 모델을 Sentence Transformer로 변환
            logging.info("Converting pretrained model to Sentence Transformer...")
            self.sentence_transformer = SentenceTransformer(self.model_path)
            
            # GPU 사용 가능시 이동
            if torch.cuda.is_available():
                self.sentence_transformer = self.sentence_transformer.to('cuda')
    
    def prepare_training_data(self, df: pd.DataFrame) -> List[InputExample]:
        """올리브영 방식: Positive pairs만 사용하여 학습 데이터 준비"""
        examples = []
        
        logging.info(f"Preparing training data with {len(df)} samples")
        
        # Label=1인 positive pairs만 사용
        positive_df = df[df['label'] == 1]
        
        for _, row in positive_df.iterrows():
            # Label 없이 positive pair만 생성
            # MultipleNegativesRankingLoss가 배치 내에서 자동으로 negative sampling 수행
            examples.append(InputExample(texts=[row['query'], row['value']]))
        
        logging.info(f"Created {len(examples)} positive pairs")
        logging.info("Negative pairs will be automatically generated in-batch by MultipleNegativesRankingLoss")
        
        # 데이터 셔플
        random.shuffle(examples)
        return examples
    
    def create_simple_evaluator(self, eval_df: pd.DataFrame):
        """간단한 평가자 생성 (상관계수 대신 정확도 기반)"""
        # 평가 데이터를 binary classification 형태로 변환
        sentences1 = []
        sentences2 = []
        scores = []
        
        queries = eval_df['query'].tolist()
        values = eval_df['value'].tolist()
        
        # Positive pairs (실제 매칭)
        for query, value in zip(queries, values):
            sentences1.append(query)
            sentences2.append(value)
            scores.append(1)  # positive
        
        # Negative pairs (랜덤 매칭)
        for i in range(min(len(queries), 20)):  # 최대 20개 negative
            neg_idx = (i + len(queries) // 2) % len(values)  # 다른 인덱스 선택
            sentences1.append(queries[i])
            sentences2.append(values[neg_idx])
            scores.append(0)  # negative
        
        logging.info(f"Created binary evaluator with {len(sentences1)} pairs")
        logging.info(f"Positive: {scores.count(1)}, Negative: {scores.count(0)}")
        
        return BinaryClassificationEvaluator(
            sentences1=sentences1,
            sentences2=sentences2,
            labels=scores,
            name="insurance_binary_eval"
        )
    
    def train_model_with_safe_evaluator(self, 
                                       train_df: pd.DataFrame,
                                       eval_df: pd.DataFrame = None,
                                       epochs: int = None,
                                       batch_size: int = None,
                                       learning_rate: float = None,
                                       warmup_steps: int = None):
        """안전한 평가자를 사용한 학습"""
        
        # 기본값 설정
        epochs = epochs or TRAINING_CONFIG['epochs']
        batch_size = batch_size or TRAINING_CONFIG['batch_size']
        learning_rate = learning_rate or TRAINING_CONFIG['learning_rate']
        
        # Sentence Transformer 준비
        self.prepare_sentence_transformer()
        
        # 학습 데이터 준비 (positive pairs만)
        train_examples = self.prepare_training_data(train_df)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        # Loss function 설정 (MultipleNegativesRankingLoss)
        train_loss = losses.MultipleNegativesRankingLoss(model=self.sentence_transformer)
        
        # Warmup steps 계산
        if warmup_steps is None:
            warmup_steps = int(len(train_dataloader) * epochs * 0.1)
        
        # 안전한 평가자 생성
        evaluator = None
        if eval_df is not None and len(eval_df) >= 5:
            try:
                evaluator = self.create_simple_evaluator(eval_df)
                logging.info("✅ Safe binary evaluator created")
            except Exception as e:
                logging.warning(f"⚠️ Failed to create safe evaluator: {e}")
                evaluator = None
        
        # 모델 학습
        logging.info("🚀 Starting training with safe evaluator...")
        logging.info(f"📊 Training examples: {len(train_examples)}")
        logging.info(f"📦 Batch size: {batch_size}")
        logging.info(f"🔄 Epochs: {epochs}")
        logging.info(f"🔥 Learning rate: {learning_rate}")
        logging.info(f"⚡ Warmup steps: {warmup_steps}")
        logging.info(f"📈 Safe evaluator enabled: {evaluator is not None}")
        
        if evaluator is not None:
            self.sentence_transformer.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                warmup_steps=warmup_steps,
                evaluator=evaluator,
                evaluation_steps=35,  # 더 자주 평가 (매 에포크마다)
                output_path=self.output_path,
                save_best_model=True,
                show_progress_bar=True,
                optimizer_params={'lr': learning_rate},
                use_amp=False,  # Mixed precision 비활성화
                checkpoint_save_steps=None,  # 체크포인트 저장 비활성화
                checkpoint_save_total_limit=None
            )
        else:
            # Fallback: 평가자 없이 학습
            self.sentence_transformer.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                warmup_steps=warmup_steps,
                output_path=self.output_path,
                show_progress_bar=True,
                optimizer_params={'lr': learning_rate}
            )
        
        logging.info(f"✅ Training completed. Saved to: {self.output_path}")
    
    def load_trained_model(self, model_path: str = None):
        """학습된 모델 로드"""
        if model_path is None:
            model_path = self.output_path
        
        logging.info(f"Loading trained model from: {model_path}")
        self.sentence_transformer = SentenceTransformer(model_path)
    
    def evaluate_model_performance(self, test_df: pd.DataFrame) -> Dict:
        """모델 성능 평가"""
        if self.sentence_transformer is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        # 임베딩 생성
        queries = test_df['query'].tolist()
        values = test_df['value'].tolist()
        labels = test_df['label'].tolist()
        
        logging.info("🔍 Generating embeddings for evaluation...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        query_embeddings = self.sentence_transformer.encode(queries, show_progress_bar=True, device=device)
        value_embeddings = self.sentence_transformer.encode(values, show_progress_bar=True, device=device)
        
        # 유사도 계산
        similarities = []
        for q_emb, v_emb in zip(query_embeddings, value_embeddings):
            sim = cosine_similarity([q_emb], [v_emb])[0][0]
            similarities.append(sim)
        
        # 성능 지표 계산
        # 유사도를 이진 분류로 변환 (임계값 0.5)
        predictions = [1 if sim > 0.5 else 0 for sim in similarities]
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1_score': f1_score(labels, predictions, zero_division=0),
            'avg_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities)
        }
        
        return metrics
    
    def recommend_products(self, 
                          user_query: str, 
                          product_values: List[str], 
                          top_k: int = 5) -> List[Tuple[int, float]]:
        """사용자 쿼리에 대해 상품 추천"""
        if self.sentence_transformer is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        # 임베딩 생성
        query_embedding = self.sentence_transformer.encode([user_query])
        product_embeddings = self.sentence_transformer.encode(product_values)
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(query_embedding, product_embeddings)[0]
        
        # 상위 k개 추천
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        recommendations = [(idx, similarities[idx]) for idx in top_indices]
        
        return recommendations

    def show_tokenization(self, df: pd.DataFrame, column: str = 'query', max_rows: int = 3):
        """특정 컬럼의 텍스트를 tokenizer로 토큰화하여 확인하는 함수"""
        if self.tokenizer is None:
            raise ValueError("tokenizer가 초기화되지 않았습니다. load_pretrained_model()을 먼저 호출하세요.")
        
        sample_texts = df[column].head(max_rows).tolist()
        
        for i, text in enumerate(sample_texts):
            tokens = self.tokenizer.tokenize(text)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            print(f"\n--- 샘플 {i+1} ---")
            print(f"원문: {text}")
            print(f"토큰: {tokens}")
            print(f"토큰 ID: {token_ids}")


def split_data_by_date(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """날짜별로 데이터를 분할하는 함수"""
    if 'date' in df.columns:
        unique_dates = sorted(df['date'].unique())
        train_dates = unique_dates[:-1]  # 마지막 날짜 제외하고 학습
        test_dates = [unique_dates[-1]]  # 마지막 날짜를 테스트
        
        train_df = df[df['date'].isin(train_dates)]
        test_df = df[df['date'].isin(test_dates)]
        
        logging.info(f"📅 Train dates: {train_dates}")
        logging.info(f"📅 Test dates: {test_dates}")
    else:
        # 날짜 정보가 없으면 랜덤 분할
        train_df, test_df = train_test_split(
            df, 
            test_size=TRAINING_CONFIG['test_size'], 
            random_state=TRAINING_CONFIG['random_state']
        )
    
    # 학습/검증 분할
    train_df, eval_df = train_test_split(
        train_df, 
        test_size=TRAINING_CONFIG['eval_size'], 
        random_state=TRAINING_CONFIG['random_state']
    )
    
    logging.info(f"📈 Train samples: {len(train_df)}")
    logging.info(f"📊 Eval samples: {len(eval_df)}")
    logging.info(f"🧪 Test samples: {len(test_df)}")
    
    return train_df, eval_df, test_df


if __name__ == "__main__":
    # 테스트 실행
    from data_preprocessing import preprocess_insurance_data
    from data_converter import InsuranceDataConverter
    
    print("🧪 InsuranceEmbeddingTrainer 테스트 시작")
    
    # 데이터 로드 및 전처리
    df = preprocess_insurance_data()
    sample_df = df.head(1000)  # 샘플 1000개만 테스트
    
    # 데이터 변환
    converter = InsuranceDataConverter()
    result_df = converter.convert_dataframe(sample_df)
    
    # 트레이너 초기화
    trainer = InsuranceEmbeddingTrainer()
    
    # 사전 학습된 모델 로드 테스트
    try:
        trainer.load_pretrained_model()
        sample_sentences = result_df['query'].head(3).tolist()
        embeddings = trainer.get_embeddings_manual(sample_sentences)
        print(f"✅ 사전 학습된 모델 임베딩 테스트: {embeddings.shape}")
    except Exception as e:
        print(f"⚠️ 사전 학습된 모델 로드 실패: {e}")
    
    print("🎯 InsuranceEmbeddingTrainer 테스트 완료") 