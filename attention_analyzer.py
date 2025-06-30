"""
어텐션 분석 모듈
AttentionAnalyzer 클래스 - 보험 추천 모델의 어텐션 패턴 분석 도구
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

from config import MODEL_PATHS, ATTENTION_CONFIG


class AttentionAnalyzer:
    """보험 추천 모델의 어텐션 패턴 분석 도구"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or MODEL_PATHS['trained_model']
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        """모델과 토크나이저 로드"""
        print(f"📂 모델 로딩 중: {self.model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path, output_attentions=True)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ 모델 로드 완료: {self.model_path}")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            print("📂 기본 모델로 대체 시도...")
            # Fallback to pretrained model
            fallback_path = MODEL_PATHS['pretrained_model']
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_path)
            self.model = AutoModel.from_pretrained(fallback_path, output_attentions=True)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ 기본 모델 로드 완료: {fallback_path}")
    
    def clean_token(self, token: str) -> str:
        """토큰 정리 함수"""
        cleaned = token.replace('▁', '').replace('##', '')
        if cleaned in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>', '<unk>']:
            return cleaned
        if not cleaned.strip():
            return '[SPACE]'
        return cleaned
    
    def get_attention_weights(self, text: str) -> Tuple[List[str], torch.Tensor]:
        """텍스트에 대한 어텐션 가중치 추출"""
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, 
            truncation=True, max_length=128
        ).to(self.device)
        
        raw_tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        print(f"raw_tokens: {raw_tokens}")
        tokens = [self.clean_token(token) for token in raw_tokens]
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            attention_weights = torch.stack(outputs.attentions)
            attention_weights = attention_weights.squeeze(1)
        
        return tokens, attention_weights.cpu()
    
    def visualize_token_importance(self, text: str, layer_idx: int = -1, top_k: int = None):
        """토큰 중요도 막대 그래프"""
        if top_k is None:
            top_k = ATTENTION_CONFIG['top_k_tokens']
            
        tokens, attention_weights = self.get_attention_weights(text)
        
        if layer_idx == -1:
            layer_idx = attention_weights.shape[0] - 1
        attention = attention_weights[layer_idx].mean(dim=0)  # 모든 헤드 평균
        
        cls_attention = attention[0, :].numpy()  # CLS 토큰의 어텐션
        
        df = pd.DataFrame({
            'token': tokens,
            'importance': cls_attention,
            'position': range(len(tokens))
        })

        df = df.sort_values('importance', ascending=False)
        
        # 특수 토큰 제외
        special_tokens = ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>', '[SPACE]']
        df_filtered = df[~df['token'].isin(special_tokens)].head(top_k)
        
        # 시각화
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(df_filtered)), df_filtered['importance'], 
                       color=plt.cm.viridis(df_filtered['importance'] / df_filtered['importance'].max()))
        
        plt.xlabel('Tokens', fontsize=12)
        plt.ylabel('Attention Weight', fontsize=12)
        plt.title(f'Token Importance Analysis (Layer {layer_idx})\nText: "{text[:50]}..."', 
                 fontsize=14, pad=20)
        plt.xticks(range(len(df_filtered)), df_filtered['token'], rotation=45, ha='right')
        
        # 값 표시
        for i, (bar, val) in enumerate(zip(bars, df_filtered['importance'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        return df_filtered
    
    def visualize_attention_heatmap(self, text: str, layer_idx: int = -1, head_idx: int = 0):
        """어텐션 히트맵 시각화"""
        tokens, attention_weights = self.get_attention_weights(text)
        
        if layer_idx == -1:
            layer_idx = attention_weights.shape[0] - 1
        
        attention = attention_weights[layer_idx, head_idx].numpy()
        
        # 토큰 길이 제한
        max_tokens = ATTENTION_CONFIG['max_tokens']
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            attention = attention[:max_tokens, :max_tokens]
        
        # 토큰 길이 제한 (표시용)
        display_tokens = [token[:6] + '..' if len(token) > 8 else token for token in tokens]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            attention, xticklabels=display_tokens, yticklabels=display_tokens,
            cmap='Blues', annot=False, cbar_kws={'label': 'Attention Weight'}
        )
        
        plt.title(f'Attention Heatmap\nLayer {layer_idx}, Head {head_idx}\nText: "{text[:50]}..."', 
                 fontsize=14, pad=20)
        plt.xlabel('Key Tokens', fontsize=12)
        plt.ylabel('Query Tokens', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def analyze_query_value_attention(self, query: str, value: str):
        """쿼리와 상품 정보의 어텐션 패턴을 동시에 분석"""
        print("🔍 쿼리-상품 어텐션 패턴 분석")
        print("=" * 60)
        
        print(f"📝 쿼리: {query[:100]}...")
        print(f"🏷️ 상품: {value[:100]}...")
        print("-" * 40)
        
        # 쿼리 분석
        print("📊 쿼리 토큰 중요도:")
        query_analysis = self.visualize_token_importance(query)
        query_top_tokens = query_analysis['token'].head(3).tolist()
        print(f"상위 토큰: {', '.join(query_top_tokens)}")
        
        # 상품 분석
        print("\n📊 상품 토큰 중요도:")
        value_analysis = self.visualize_token_importance(value)
        value_top_tokens = value_analysis['token'].head(3).tolist()
        print(f"상위 토큰: {', '.join(value_top_tokens)}")
        
        # 쿼리 어텐션 히트맵
        print("\n📊 쿼리 어텐션 히트맵:")
        self.visualize_attention_heatmap(query, layer_idx=-1, head_idx=0)
        
        return {
            'query_analysis': query_analysis,
            'value_analysis': value_analysis,
            'query_top_tokens': query_top_tokens,
            'value_top_tokens': value_top_tokens
        }


def analyze_test_samples(test_df: pd.DataFrame, 
                        model_path: str = None,
                        sample_size: int = 3) -> Tuple['AttentionAnalyzer', List[Dict]]:
    """
    test_df에서 샘플을 뽑아서 어텐션 시각화
    
    Args:
        test_df: 테스트 데이터프레임 (query, value 컬럼 필요)
        model_path: 모델 경로
        sample_size: 분석할 샘플 개수
        
    Returns:
        (analyzer, results) 튜플
    """
    print("🔍 테스트 데이터 어텐션 분석 시작...")
    print(f"📊 test_df 크기: {test_df.shape}")
    print(f"📋 컬럼명: {list(test_df.columns)}")
    
    # 분석기 생성 및 모델 로드
    analyzer = AttentionAnalyzer(model_path)
    analyzer.load_model()
    
    # 샘플 데이터 추출
    sample_data = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)
    
    print(f"\n📊 {len(sample_data)}개 샘플 분석 시작")
    print("=" * 60)
    
    results = []
    
    for i, (idx, row) in enumerate(sample_data.iterrows()):
        query = str(row['query'])
        value = str(row['value'])
        
        print(f"\n🔍 샘플 {i+1}/{len(sample_data)}")
        
        # 쿼리-상품 어텐션 분석
        analysis_result = analyzer.analyze_query_value_attention(query, value)
        
        results.append({
            'index': idx,
            'query': query,
            'value': value,
            **analysis_result
        })
        
        print(f"✅ 샘플 {i+1} 분석 완료\n")
    
    print("🎉 모든 샘플 분석 완료!")
    return analyzer, results


def compare_attention_patterns(analyzer: 'AttentionAnalyzer', 
                              queries: List[str], 
                              labels: List[str] = None):
    """여러 쿼리의 어텐션 패턴 비교"""
    print("🔍 어텐션 패턴 비교 분석")
    print("=" * 50)
    
    if labels is None:
        labels = [f"Query {i+1}" for i in range(len(queries))]
    
    all_analyses = []
    
    for i, (query, label) in enumerate(zip(queries, labels)):
        print(f"\n📝 {label}: {query[:80]}...")
        analysis = analyzer.visualize_token_importance(query, top_k=5)
        
        top_tokens = analysis['token'].head(3).tolist()
        avg_importance = analysis['importance'].mean()
        
        all_analyses.append({
            'label': label,
            'query': query,
            'top_tokens': top_tokens,
            'avg_importance': avg_importance,
            'analysis': analysis
        })
        
        print(f"상위 토큰: {', '.join(top_tokens)}")
        print(f"평균 중요도: {avg_importance:.4f}")
    
    # 비교 요약
    print("\n📊 비교 요약:")
    print("-" * 30)
    for analysis in all_analyses:
        print(f"{analysis['label']:15}: {', '.join(analysis['top_tokens'][:2])} (평균: {analysis['avg_importance']:.3f})")
    
    return all_analyses


if __name__ == "__main__":
    # 테스트 실행
    from data_preprocessing import preprocess_insurance_data
    from data_converter import InsuranceDataConverter
    from model_trainer import split_data_by_date
    
    print("🧪 AttentionAnalyzer 테스트 시작")
    
    # 데이터 로드 및 전처리
    df = preprocess_insurance_data()
    sample_df = df.head(1000)  # 샘플 1000개만
    
    # 데이터 변환
    converter = InsuranceDataConverter()
    result_df = converter.convert_dataframe(sample_df)
    
    # 데이터 분할
    train_df, eval_df, test_df = split_data_by_date(result_df)
    
    # 어텐션 분석 실행
    try:
        analyzer, results = analyze_test_samples(
            test_df=test_df.head(10),  # 샘플 10개만
            sample_size=2
        )
        print("✅ 어텐션 분석 완료")
    except Exception as e:
        print(f"❌ 어텐션 분석 실패: {e}")
    
    print("🎯 AttentionAnalyzer 테스트 완료") 