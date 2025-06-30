"""
보험 추천 모델 종합 성능 비교
현재 Fine-tuned 모델 vs 다양한 Baseline 모델들 성능 비교
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from model_trainer import InsuranceEmbeddingTrainer, split_data_by_date
from baseline_models import run_baseline_comparison, get_all_baseline_models
from data_preprocessing import preprocess_insurance_data
from data_converter import InsuranceDataConverter
from config import MODEL_PATHS, TRAINING_CONFIG


def setup_korean_font():
    """한글 폰트 설정"""
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'AppleGothic']
    plt.rcParams['axes.unicode_minus'] = False


def evaluate_finetuned_model(test_df: pd.DataFrame, model_path: str = None) -> Dict:
    """Fine-tuned 모델 성능 평가"""
    logging.info("🎯 Fine-tuned 모델 성능 평가 중...")
    
    if model_path is None:
        model_path = MODEL_PATHS['trained_model']
    
    trainer = InsuranceEmbeddingTrainer()
    
    try:
        trainer.load_trained_model(model_path)
        metrics = trainer.evaluate_model_performance(test_df)
        metrics['model_name'] = 'Fine-tuned Insurance Model'
        logging.info(f"✅ Fine-tuned 모델 - F1: {metrics['f1_score']:.4f}, Acc: {metrics['accuracy']:.4f}")
        return metrics
    
    except Exception as e:
        logging.error(f"❌ Fine-tuned 모델 평가 실패: {e}")
        return {
            'model_name': 'Fine-tuned Insurance Model',
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'avg_similarity': 0.0,
            'std_similarity': 0.0,
            'error': str(e)
        }


def comprehensive_model_comparison(train_df: pd.DataFrame, 
                                 test_df: pd.DataFrame,
                                 include_heavy_models: bool = False,
                                 finetuned_model_path: str = None) -> pd.DataFrame:
    """
    종합적인 모델 성능 비교
    
    Args:
        train_df: 학습 데이터
        test_df: 테스트 데이터
        include_heavy_models: 무거운 모델들(HuggingFace 등) 포함 여부
        finetuned_model_path: Fine-tuned 모델 경로
    
    Returns:
        모든 모델의 성능 비교 결과 DataFrame
    """
    logging.info("🏆 종합적인 모델 성능 비교 시작")
    logging.info("=" * 70)
    
    all_results = []
    
    # 1. Fine-tuned 모델 평가
    finetuned_result = evaluate_finetuned_model(test_df, finetuned_model_path)
    all_results.append(finetuned_result)
    
    # 2. Baseline 모델들 선택
    if include_heavy_models:
        # 모든 baseline 모델 포함
        selected_baselines = None
    else:
        # 빠른 baseline 모델들만 선택
        selected_baselines = [
            "Random Baseline",
            "TF-IDF Baseline",
            "Count Vectorizer Baseline", 
            "Word2Vec Baseline",
            "Pretrained paraphrase-multilingual-MiniLM-L12-v2"
        ]
    
    # 3. Baseline 모델들 평가
    baseline_results = run_baseline_comparison(
        train_df=train_df,
        test_df=test_df,
        selected_models=selected_baselines
    )
    
    # 4. 결과 통합
    for _, row in baseline_results.iterrows():
        all_results.append(row.to_dict())
    
    # 5. 최종 결과 DataFrame 생성
    final_results = pd.DataFrame(all_results)
    final_results = final_results.sort_values('f1_score', ascending=False)
    
    logging.info("=" * 70)
    logging.info("🎉 종합적인 모델 성능 비교 완료")
    logging.info("=" * 70)
    
    return final_results


def visualize_comparison_results(results_df: pd.DataFrame, save_path: str = None):
    """모델 비교 결과 시각화"""
    setup_korean_font()
    
    # 결과가 비어있는 경우 처리
    if results_df.empty:
        logging.warning("⚠️ 시각화할 결과가 없습니다.")
        return
    
    # 에러가 있는 모델 제외
    clean_results = results_df[~results_df['model_name'].str.contains('error', na=False)]
    
    if clean_results.empty:
        logging.warning("⚠️ 정상적으로 평가된 모델이 없습니다.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('보험 추천 모델 성능 비교', fontsize=16, fontweight='bold')
    
    # 1. F1 Score 비교
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(clean_results)), clean_results['f1_score'], 
                    color=plt.cm.viridis(np.linspace(0, 1, len(clean_results))))
    ax1.set_title('F1 Score 비교', fontweight='bold')
    ax1.set_ylabel('F1 Score')
    ax1.set_xticks(range(len(clean_results)))
    ax1.set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                        for name in clean_results['model_name']], rotation=45, ha='right')
    
    # 값 표시
    for i, (bar, val) in enumerate(zip(bars1, clean_results['f1_score'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Accuracy 비교
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(clean_results)), clean_results['accuracy'],
                    color=plt.cm.plasma(np.linspace(0, 1, len(clean_results))))
    ax2.set_title('Accuracy 비교', fontweight='bold')
    ax2.set_ylabel('Accuracy')
    ax2.set_xticks(range(len(clean_results)))
    ax2.set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                        for name in clean_results['model_name']], rotation=45, ha='right')
    
    # 값 표시
    for i, (bar, val) in enumerate(zip(bars2, clean_results['accuracy'])):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Precision vs Recall 산점도
    ax3 = axes[1, 0]
    scatter = ax3.scatter(clean_results['recall'], clean_results['precision'], 
                         s=100, c=clean_results['f1_score'], cmap='viridis', alpha=0.7)
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision vs Recall', fontweight='bold')
    
    # 모델 이름 표시
    for i, name in enumerate(clean_results['model_name']):
        short_name = name[:10] + '...' if len(name) > 10 else name
        ax3.annotate(short_name, 
                    (clean_results.iloc[i]['recall'], clean_results.iloc[i]['precision']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.colorbar(scatter, ax=ax3, label='F1 Score')
    
    # 4. 평균 유사도 분포
    ax4 = axes[1, 1]
    bars4 = ax4.bar(range(len(clean_results)), clean_results['avg_similarity'],
                    color=plt.cm.coolwarm(np.linspace(0, 1, len(clean_results))))
    ax4.set_title('평균 유사도 비교', fontweight='bold')
    ax4.set_ylabel('Average Similarity')
    ax4.set_xticks(range(len(clean_results)))
    ax4.set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                        for name in clean_results['model_name']], rotation=45, ha='right')
    
    # 값 표시
    for i, (bar, val) in enumerate(zip(bars4, clean_results['avg_similarity'])):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"📊 시각화 결과 저장: {save_path}")
    
    plt.show()


def print_detailed_comparison(results_df: pd.DataFrame):
    """상세한 비교 결과 출력"""
    print("\n" + "=" * 80)
    print("📊 모델 성능 비교 상세 결과")
    print("=" * 80)
    
    if results_df.empty:
        print("❌ 비교할 결과가 없습니다.")
        return
    
    # 에러가 있는 모델과 정상 모델 분리
    error_models = results_df[results_df.get('error', '').notna()]
    clean_results = results_df[results_df.get('error', '').isna() | (results_df.get('error', '') == '')]
    
    if not clean_results.empty:
        print(f"📈 정상 평가 모델: {len(clean_results)}개")
        print("-" * 80)
        
        # 성능 지표별 순위
        metrics = ['f1_score', 'accuracy', 'precision', 'recall', 'avg_similarity']
        
        for metric in metrics:
            if metric in clean_results.columns:
                print(f"\n🏆 {metric.upper()} 순위:")
                sorted_models = clean_results.sort_values(metric, ascending=False)
                for i, (_, row) in enumerate(sorted_models.head(5).iterrows()):
                    print(f"  {i+1}. {row['model_name'][:40]:40} {row[metric]:.4f}")
        
        # 최고 성능 모델
        best_model = clean_results.loc[clean_results['f1_score'].idxmax()]
        print(f"\n🥇 최고 성능 모델: {best_model['model_name']}")
        print(f"   F1 Score: {best_model['f1_score']:.4f}")
        print(f"   Accuracy: {best_model['accuracy']:.4f}")
        print(f"   Precision: {best_model['precision']:.4f}")
        print(f"   Recall: {best_model['recall']:.4f}")
        
        # Fine-tuned 모델 순위
        finetuned_row = clean_results[clean_results['model_name'].str.contains('Fine-tuned', na=False)]
        if not finetuned_row.empty:
            finetuned_rank = (clean_results['f1_score'] > finetuned_row.iloc[0]['f1_score']).sum() + 1
            print(f"\n🎯 Fine-tuned 모델 순위: {finetuned_rank}/{len(clean_results)}")
            print(f"   현재 모델이 {len(clean_results) - finetuned_rank}개 baseline을 앞섬")
    
    # 에러 모델 리포트
    if not error_models.empty:
        print(f"\n❌ 평가 실패 모델: {len(error_models)}개")
        print("-" * 40)
        for _, row in error_models.iterrows():
            print(f"  • {row['model_name']}: {row.get('error', 'Unknown error')}")
    
    print("=" * 80)


def quick_comparison(sample_size: int = 1000, test_size: int = 200):
    """빠른 모델 비교 (작은 데이터셋 사용)"""
    logging.info(f"⚡ 빠른 모델 비교 시작 (샘플 크기: {sample_size})")
    
    # 데이터 로드 및 전처리
    df = preprocess_insurance_data()
    sample_df = df.head(sample_size)
    
    # 데이터 변환
    converter = InsuranceDataConverter()
    result_df = converter.convert_dataframe(sample_df)
    
    # 데이터 분할
    train_df, eval_df, test_df = split_data_by_date(result_df)
    test_df = test_df.head(test_size)
    
    # 모델 비교 실행
    results = comprehensive_model_comparison(
        train_df=train_df,
        test_df=test_df,
        include_heavy_models=False  # 빠른 비교를 위해 무거운 모델 제외
    )
    
    # 결과 출력 및 시각화
    print_detailed_comparison(results)
    visualize_comparison_results(results, save_path='quick_model_comparison.png')
    
    return results


def full_comparison(sample_size: int = None):
    """전체 모델 비교 (모든 baseline 포함)"""
    logging.info("🔥 전체 모델 비교 시작 (모든 baseline 포함)")
    
    # 데이터 로드 및 전처리
    df = preprocess_insurance_data()
    if sample_size:
        df = df.head(sample_size)
    
    # 데이터 변환
    converter = InsuranceDataConverter()
    result_df = converter.convert_dataframe(df)
    
    # 데이터 분할
    train_df, eval_df, test_df = split_data_by_date(result_df)
    
    # 모델 비교 실행
    results = comprehensive_model_comparison(
        train_df=train_df,
        test_df=test_df,
        include_heavy_models=True  # 모든 모델 포함
    )
    
    # 결과 출력 및 시각화
    print_detailed_comparison(results)
    visualize_comparison_results(results, save_path='full_model_comparison.png')
    
    return results


if __name__ == "__main__":
    # 빠른 비교 실행
    print("🚀 보험 추천 모델 종합 성능 비교")
    
    try:
        # 빠른 비교부터 시작
        quick_results = quick_comparison(sample_size=1000, test_size=200)
        
        # 결과 저장
        quick_results.to_csv('model_comparison_results.csv', index=False, encoding='utf-8-sig')
        logging.info("📁 결과 저장: model_comparison_results.csv")
        
        print("\n🎯 빠른 비교 완료!")
        print("전체 비교를 원한다면 full_comparison() 함수를 호출하세요.")
        
    except Exception as e:
        logging.error(f"❌ 모델 비교 실패: {e}")
        print(f"오류 발생: {e}")

    # 전체 비교 (모든 baseline 포함)
    results = full_comparison(sample_size=5000)
    
    # 결과 저장
    results.to_csv('full_model_comparison_results.csv', index=False, encoding='utf-8-sig')
    logging.info("📁 결과 저장: full_model_comparison_results.csv") 