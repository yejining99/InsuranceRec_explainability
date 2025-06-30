"""
보험 추천 시스템 메인 실행 파일
전체 파이프라인: 데이터 전처리 → 모델 학습 → 성능 평가 → 어텐션 분석
"""

import os
import logging
from datetime import datetime
from typing import Tuple, Dict, List

from data_preprocessing import preprocess_insurance_data
from data_converter import InsuranceDataConverter
from model_trainer import InsuranceEmbeddingTrainer, split_data_by_date
from attention_analyzer import analyze_test_samples
from config import MODEL_PATHS, TRAINING_CONFIG


def main_training_pipeline(file_path: str = None, 
                          sample_size: int = None,
                          enable_attention_analysis: bool = True):
    """
    보험 추천 모델 전체 학습 파이프라인
    
    Args:
        file_path: 데이터 파일 경로 (None이면 config에서 가져옴)
        sample_size: 샘플 크기 (None이면 전체 데이터 사용)
        enable_attention_analysis: 어텐션 분석 수행 여부
    """
    
    logging.info("🎯 보험 추천 모델 학습 파이프라인 시작")
    start_time = datetime.now()
    
    try:
        # 1. 데이터 전처리 및 로딩
        logging.info("=" * 60)
        logging.info("🚀 1단계: 데이터 전처리 및 로딩")
        logging.info("=" * 60)
        
        fin_aggregated_df = preprocess_insurance_data(file_path)
        
        # 샘플 크기 제한 (테스트용)
        if sample_size:
            fin_aggregated_df = fin_aggregated_df.head(sample_size)
            logging.info(f"📊 샘플 크기 제한: {sample_size}개")
        
        # 2. 데이터 변환 (Query-Value pair)
        logging.info("=" * 60)
        logging.info("🔄 2단계: Query-Value pair 변환")
        logging.info("=" * 60)
        
        converter = InsuranceDataConverter()
        result_df = converter.convert_dataframe(fin_aggregated_df)
        
        # 3. 데이터 분할
        logging.info("=" * 60)
        logging.info("📊 3단계: 데이터 분할")
        logging.info("=" * 60)
        
        train_df, eval_df, test_df = split_data_by_date(result_df)
        
        # 4. 모델 학습
        logging.info("=" * 60)
        logging.info("🚀 4단계: 모델 학습")
        logging.info("=" * 60)
        
        trainer = InsuranceEmbeddingTrainer()
        
        # 사전 학습된 모델 로드 (선택적)
        try:
            trainer.load_pretrained_model()
            
            # 사전 학습된 모델로 샘플 임베딩 테스트
            sample_sentences = train_df['query'].head(3).tolist()
            embeddings = trainer.get_embeddings_manual(sample_sentences)
            logging.info(f"✅ 사전 학습된 모델 임베딩 테스트: {embeddings.shape}")
            
        except Exception as e:
            logging.warning(f"⚠️ 사전 학습된 모델 로드 실패: {e}")
            logging.info("🌐 온라인 모델 사용으로 전환")
        
        # 모델 학습 실행
        trainer.train_model_with_safe_evaluator(
            train_df=train_df,
            eval_df=eval_df,
            epochs=TRAINING_CONFIG['epochs'],
            batch_size=TRAINING_CONFIG['batch_size'],
            learning_rate=TRAINING_CONFIG['learning_rate']
        )
        
        # 5. 모델 성능 평가
        logging.info("=" * 60)
        logging.info("📊 5단계: 모델 성능 평가")
        logging.info("=" * 60)
        
        metrics = trainer.evaluate_model_performance(test_df)
        
        print("\n" + "=" * 60)
        print("📊 모델 성능 평가 결과")
        print("=" * 60)
        for metric, value in metrics.items():
            print(f"{metric:20}: {value:.4f}")
        print("=" * 60)
        
        # 6. 추천 시스템 테스트
        logging.info("🎯 6단계: 추천 시스템 테스트")
        
        test_query = test_df['query'].iloc[0]
        test_values = test_df['value'].head(10).tolist()
        
        recommendations = trainer.recommend_products(
            user_query=test_query,
            product_values=test_values,
            top_k=5
        )
        
        print("\n" + "=" * 60)
        print("🎯 추천 시스템 테스트 결과")
        print("=" * 60)
        print(f"👤 사용자 쿼리: {test_query[:100]}...")
        print("\n📋 추천 상품:")
        for i, (idx, score) in enumerate(recommendations):
            print(f"{i+1}. 유사도: {score:.4f}")
            print(f"   상품: {test_values[idx][:100]}...")
            print()
        
        # 7. 어텐션 분석 (선택적)
        if enable_attention_analysis:
            logging.info("=" * 60)
            logging.info("🔍 7단계: 어텐션 분석")
            logging.info("=" * 60)
            
            try:
                analyzer, attention_results = analyze_test_samples(
                    test_df=test_df.head(20),  # 상위 20개 샘플만
                    model_path=MODEL_PATHS['trained_model'],
                    sample_size=3
                )
                logging.info("✅ 어텐션 분석 완료")
            except Exception as e:
                logging.warning(f"⚠️ 어텐션 분석 실패: {e}")
                attention_results = None
        else:
            attention_results = None
        
        # 8. 결과 저장 확인
        if os.path.exists(trainer.output_path):
            logging.info(f"✅ 모델이 성공적으로 저장되었습니다: {trainer.output_path}")
        else:
            logging.error(f"❌ 모델 저장 실패: {trainer.output_path}")
        
        # 파이프라인 완료
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        
        print("\n" + "=" * 60)
        print("🎉 파이프라인 완료")
        print("=" * 60)
        print(f"⏱️  총 소요 시간: {elapsed_time}")
        print(f"📊 처리된 데이터: {len(result_df)}개 레코드")
        print(f"🏆 최종 성능 (F1 Score): {metrics.get('f1_score', 0):.4f}")
        print(f"💾 모델 저장 위치: {trainer.output_path}")
        print("=" * 60)
        
        return {
            'trainer': trainer,
            'metrics': metrics,
            'test_df': test_df,
            'attention_results': attention_results,
            'elapsed_time': elapsed_time
        }
    
    except Exception as e:
        logging.error(f"❌ 파이프라인 실행 중 오류 발생: {e}")
        raise


def quick_test_pipeline(sample_size: int = 100):
    """빠른 테스트를 위한 간소화된 파이프라인"""
    logging.info("🧪 빠른 테스트 파이프라인 시작")
    
    return main_training_pipeline(
        sample_size=sample_size,
        enable_attention_analysis=False
    )


def load_and_test_trained_model(model_path: str = None, test_queries: List[str] = None):
    """학습된 모델을 로드하여 테스트"""
    logging.info("🔍 학습된 모델 테스트")
    
    if model_path is None:
        model_path = MODEL_PATHS['trained_model']
    
    # 모델 로드
    trainer = InsuranceEmbeddingTrainer()
    try:
        trainer.load_trained_model(model_path)
        logging.info(f"✅ 모델 로드 성공: {model_path}")
    except Exception as e:
        logging.error(f"❌ 모델 로드 실패: {e}")
        return None
    
    # 테스트 데이터 생성
    if test_queries is None:
        test_queries = [
            "35세 남성 고객으로 직업등급 1급 사무직, 상해등급 1급 매우낮음에 해당합니다. 희망하는 보험료는 5~10만원 이하이고 암, 사망 테마에 특별한 관심이 있습니다.",
            "42세 여성 고객으로 직업등급 2급 일반직, 상해등급 3급 낮음에 해당합니다. 희망하는 보험료는 10~15만원 이하이고 치매, 뇌혈관질환 테마에 특별한 관심이 있습니다."
        ]
    
    # 샘플 상품 정보 (실제 운영시에는 DB에서 가져옴)
    sample_products = [
        "무배당 종합보험 상품으로 월 보험료 8만 5천원입니다. 납입조건은 20년납이며 표준형 방식을 적용합니다.",
        "건강보험 플러스 상품으로 월 보험료 12만원입니다. 납입조건은 15년납이며 해지환급금미지급형 방식을 적용합니다.",
        "암보험 프리미엄 상품으로 월 보험료 6만원입니다. 납입조건은 전기납이며 표준형 방식을 적용합니다."
    ]
    
    # 추천 테스트
    print("\n" + "=" * 60)
    print("🎯 학습된 모델 추천 테스트")
    print("=" * 60)
    
    for i, query in enumerate(test_queries):
        print(f"\n🔍 테스트 {i+1}")
        print(f"👤 고객 프로필: {query}")
        print("\n📋 추천 결과:")
        
        recommendations = trainer.recommend_products(
            user_query=query,
            product_values=sample_products,
            top_k=3
        )
        
        for j, (idx, score) in enumerate(recommendations):
            print(f"  {j+1}. 유사도: {score:.4f} - {sample_products[idx]}")
        print()
    
    return trainer


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    
    print("🚀 보험 추천 시스템 메인 파이프라인")
    print("=" * 60)
    print("선택하세요:")
    print("1. 전체 파이프라인 실행 (데이터 전처리 → 학습 → 평가 → 어텐션 분석)")
    print("2. 빠른 테스트 (소량 데이터로 빠른 테스트)")
    print("3. 학습된 모델 테스트 (기존 모델 로드하여 추천 테스트)")
    print("=" * 60)
    
    choice = input("선택 (1/2/3): ").strip()
    
    try:
        if choice == "1":
            print("🎯 전체 파이프라인을 시작합니다...")
            results = main_training_pipeline()
            
        elif choice == "2":
            print("🧪 빠른 테스트를 시작합니다...")
            results = quick_test_pipeline(sample_size=200)
            
        elif choice == "3":
            print("🔍 학습된 모델 테스트를 시작합니다...")
            trainer = load_and_test_trained_model()
            
        else:
            print("❌ 잘못된 선택입니다. 1, 2, 3 중 하나를 선택해주세요.")
            
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        logging.error(f"Main pipeline error: {e}", exc_info=True) 