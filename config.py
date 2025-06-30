"""
보험 추천 시스템 설정 파일
"""

import os

# 모델 경로 설정
MODEL_PATHS = {
    'pretrained_model': './MiniLM-L12-v2/0_Transformer',
    'trained_model': './trained_insurance_model_retoken'
}

# 데이터 경로 설정
DATA_PATHS = {
    'aggregated_data': "/workspace/recomAI/aggregated_df_250625_info.parquet"
}

# 학습 하이퍼파라미터
TRAINING_CONFIG = {
    'epochs': 4,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'max_length': 512,
    'test_size': 0.2,
    'eval_size': 0.1,
    'random_state': 42
}

# 보험료 구간 설정
PREMIUM_BINS = {
    'bins': [-float('inf'), 50000, 100000, 150000, 200000, float('inf')],
    'labels': ['5만원 이하', '5~10만원 이하', '10~15만원 이하', '15~20만원 이하', '20만원 초과']
}

# 테마 컬럼 설정
THEME_COLUMNS = [
    'cov_치매', 'cov_심장질환', 'cov_후유장해', 'cov_뇌혈관질환', 'cov_암', 'cov_사망',
    'cov_입원비(일당)', 'cov_운전자', 'cov_치아,화상,골절',
    'cov_수술비', 'cov_의료비', 'cov_법률,배상책임'
]

# 필요한 컬럼 설정
REQUIRED_COLUMNS = [
    'GNDR_CD', 'INS_AGE', 'JOB_GRD_CD', 'INJR_GRD', 'DRV_USG_DIV_CD', 'CHN_DIV',
    'SBCP_YYMM', 'UNT_PD_NM', 'SLZ_PREM', 'PY_INS_PRD_NAME', 'LWRT_TMN_RFD_TP_CD',
    'PY_EXEM_TP_CD', 'HNDY_ISP_TP_NM', 'PLAN_NM', 'PD_COV_NM', 'SBC_AMT'
] + THEME_COLUMNS

# 추천 시스템 설정
RECOMMENDATION_CONFIG = {
    'top_k': 5,
    'similarity_threshold': 0.5
}

# 어텐션 분석 설정
ATTENTION_CONFIG = {
    'max_tokens': 100,
    'top_k_tokens': 8,
    'default_layer': -1,
    'default_head': 0
} 