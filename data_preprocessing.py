"""
데이터 전처리 모듈
보험 데이터 로딩 및 기본 전처리 기능
"""

import pandas as pd
import numpy as np
from config import DATA_PATHS, PREMIUM_BINS, THEME_COLUMNS, REQUIRED_COLUMNS


def load_aggregated_data(file_path: str = None) -> pd.DataFrame:
    """
    집계된 보험 데이터를 로드합니다.
    
    Args:
        file_path: 데이터 파일 경로 (기본값: config에서 가져옴)
        
    Returns:
        로드된 데이터프레임
    """
    if file_path is None:
        file_path = DATA_PATHS['aggregated_data']
    
    print(f"📂 데이터 로딩 중: {file_path}")
    
    try:
        aggregated_df = pd.read_parquet(file_path)
        print(f"✅ 데이터 로딩 완료: {aggregated_df.shape}")
        return aggregated_df
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        raise


def select_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    필요한 컬럼만 선택합니다.
    
    Args:
        df: 원본 데이터프레임
        
    Returns:
        필요한 컬럼만 포함된 데이터프레임
    """
    print(f"📋 필요한 컬럼 선택 중... (총 {len(REQUIRED_COLUMNS)}개)")
    
    # 존재하는 컬럼만 선택
    available_columns = [col for col in REQUIRED_COLUMNS if col in df.columns]
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    
    if missing_columns:
        print(f"⚠️ 누락된 컬럼: {missing_columns}")
    
    fin_aggregated_df = df[available_columns].copy()
    print(f"✅ 컬럼 선택 완료: {fin_aggregated_df.shape}")
    
    return fin_aggregated_df


def create_premium_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    보험료 구간을 생성합니다.
    
    Args:
        df: 데이터프레임
        
    Returns:
        보험료 구간이 추가된 데이터프레임
    """
    print("💰 보험료 구간 생성 중...")
    
    df = df.copy()
    df['tar_prem'] = pd.cut(
        df['SLZ_PREM'], 
        bins=PREMIUM_BINS['bins'], 
        labels=PREMIUM_BINS['labels']
    )
    
    print(f"✅ 보험료 구간 생성 완료")
    print("📊 보험료 구간별 분포:")
    print(df['tar_prem'].value_counts())
    
    return df


def extract_themes(row: pd.Series) -> str:
    """
    테마 컬럼에서 활성화된 테마를 추출합니다.
    
    Args:
        row: 데이터프레임의 행
        
    Returns:
        활성화된 테마들을 콤마로 구분한 문자열
    """
    themes = [col.replace('cov_', '') for col in THEME_COLUMNS if row[col] >= 1]
    return ', '.join(themes) if themes else '없음'


def create_target_theme(df: pd.DataFrame) -> pd.DataFrame:
    """
    타겟 테마 컬럼을 생성합니다.
    
    Args:
        df: 데이터프레임
        
    Returns:
        타겟 테마가 추가된 데이터프레임
    """
    print("🎯 타겟 테마 생성 중...")
    
    df = df.copy()
    df['tar_theme'] = df.apply(extract_themes, axis=1)
    
    # 테마 컬럼 제거
    df = df.drop(columns=THEME_COLUMNS)
    
    print(f"✅ 타겟 테마 생성 완료")
    print("📊 테마별 분포:")
    theme_counts = df['tar_theme'].value_counts().head(10)
    print(theme_counts)
    
    return df


def preprocess_insurance_data(file_path: str = None) -> pd.DataFrame:
    """
    보험 데이터 전체 전처리 파이프라인
    
    Args:
        file_path: 데이터 파일 경로
        
    Returns:
        전처리된 데이터프레임
    """
    print("🚀 보험 데이터 전처리 시작")
    print("=" * 50)
    
    # 1. 데이터 로딩
    df = load_aggregated_data(file_path)
    
    # 2. 필요한 컬럼 선택
    df = select_required_columns(df)
    
    # 3. 보험료 구간 생성
    df = create_premium_bins(df)
    
    # 4. 타겟 테마 생성
    df = create_target_theme(df)
    
    print("=" * 50)
    print(f"✅ 전처리 완료: {df.shape}")
    print(f"📋 최종 컬럼: {list(df.columns)}")
    
    return df


if __name__ == "__main__":
    # 테스트 실행
    fin_aggregated_df = preprocess_insurance_data()
    print("\n📊 샘플 데이터:")
    print(fin_aggregated_df.head()) 