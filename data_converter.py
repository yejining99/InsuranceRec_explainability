"""
보험 데이터 변환 모듈
InsuranceDataConverter 클래스 - 보험 데이터를 Query-Value pair로 변환
"""

import pandas as pd
import re
from typing import Dict


class InsuranceDataConverter:
    """보험 데이터를 Query-Value pair로 변환하는 클래스"""
    
    def __init__(self):
        # 코드 매핑 사전들
        self.gender_map = {
            '1': '남성',
            '2': '여성'
        }
        
        self.job_grade_map = {
            '1': '1급 사무직',
            '2': '2급 일반직',
            '3': '3급 위험직',
        }
        
        self.injury_grade_map = {
            '1': '1급 매우낮음',
            '2': '2급 낮음', 
            '3': '3급 낮음',
            '4': '4급 보통',
            '5': '5급 높음',
            '6': '6급 높음',
            '7': '7급 매우높음',
            '8': '8급 매우높음',
            '9': '9급 고위험',
            '10': '10급 고위험'
        }
        
        self.payment_exemption_map = {
            '00': '납입면제 미적용형',
            '01': '납입면제1형',
            '02': '납입면제2형',
            '03': '납입면제3형',
            '04': '납입면제4형',
            '05': '납입면제5형',
            '06': '납입면제6형'
        }
        
        self.surrender_refund_map = {
            '00': '표준형',
            '01': '해지환급금미지급형',
            '02': '해지환급금미지급형',
            '03': '해지환급금미지급형',
            '04': '해지환급금50%지급형',
            '05': '해지환급금미지급형',
            '06': '해지환급금미지급형',
            '07': '해지환급금지급형'
        }
        
        self.health_declaration_map = {
            '01': '일반고지형',
            '02': '건강고지형 6년',
            '03': '건강고지형 7년',
            '04': '건강고지형 8년',
            '05': '건강고지형 9년',
            '06': '건강고지형 10년'
        }

    def decode_value(self, value, mapping_dict: Dict) -> str:
        """코드값을 의미있는 텍스트로 변환"""
        if pd.isna(value):
            return "정보없음"
        return mapping_dict.get(str(value), str(value))

    def format_premium(self, premium) -> str:
        """보험료를 읽기 쉬운 형태로 포맷"""
        if pd.isna(premium):
            return "보험료 정보없음"
        
        premium = int(premium)
        return self.format_amount_korean(premium)
    
    def format_amount_korean(self, amount: int) -> str:
        """금액을 한국어 단위로 포맷 (예: 50000 -> 5만원)"""
        if amount == 0:
            return "0원"
        
        # 억 단위
        if amount >= 100000000:
            uk = amount // 100000000
            remainder = amount % 100000000
            if remainder == 0:
                return f"{uk}억원"
            elif remainder >= 10000000:  # 천만 단위
                man = remainder // 10000000
                return f"{uk}억 {man}천만원"
            elif remainder >= 10000:  # 만 단위
                man = remainder // 10000
                return f"{uk}억 {man}만원"
            else:
                return f"{uk}억 {remainder}원"
        
        # 만 단위 
        elif amount >= 10000:
            man = amount // 10000
            remainder = amount % 10000
            if remainder == 0:
                return f"{man}만원"
            else:
                return f"{man}만 {remainder}원"
        
        # 만 미만
        else:
            return f"{amount}원"

    def preprocess_product_name(self, name: str) -> str:
        """
        보험 상품명 전처리 함수

        Args:
            name: 원본 상품명

        Returns:
            전처리된 상품명
        """
        # 1. '(무)' 제거
        name = re.sub(r'\(무\)', '', name)

        # 2. '메리츠' 제거
        name = re.sub(r'\b메리츠\b', '', name)

        # 3. '2504' 등 숫자 연도 제거 (2504, 2505 등)
        name = re.sub(r'\d{4}', '', name)

        # 4. 괄호 내 텍스트 제거 (단, 갱신형/세만기형은 남김)
        # 예: (통합간편심사형) → 제거
        name = re.sub(r'\((?!.*갱신형|세만기형).*?\)', '', name)

        # 5. 괄호 자체 제거 (남은 경우)
        name = re.sub(r'[()]', '', name)

        # 6. 공백 정리
        name = re.sub(r'\s+', ' ', name).strip()

        return name

    def format_coverage_and_amounts(self, coverage_str, amount_str) -> str:
        """담보명과 가입금액을 매칭하여 포맷"""
        if pd.isna(coverage_str) or pd.isna(amount_str):
            return "담보 정보없음"
        
        # 담보명 파싱 (! 구분)
        coverages = coverage_str.split('!')
        coverages = [cov.strip() for cov in coverages if cov.strip()]
        
        # 가입금액 파싱 (, 구분)
        amounts = str(amount_str).split(',')
        amounts = [amt.strip() for amt in amounts if amt.strip()]
        
        # 담보와 금액 매칭
        coverage_list = []
        for i, coverage in enumerate(coverages):
            if i < len(amounts):
                amount = amounts[i]
                # 금액 포맷팅 (한국어 단위로)
                try:
                    amount_int = int(amount)
                    formatted_amount = self.format_amount_korean(amount_int)
                except:
                    formatted_amount = amount
                
                # 담보명 정리 (불필요한 기호 제거)
                clean_coverage = re.sub(r'[갱신형|!\[\]()]', '', coverage).strip()
                coverage_list.append(f"{clean_coverage} {formatted_amount}")
            else:
                clean_coverage = re.sub(r'[갱신형|!\[\]()]', '', coverage).strip()
                coverage_list.append(clean_coverage)
        
        return ", ".join(coverage_list[:3]) + ("..." if len(coverage_list) > 3 else "")

    def format_target_theme(self, theme_str) -> str:
        """타겟 테마를 읽기 쉽게 포맷"""
        if pd.isna(theme_str):
            return "특별한 관심사항 없음"
        
        themes = str(theme_str).split(',')
        themes = [theme.strip() for theme in themes if theme.strip()]
        
        theme_map = {
            '사망': '사망',
            '암': '암',
            '치매': '치매', 
            '뇌질환': '뇌혈관',
            '심장질환': '심장질환',
            '수술비': '수술비',
            '간병': '간병',
            '치아': '치아',
            '화상': '화상',
            '골절': '골절'
        }
        
        formatted_themes = []
        for theme in themes[:4]:  # 최대 4개만
            formatted_themes.append(theme_map.get(theme, theme))
        
        return ", ".join(formatted_themes)

    def convert_to_query_value_pair(self, row: pd.Series) -> Dict:
        """단일 행을 Query-Value pair로 변환"""
        
        # === QUERY: 사용자 정보 (고객 프로필) ===
        gender = self.decode_value(row['GNDR_CD'], self.gender_map)
        age = f"{row['INS_AGE']}세" if not pd.isna(row['INS_AGE']) else "연령 정보없음"
        job_grade = self.decode_value(row['JOB_GRD_CD'], self.job_grade_map)
        injury_grade = self.decode_value(row['INJR_GRD'], self.injury_grade_map)
        target_premium = row.get('tar_prem', '희망보험료 정보없음')
        target_theme = self.format_target_theme(row.get('tar_theme', ''))
        
        query = (
            f"{age} {gender} 고객으로 직업등급 {job_grade}, 상해등급 {injury_grade}에 해당합니다. "
            f"희망하는 보험료는 {target_premium}이고 {target_theme} 테마에 특별한 관심이 있습니다."
        )
        
        # === VALUE: 보험상품 정보 (증권 정보) ===
        product_name = row.get('UNT_PD_NM', '상품명 정보없음')
        product_name = self.preprocess_product_name(product_name)
        premium = self.format_premium(row.get('SLZ_PREM'))
        payment_period = row.get('PY_INS_PRD_NAME', '납입기간 정보없음')
        surrender_type = self.decode_value(row.get('LWRT_TMN_RFD_TP_CD'), self.surrender_refund_map)
        payment_exemption = self.decode_value(row.get('PY_EXEM_TP_CD'), self.payment_exemption_map)
        simple_review = row.get('HNDY_ISP_TP_NM', '심사유형 정보없음')
        plan_name = row.get('PLAN_NM', '')
        
        # 담보 정보
        coverage_info = self.format_coverage_and_amounts(
            row.get('PD_COV_NM'), 
            row.get('SBC_AMT')
        )
        
        value = (
            f"{product_name} 상품으로 월 보험료 {premium}입니다. "
            f"납입조건은 {payment_period}이며 {surrender_type} 방식을 적용합니다. "
            f"{payment_exemption} 조건이 포함되고 {simple_review}으로 간편하게 가입 가능합니다."
        )
        
        if plan_name and str(plan_name) != 'None' and str(plan_name).strip():
            value += f" {plan_name} 플랜이 적용됩니다."
        
        value += f" 주요 보장내용: {coverage_info}"
        
        return {
            'date': row.get('SBCP_YYMM'),
            'query': query.strip(),
            'value': value.strip(),
            'label': 1
        }

    def convert_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """전체 데이터프레임을 Query-Value pair로 변환"""
        print("🔄 데이터프레임을 Query-Value pair로 변환 중...")
        
        results = []
        
        for idx, row in df.iterrows():
            converted = self.convert_to_query_value_pair(row)
            converted['original_index'] = idx
            results.append(converted)
        
        result_df = pd.DataFrame(results)
        
        print(f"✅ 변환 완료: {len(result_df)}개 레코드")
        print("📊 샘플 Query:", result_df['query'].iloc[0][:100] + "...")
        print("📊 샘플 Value:", result_df['value'].iloc[0][:100] + "...")
        
        return result_df


if __name__ == "__main__":
    # 테스트 실행
    from data_preprocessing import preprocess_insurance_data
    
    # 샘플 데이터로 테스트
    print("🧪 InsuranceDataConverter 테스트 시작")
    
    # 데이터 로드 및 전처리
    df = preprocess_insurance_data()
    sample_df = df.head(100)  # 샘플 100개만
    
    # 변환기 초기화 및 실행
    converter = InsuranceDataConverter()
    result_df = converter.convert_dataframe(sample_df)
    
    print("\n📋 변환 결과:")
    print(f"원본 데이터: {sample_df.shape}")
    print(f"변환 데이터: {result_df.shape}")
    print(f"컬럼: {list(result_df.columns)}")
    
    print(f"\n📝 Query 샘플:\n{result_df['query'].iloc[0]}")
    print(f"\n📝 Value 샘플:\n{result_df['value'].iloc[0]}") 