"""
demo_app.py — 중고차 가격 왜곡 탐지 웹 데모
실행: streamlit run demo_app.py
"""
import streamlit as st
import joblib
import json
import os
import sys

# detector.py를 같은 폴더에서 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import CarPriceDistortionDetector  # noqa: F401  (joblib 로드용)


# ===========================================================
# 페이지 설정
# ===========================================================
st.set_page_config(
    page_title="중고차 가격 왜곡 탐지",
    page_icon="🚗",
    layout="wide",
)


# ===========================================================
# 모델 로드 (캐싱)
# ===========================================================
@st.cache_resource
def load_model():
    here = os.path.dirname(os.path.abspath(__file__))
    det = joblib.load(os.path.join(here, 'detector.joblib'))
    with open(os.path.join(here, 'categories.json'), 'r', encoding='utf-8') as f:
        cats = json.load(f)
    return det, cats


det, cats = load_model()


# ===========================================================
# 헤더
# ===========================================================
st.title("🚗 중고차 가격 왜곡 탐지")
st.caption("매물 정보를 입력하면 AI가 적정가를 예측하고 사기·허위매물 위험도를 분석합니다.")


# ===========================================================
# 사이드바: 입력
# ===========================================================
with st.sidebar:
    st.header("📝 차량 정보 입력")

    # ── 필수 ──
    st.markdown("**필수 입력**")
    brand_options = cats['brands']
    default_idx = brand_options.index('현대') if '현대' in brand_options else 0
    brand = st.selectbox("브랜드", brand_options, index=default_idx)

    model_options = cats['brand_models'].get(brand, [])
    model = st.selectbox("모델명", model_options)

    year = st.number_input("연식 (예: 2020)", min_value=1990, max_value=2026,
                           value=2020, step=1)
    listed_price = st.number_input("매물 가격 (만원)", min_value=1,
                                   value=2000, step=10)

    # ── 선택 ──
    st.divider()
    st.markdown("**선택 입력** _(입력하면 정확도 ↑)_")

    mileage = st.number_input("주행거리 (km)", min_value=0, value=0, step=1000,
                              help="0으로 두면 평균값으로 자동 처리")
    fuel = st.selectbox("연료", ["(선택안함)"] + cats['fuels'])
    accident_cnt = st.number_input("사고 횟수", min_value=0, value=0, step=1)
    total_cost = st.number_input("총 사고수리비 (만원)", min_value=0,
                                 value=0, step=10)
    owner_cnt = st.number_input("소유자 변경 횟수", min_value=0, value=0, step=1)
    loan = st.checkbox("저당/압류 이력 있음")

    analyze = st.button("🔍 분석하기", type="primary", use_container_width=True)

    st.divider()
    with st.expander("ℹ️ 모델 정보"):
        st.caption(f"학습 데이터: **{det.train_rows:,}** 대")
        st.caption(f"Test R² (log): **{det.metrics['r2_log']:.3f}**")
        st.caption(f"Test MAPE: **{det.metrics['mape_pct']:.1f}%**")
        st.caption("XGBoost (튜닝) + SHAP Explainable AI")


# ===========================================================
# 초기 화면 (분석 전)
# ===========================================================
if not analyze:
    st.info("👈 왼쪽 사이드바에서 차량 정보를 입력하고 **[분석하기]** 버튼을 눌러주세요.")

    with st.expander("💡 이 시스템은 어떻게 작동하나요?", expanded=False):
        t = det.thresholds
        st.markdown(f"""
**1. 적정가 예측**
머신러닝 모델(XGBoost)이 학습 데이터를 기반으로 입력 차량의 **예상 적정가**를 예측합니다.

**2. 가격 편차율 계산**
입력하신 **매물 가격**과 **예측 적정가**의 차이를 비율로 계산합니다:
`편차율(%) = (매물가 − 예측가) / 예측가 × 100`

**3. 비대칭 임계값으로 분류**
학습 데이터의 잔차 분포 기반으로 결정된 5단계 분류:

| 분류 | 편차율 범위 | 의미 |
|---|---|---|
| 🚨 Danger (저가) | ≤ {t['DANGER_LOW']:.1f}% | 허위매물·사기 강한 의심 |
| ⚠️ Warning (저가) | {t['DANGER_LOW']:.1f}% ~ {t['WARNING_LOW']:.1f}% | 약간 저렴, 확인 권장 |
| ✅ Fair | {t['WARNING_LOW']:.1f}% ~ {t['WARNING_HIGH']:.1f}% | 정상 시세 |
| ⚠️ Warning (고가) | {t['WARNING_HIGH']:.1f}% ~ {t['DANGER_HIGH']:.1f}% | 약간 비쌈, 협상 여지 |
| 🚨 Danger (고가) | ≥ {t['DANGER_HIGH']:.1f}% | 과도하게 비쌈 |

**4. AI 설명 (SHAP)**
가격 결정에 영향을 미친 주요 요인 Top-5를 함께 제공합니다.
""")

    with st.expander("🧪 시연용 예시 케이스", expanded=False):
        st.markdown("""
- **정상 매물**: 현대 / 더 뉴 그랜저 IG / 2021 / 매물가 **2,400만원**
- **저가 의심**: 현대 / 팰리세이드 / 2020 / 매물가 **1,700만원**
- **사고차 + 저가**: 기아 / 더 뉴 K5 / 2018 / 사고 8회, 수리비 1,200만원 / 매물가 **950만원**
- **고가 의심**: 기아 / 카니발 4세대 / 2023 / 매물가 **5,500만원**
""")

    st.stop()


# ===========================================================
# 분석 실행
# ===========================================================
car_info = {
    'brand': brand,
    'model': model,
    'year': int(year),
    'mileage': float(mileage) if mileage > 0 else None,
    'fuel': fuel if fuel != "(선택안함)" else None,
    'accident_cnt': float(accident_cnt),
    'total_accident_cost': float(total_cost),
    'owner_change_cnt': float(owner_cnt),
    'loan': 1 if loan else 0,
}

with st.spinner("AI 분석 중..."):
    result = det.assess_listing(car_info, float(listed_price), explain=True)


# ===========================================================
# 결과 — 메트릭 카드
# ===========================================================
col1, col2, col3 = st.columns(3)
col1.metric("💰 매물 가격", f"{result['입력가격(만원)']:,.0f} 만원")
col2.metric("🎯 예상 적정가", f"{result['예상적정가(만원)']:,.0f} 만원")

dev = result['가격편차율(%)']
col3.metric(
    "📊 가격 편차율",
    f"{dev:+.1f}%",
    delta=f"{result['입력가격(만원)'] - result['예상적정가(만원)']:+,.0f} 만원",
    delta_color="inverse",  # 매물가가 적정가보다 낮으면 빨강(저가의심)
)


# ===========================================================
# 분류 박스
# ===========================================================
cat = result['분류']
risk = result['위험도(%)']

if 'Danger' in cat:
    st.error(f"## 🚨 {cat}  ·  위험도 {risk:.0f}%")
elif 'Warning' in cat:
    st.warning(f"## ⚠️ {cat}  ·  위험도 {risk:.0f}%")
else:
    st.success(f"## ✅ {cat}  ·  위험도 {risk:.0f}%")

# 위험도 게이지
st.progress(min(int(risk), 100) / 100, text=f"위험도: {risk:.1f}%")
st.caption("0~33%: Fair (정상) · 33~67%: Warning (주의) · 67~100%: Danger (위험)")


# ===========================================================
# 판정 메시지
# ===========================================================
st.markdown("### 📝 판정 메시지")
st.info(result['판정메시지'])


# ===========================================================
# 추가 검토사항
# ===========================================================
st.markdown("### 📋 추가 검토사항")
for i, item in enumerate(result['추가검토사항'], 1):
    if '[경고]' in item:
        st.error(f"**{i}.** {item.replace('[경고] ', '')}")
    elif 'Danger' in cat and i <= 2:
        st.warning(f"**{i}.** {item}")
    else:
        st.markdown(f"**{i}.** {item}")


# ===========================================================
# SHAP — 주요 영향 요인
# ===========================================================
st.markdown("### 🔍 AI 가격 결정 요인 (SHAP)")
st.caption("이 차량이 평균 대비 어떤 요인 때문에 가격이 결정됐는지 보여줍니다.")

cols = st.columns(min(5, len(result['주요_영향요인'])))
for col, factor in zip(cols, result['주요_영향요인']):
    direction_emoji = "🔼" if factor['direction'] == '↑' else "🔽"
    direction_text = "가격 ↑" if factor['direction'] == '↑' else "가격 ↓"
    feature_name = factor['feature']
    # 더미 변수 이름 정리
    if feature_name.startswith('brand_'):
        feature_name = f"브랜드: {feature_name[6:]}"
    elif feature_name.startswith('model_'):
        feature_name = f"모델: {feature_name[6:]}"
    elif feature_name.startswith('fuel_'):
        feature_name = f"연료: {feature_name[5:]}"
    elif feature_name == 'car_age_months':
        feature_name = "차령(개월)"
    elif feature_name == 'mileage':
        feature_name = "주행거리"
    elif feature_name == 'accident_cnt':
        feature_name = "사고 횟수"
    elif feature_name == 'owner_change_cnt':
        feature_name = "소유자 변경"
    elif feature_name == 'total_accident_cost_log':
        feature_name = "총 수리비"

    with col:
        st.metric(
            label=f"{direction_emoji} {feature_name}",
            value=f"{factor['price_effect_pct']:+.1f}%",
        )


# ===========================================================
# 푸터
# ===========================================================
st.divider()
st.caption(
    "🎓 한동대학교 ML 팀프로젝트 · 러닝(Running) 머신팀 · 2026  ·  "
    "본 결과는 통계적 추정이며 실제 매물 검토 시 직접 점검을 권장합니다."
)
