"""
detector.py — 중고차 가격 왜곡 탐지 시스템
CarPriceDistortionDetector:
  - fit(data_path): 데이터 로드 → 전처리 → 학습 → 잔차 분석 → 임계값 결정
  - assess_listing(car_info, listed_price): 예측 + 분류 + 위험도 + 메시지 + 검토사항
  - explain(car_info): SHAP 기반 피처 영향도 설명
"""
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')


class CarPriceDistortionDetector:
    """중고차 매물 가격 왜곡 탐지기"""

    # 사용자 입력에 1:1 매칭되는 피처
    CAT_FEATURES = ['brand', 'model', 'fuel']
    NUM_FEATURES = ['car_age_months', 'mileage', 'accident_cnt',
                    'total_accident_cost_log', 'owner_change_cnt', 'loan']
    RARE_FUEL = ['수소', 'LPG+전기', '가솔린+LPG']
    MIN_MODEL_FREQ = 30

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.pipeline = None         # sklearn Pipeline (prep + XGB)
        self.thresholds = None       # 분류 임계값
        self.metrics = None          # 평가 지표
        self.keep_models = None      # 학습 시 유지된 model 카테고리
        self.ref_year = None         # 기준 시점(연)
        self.ref_month = None        # 기준 시점(월)
        self.train_rows = 0
        self.shap_explainer = None
        self.feature_names_after_prep = None
        self.train_defaults = {}     # 누락 입력 채우기용

    # =========================================================
    # 1) fit
    # =========================================================
    def fit(self, data_path: str):
        # ------ 로드 + 정제 ------
        df = pd.read_excel(data_path)
        df = df.drop_duplicates().reset_index(drop=True)
        df = df.drop(columns=['id', 'vehicleNo'], errors='ignore')
        df = df[df['price'] > 50].reset_index(drop=True)
        df = df[df['brand'] != '기타 제조사'].reset_index(drop=True)
        df['fuel'] = df['fuel'].replace(self.RARE_FUEL, '기타연료')

        # 빈도 낮은 model 통합
        mf = df['model'].value_counts()
        self.keep_models = mf[mf >= self.MIN_MODEL_FREQ].index.tolist()
        df['model'] = df['model'].where(df['model'].isin(self.keep_models), 'model_other')

        # year(YYYYMM) → car_age_months
        df['reg_year'] = df['year'] // 100
        df['reg_month'] = df['year'] % 100
        mxy = df['year'].max()
        self.ref_year = int(mxy // 100)
        self.ref_month = int(mxy % 100) + 1
        if self.ref_month > 12:
            self.ref_year += 1
            self.ref_month = 1
        df['car_age_months'] = (
            (self.ref_year - df['reg_year']) * 12 + (self.ref_month - df['reg_month'])
        ).clip(lower=0)

        df['total_accident_cost_log'] = np.log1p(df['total_accident_cost'])

        # ------ X, y ------
        X = df[self.CAT_FEATURES + self.NUM_FEATURES]
        y_log = np.log1p(df['price'])
        y_raw = df['price']

        # ------ Train/Test split ------
        X_tr, X_te, y_tr, y_te, ytr_raw, yte_raw = train_test_split(
            X, y_log, y_raw, test_size=0.2, random_state=self.random_state)

        # ------ 누락 입력 기본값 (Train 통계 기반) ------
        self.train_defaults = {
            'mileage': float(X_tr['mileage'].median()),
            'fuel': X_tr['fuel'].mode()[0],
            'accident_cnt': 0.0,                 # 보수적 가정
            'total_accident_cost_log': 0.0,      # 보수적 가정
            'owner_change_cnt': float(X_tr['owner_change_cnt'].median()),
            'loan': 0.0,                         # 보수적 가정
        }

        # ------ Pipeline (튜닝된 XGBoost) ------
        prep = ColumnTransformer([
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False),
             self.CAT_FEATURES),
            ('num', 'passthrough', self.NUM_FEATURES),
        ])
        reg = xgb.XGBRegressor(
            n_estimators=1200, max_depth=4, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9, min_child_weight=1,
            random_state=self.random_state, n_jobs=-1, verbosity=0)
        self.pipeline = Pipeline([('prep', prep), ('reg', reg)])
        self.pipeline.fit(X_tr, y_tr)
        self.train_rows = len(X_tr)

        # ------ 평가 ------
        pred_te_log = self.pipeline.predict(X_te)
        pred_te_raw = np.clip(np.expm1(pred_te_log), 1, None)
        self.metrics = {
            'r2': float(r2_score(yte_raw, pred_te_raw)),
            'r2_log': float(r2_score(y_te, pred_te_log)),
            'rmse_manwon': float(np.sqrt(mean_squared_error(yte_raw, pred_te_raw))),
            'mae_manwon': float(mean_absolute_error(yte_raw, pred_te_raw)),
            'mape_pct': float(np.mean(np.abs((yte_raw - pred_te_raw) / yte_raw)) * 100),
        }

        # ------ 잔차 분석 + 임계값 결정 ------
        pred_tr_log = self.pipeline.predict(X_tr)
        pred_tr_raw = np.clip(np.expm1(pred_tr_log), 1, None)
        deviation_tr = (ytr_raw.values - pred_tr_raw) / pred_tr_raw * 100

        # 학습 잔차 percentile + 도메인 cap (MAPE × 1.5)
        domain_cap = self.metrics['mape_pct'] * 1.5
        self.thresholds = {
            'DANGER_LOW':   max(float(np.percentile(deviation_tr, 5)),  -domain_cap),
            'WARNING_LOW':  max(float(np.percentile(deviation_tr, 15)), -domain_cap * 0.5),
            'WARNING_HIGH': min(float(np.percentile(deviation_tr, 85)),  domain_cap * 0.5),
            'DANGER_HIGH':  min(float(np.percentile(deviation_tr, 95)),  domain_cap),
        }

        # ------ SHAP explainer ------
        X_tr_transformed = self.pipeline.named_steps['prep'].transform(X_tr)
        self.feature_names_after_prep = self._get_feature_names()
        self.shap_explainer = shap.TreeExplainer(self.pipeline.named_steps['reg'])
        return self

    # =========================================================
    # 2) assess_listing
    # =========================================================
    def assess_listing(self, car_info: dict, listed_price: float,
                       explain: bool = False) -> dict:
        # 입력 정규화 (필수 + 누락 채우기)
        X, missing_fields = self._prepare_input(car_info)

        # 예측
        pred_log = self.pipeline.predict(X)[0]
        predicted = float(np.expm1(pred_log))

        # 편차율 (저가=음수, 고가=양수)
        deviation = (listed_price - predicted) / predicted * 100

        # 분류 + 위험도
        category, risk = self._classify(deviation)

        # 메시지
        message = self._generate_message(category, deviation)

        # 추가 검토사항
        suggestions = self._generate_suggestions(car_info, category, missing_fields)

        result = {
            '입력가격(만원)': float(listed_price),
            '예상적정가(만원)': round(predicted, 1),
            '가격편차율(%)': round(deviation, 2),
            '분류': category,
            '위험도(%)': round(risk, 1),
            '판정메시지': message,
            '추가검토사항': suggestions,
        }

        if explain:
            result['주요_영향요인'] = self._explain_prediction(X)

        return result

    # =========================================================
    # 3) 내부: 입력 정규화
    # =========================================================
    def _prepare_input(self, car_info: dict) -> tuple:
        info = dict(car_info)  # 복사

        # 필수
        brand = info.get('brand')
        model = info.get('model')
        year = info.get('year')
        if brand is None or model is None or year is None:
            raise ValueError("brand, model, year는 필수 입력입니다.")

        # car_age_months 계산
        # year는 YYYY (예: 2020) 또는 YYYYMM (예: 202007) 모두 허용
        if year >= 1000 and year < 9999:
            reg_year, reg_month = int(year), 6  # 연도만 입력 시 6월로 가정
        else:
            reg_year, reg_month = int(year) // 100, int(year) % 100
        car_age_months = max(
            (self.ref_year - reg_year) * 12 + (self.ref_month - reg_month), 0
        )

        # model 정규화 (학습 카테고리에 없으면 model_other)
        if model not in self.keep_models:
            model = 'model_other'

        # fuel 정규화 (희소 카테고리 통합)
        fuel = info.get('fuel')
        if fuel in self.RARE_FUEL:
            fuel = '기타연료'

        # 누락 체크
        missing = []
        for key in ['mileage', 'fuel', 'accident_cnt',
                    'total_accident_cost', 'owner_change_cnt', 'loan']:
            v = info.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                missing.append(key)

        # 누락 채우기
        if info.get('mileage') is None:
            mileage = self.train_defaults['mileage']
        else:
            mileage = float(info['mileage'])

        if fuel is None:
            fuel = self.train_defaults['fuel']

        accident_cnt = float(info.get('accident_cnt') or 0.0)
        total_accident_cost = float(info.get('total_accident_cost') or 0.0)
        total_accident_cost_log = np.log1p(total_accident_cost)
        owner_change_cnt = (float(info['owner_change_cnt'])
                            if info.get('owner_change_cnt') is not None
                            else self.train_defaults['owner_change_cnt'])
        loan = float(info.get('loan') or 0.0)

        X = pd.DataFrame([{
            'brand': brand,
            'model': model,
            'fuel': fuel,
            'car_age_months': car_age_months,
            'mileage': mileage,
            'accident_cnt': accident_cnt,
            'total_accident_cost_log': total_accident_cost_log,
            'owner_change_cnt': owner_change_cnt,
            'loan': loan,
        }])
        return X, missing

    # =========================================================
    # 4) 내부: 분류 + 위험도 계산
    #    위험도(%) = 임계값 대비 비율 (사용자 결정 B)
    # =========================================================
    def _classify(self, dev: float) -> tuple:
        t = self.thresholds
        DL, WL, WH, DH = t['DANGER_LOW'], t['WARNING_LOW'], t['WARNING_HIGH'], t['DANGER_HIGH']

        if dev < 0:  # 저가 측 (허위매물 의심)
            if dev >= WL:
                # Fair 구간: 0% ~ 33%
                risk = abs(dev) / abs(WL) * 33 if WL != 0 else 0
                category = 'Fair'
            elif dev >= DL:
                # Warning(저가): 33% ~ 67%
                risk = 33 + (WL - dev) / (WL - DL) * 34
                category = 'Warning(저가)'
            else:
                # Danger(저가): 67% ~ 100%
                risk = 67 + (DL - dev) / max(abs(DL), 1) * 33
                category = 'Danger(저가)'
        else:  # 고가 측 (과대광고 의심)
            if dev <= WH:
                risk = dev / WH * 33 if WH != 0 else 0
                category = 'Fair'
            elif dev <= DH:
                risk = 33 + (dev - WH) / (DH - WH) * 34
                category = 'Warning(고가)'
            else:
                risk = 67 + (dev - DH) / max(DH, 1) * 33
                category = 'Danger(고가)'

        risk = min(max(risk, 0), 100)
        return category, risk

    # =========================================================
    # 5) 내부: 판정 메시지
    # =========================================================
    def _generate_message(self, category: str, deviation: float) -> str:
        msgs = {
            'Fair': f"적정 시세 범위입니다 (편차 {deviation:+.1f}%).",
            'Warning(저가)': f"시세보다 {abs(deviation):.1f}% 저렴합니다. "
                            f"옵션/트림/관리상태 차이일 수 있으나 추가 확인을 권장합니다.",
            'Danger(저가)': f"시세보다 {abs(deviation):.1f}% 비정상적으로 저렴합니다. "
                           f"허위매물·사기·중대 결함 가능성이 있습니다. "
                           f"차량 실물 점검과 등록증 확인이 필수입니다.",
            'Warning(고가)': f"시세보다 {deviation:.1f}% 비쌉니다. "
                           f"풀옵션/특수 사양일 수 있으나 가격 협상 여지가 있습니다.",
            'Danger(고가)': f"시세보다 {deviation:.1f}% 과도하게 비쌉니다. "
                           f"동급 매물과 비교 검토를 강력 권장합니다.",
        }
        return msgs.get(category, "")

    # =========================================================
    # 6) 내부: 추가 검토사항 (룰 기반)
    # =========================================================
    def _generate_suggestions(self, car_info: dict, category: str,
                              missing_fields: list) -> list:
        sug = []

        # 입력 정보 기반 검토
        ac = car_info.get('accident_cnt')
        if ac is not None and ac >= 5:
            sug.append(f"사고 이력이 {int(ac)}회로 많습니다. 사고 부위 직접 확인을 권장합니다.")

        cost = car_info.get('total_accident_cost')
        if cost is not None and cost >= 1000:
            sug.append(f"누적 수리비가 {cost:.0f}만원입니다. 중대사고 여부와 수리부위를 점검하세요.")

        own = car_info.get('owner_change_cnt')
        if own is not None and own >= 3:
            sug.append(f"소유자 변경이 {int(own)}회로 잦습니다. 사용 이력(영업용 가능성)을 확인하세요.")

        ml = car_info.get('mileage')
        if ml is not None and ml >= 200000:
            sug.append(f"주행거리가 {ml:,.0f}km로 매우 높습니다. 엔진·미션·하체 정밀 점검이 필요합니다.")

        if car_info.get('loan') == 1:
            sug.append("저당/압류 이력이 있습니다. 명의이전 가능 여부를 반드시 확인하세요.")

        year = car_info.get('year')
        if year is not None:
            year_int = int(year) if year < 9999 else int(year) // 100
            if (self.ref_year - year_int) >= 10:
                sug.append(f"차령이 {self.ref_year - year_int}년으로 노후 차량입니다. "
                           f"부품 교체 이력과 노후 부속을 점검하세요.")

        # 위험도 기반 추가 안내
        if category == 'Danger(저가)':
            sug.append("[경고] 시세 대비 비정상 저가. 직거래 사기/대포차/침수차 가능성 확인 필수.")
            sug.append("자동차등록증, 자동차이력 무료조회(보험개발원), 명의자 일치 여부를 반드시 확인하세요.")
        elif category == 'Warning(저가)':
            sug.append("실차 확인 시 색상·옵션·정비이력을 시세와 비교하세요.")

        # 누락 필드 안내
        if missing_fields:
            sug.append(f"입력되지 않은 정보({len(missing_fields)}개): "
                       f"{', '.join(missing_fields)}. 입력 시 정확도가 향상됩니다.")

        if not sug:
            sug.append("특이사항 없음. 일반적 매매 절차를 따라 진행하세요.")

        return sug

    # =========================================================
    # 7) 내부: SHAP 기반 설명
    # =========================================================
    def _get_feature_names(self):
        prep = self.pipeline.named_steps['prep']
        cat_names = list(prep.named_transformers_['cat'].get_feature_names_out(self.CAT_FEATURES))
        return cat_names + self.NUM_FEATURES

    def _explain_prediction(self, X: pd.DataFrame, top_n: int = 5):
        X_trans = self.pipeline.named_steps['prep'].transform(X)
        shap_values = self.shap_explainer.shap_values(X_trans)[0]  # (n_features,)
        # 절댓값 기준 top_n
        idx = np.argsort(np.abs(shap_values))[::-1][:top_n]
        result = []
        for i in idx:
            fname = self.feature_names_after_prep[i]
            sv = float(shap_values[i])
            # log 스케일 → 원본 영향 추정: exp(sv) - 1 = 가격에 미치는 비율
            pct_effect = (np.exp(sv) - 1) * 100
            direction = '↑' if sv > 0 else '↓'
            result.append({
                'feature': fname,
                'shap_value': round(sv, 4),
                'price_effect_pct': round(pct_effect, 2),
                'direction': direction,
            })
        return result


# =================================================================
# 결과 출력 헬퍼
# =================================================================
def print_result(result: dict):
    print("\n" + "=" * 64)
    print("  중고차 가격 왜곡 탐지 결과")
    print("=" * 64)
    print(f"  입력 가격    : {result['입력가격(만원)']:>10,.0f} 만원")
    print(f"  예상 적정가  : {result['예상적정가(만원)']:>10,.1f} 만원")
    print(f"  가격 편차율  : {result['가격편차율(%)']:>+10.2f} %")
    print(f"  분류         : {result['분류']}")
    print(f"  위험도       : {result['위험도(%)']:.1f} %")
    print(f"  판정 메시지  : {result['판정메시지']}")
    print(f"\n  추가 검토사항:")
    for i, item in enumerate(result['추가검토사항'], 1):
        print(f"    {i}. {item}")
    if '주요_영향요인' in result:
        print(f"\n  주요 영향 요인 (SHAP):")
        for f in result['주요_영향요인']:
            print(f"    {f['direction']} {f['feature']:<25} "
                  f"가격 영향 {f['price_effect_pct']:+.1f}%")
    print("=" * 64)
