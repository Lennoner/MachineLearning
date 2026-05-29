# 🚗 중고차 가격 왜곡 탐지 — 웹 데모

한동대학교 ML 팀프로젝트 · 러닝(Running) 머신팀 · 2026

매물 정보 입력 → AI 적정가 예측 + 사기/허위매물 위험도 분석

---

## 실행 방법

### 1. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

> Python 3.9 이상 권장

### 2. 데모 실행

```bash
streamlit run demo_app.py
```

명령어를 실행하면 브라우저에서 자동으로 `http://localhost:8501` 이 열립니다.

---

## 파일 구성

| 파일 | 용도 |
|---|---|
| `demo_app.py` | Streamlit 웹 데모 (이 파일을 실행) |
| `detector.py` | 모델 클래스 정의 |
| `detector.joblib` | 학습된 모델 (XGBoost + SHAP, 0.9MB) |
| `categories.json` | 브랜드·모델·연료 dropdown 목록 |
| `meta.json` | 모델 메타 정보 (임계값·성능 지표) |
| `requirements.txt` | 필요한 Python 패키지 |

---

## 모델 성능

| 지표 | 값 |
|---|---|
| 학습 데이터 | 2,036 대 (Test 509 대) |
| **R² (log scale)** | **0.85** ← PPT 목표 달성 |
| R² (원본 스케일) | 0.62 |
| MAPE | 23.5% |
| 모델 | XGBoost (RandomizedSearch 튜닝) |
| 설명 가능성 | SHAP TreeExplainer |

---

## 분류 임계값

학습 잔차 분포 기반 비대칭 5단계 분류:

| 분류 | 편차율 | 의미 |
|---|---|---|
| 🚨 Danger (저가) | ≤ -25.7% | 허위매물·사기 강한 의심 |
| ⚠️ Warning (저가) | -25.7% ~ -15.3% | 약간 저렴, 확인 권장 |
| ✅ Fair | -15.3% ~ +16.4% | 정상 시세 |
| ⚠️ Warning (고가) | +16.4% ~ +34.0% | 약간 비쌈, 협상 여지 |
| 🚨 Danger (고가) | ≥ +34.0% | 과도하게 비쌈 |

---

## 시연용 예시 케이스

데모 실행 후 사이드바에 아래 값들을 입력해서 결과 비교:

| 시나리오 | 입력값 | 기대 결과 |
|---|---|---|
| **정상 매물** | 현대 / 더 뉴 그랜저 IG / 2021 / 매물가 2,400만원 | Fair (위험도 ~25%) |
| **저가 의심** | 현대 / 팰리세이드 / 2020 / 매물가 1,700만원 | Danger 저가 (위험도 ~84%) |
| **사고차 + 저가** | 기아 / 더 뉴 K5 / 2018 / 사고 8회, 수리비 1,200만원 / 매물가 950만원 | Danger 저가 (위험도 ~73%) + 사고 관련 검토사항 |
| **고가 의심** | 기아 / 카니발 4세대 / 2023 / 매물가 5,500만원 | Warning 고가 (위험도 ~45%) |

---

## 트러블슈팅

**Q. `streamlit: command not found`**
A. `pip install streamlit` 한 다음 재실행. 또는 `python -m streamlit run demo_app.py`

**Q. 모델 로드 실패**
A. `detector.py`, `detector.joblib`, `categories.json` 세 파일이 모두 같은 폴더에 있는지 확인

**Q. 한글이 깨져요**
A. 터미널 인코딩을 UTF-8로 설정 (`export LANG=ko_KR.UTF-8` 또는 Windows는 PowerShell에서 `chcp 65001`)
