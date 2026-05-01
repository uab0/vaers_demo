"""
COVID-19 疫苗不良反應風險預警系統
基於 VAERS 資料庫訓練的 XGBoost 模型，提供個體化風險評估與 SHAP 解釋。
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import shap
import matplotlib.pyplot as plt

# ============================================================
# 1. 常數與設定
# ============================================================

EXPECTED_FEATURE_COUNT = 134

FEATURE_COLUMNS = (
    ['AGE_YRS', 'SEX_F', 'NUMDAYS_LOG', 'DOSE_NUM', 'DRUG_COUNT', 'Elderly_Polypharmacy']
    + [f'ClinVec_{i}' for i in range(128)]
)

# 白話文映射表
FEATURE_NAME_MAP = {
    'AGE_YRS': '年齡',
    'SEX_F': '生理性別',
    'NUMDAYS_LOG': '發病天數',
    'DOSE_NUM': '疫苗劑次',
    'DRUG_COUNT': '用藥數量',
    'Elderly_Polypharmacy': '高齡多重用藥',
    'ClinVec_aggregated': '您目前的用藥組合',
}

# TODO: 以下燈號切點為開發階段預設值
# 依據模型校準結果與臨床意義進行調整。目前以 s2_opt_threshold (≈ 2.28) 為基準。
TH_YELLOW = 1.14   # 黃燈下限
TH_ORANGE = 2.28   # 橘燈下限 (= s2_opt_threshold)
TH_RED = 4.28      # 紅燈下限


# ============================================================
# 2. 資源載入 (快取)
# ============================================================

@st.cache_resource
def load_models():
    """載入 XGBoost 模型、機率校準器與閾值字典。"""
    try:
        with open('models/xgb_model_s2.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        calibrated_model = joblib.load('models/calibrated_model.pkl')
        with open('models/thresholds.pkl', 'rb') as f:
            thresholds = pickle.load(f)
        return xgb_model, calibrated_model, thresholds
    except FileNotFoundError as e:
        st.error(f"❌ 模型檔案載入失敗：{e}。請確認專案目錄下包含所有必要的 .pkl 檔案。")
        st.stop()
    except Exception as e:
        st.error(f"❌ 模型載入時發生錯誤：{e}")
        st.stop()


@st.cache_data
def load_drug_vectors():
    """載入藥物向量字典。"""
    try:
        df = pd.read_csv('data/rxnorm_vectors.csv')
        return df
    except FileNotFoundError:
        st.error("❌ 找不到 rxnorm_vectors.csv。請確認檔案存在於專案目錄下。")
        st.stop()
    except Exception as e:
        st.error(f"❌ 藥物字典載入時發生錯誤：{e}")
        st.stop()


# ============================================================
# 3. 特徵工程
# ============================================================

def build_features(age, sex_f, numdays, dose_num, selected_drugs, drug_vectors_df):
    """
    將使用者輸入轉換為 134 維特徵向量。

    ClinVec 聚合策略為 Mean Pooling，此決策對齊訓練階段 pipeline（見 ADR 決策二）。
    若未來訓練端改為 Max-Pooling，此處必須同步修改。

    Parameters
    ----------
    age : int
        使用者年齡
    sex_f : int
        生理性別 (女性=1, 男性=0)
    numdays : int
        發病天數 (施打後幾天出現不適症狀)
    dose_num : int
        疫苗劑次
    selected_drugs : list[str]
        使用者選取的藥物名稱列表
    drug_vectors_df : pd.DataFrame
        rxnorm_vectors.csv 的 DataFrame

    Returns
    -------
    pd.DataFrame
        1 row x 134 columns 的特徵 DataFrame
    """
    drug_count = len(selected_drugs)

    # 衍生特徵
    numdays_log = np.log1p(numdays)
    elderly_polypharmacy = 1 if (age > 65 and drug_count > 5) else 0

    # ClinVec 向量
    clinvec_cols = [f'ClinVec_{i}' for i in range(128)]
    if drug_count == 0:
        clinvec_values = [0.0] * 128
    else:
        matched = drug_vectors_df[drug_vectors_df['RXNORM_NAME'].isin(selected_drugs)]
        if len(matched) > 0:
            clinvec_values = matched[clinvec_cols].mean().tolist()
        else:
            clinvec_values = [0.0] * 128

    # 組裝特徵向量
    row = {
        'AGE_YRS': float(age),
        'SEX_F': int(sex_f),
        'NUMDAYS_LOG': float(numdays_log),
        'DOSE_NUM': float(dose_num),
        'DRUG_COUNT': int(drug_count),
        'Elderly_Polypharmacy': int(elderly_polypharmacy),
    }
    for i, val in enumerate(clinvec_values):
        row[f'ClinVec_{i}'] = float(val)

    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    # 防禦性驗證
    assert df.shape[1] == EXPECTED_FEATURE_COUNT, (
        f"特徵數量錯誤：預期 {EXPECTED_FEATURE_COUNT}，實際 {df.shape[1]}"
    )

    return df


# ============================================================
# 4. 預測與燈號
# ============================================================

def predict_risk(calibrated_model, feature_df):
    """使用校準模型預測風險分數 (0-10)。"""
    prob = calibrated_model.predict_proba(feature_df)[:, 1][0]
    risk_score = round(prob * 10, 2)
    return risk_score


def get_traffic_light(score):
    """依據風險分數判定燈號與臨床建議。"""
    if score >= TH_RED:
        return '🔴', '極高風險', '建議儘速就醫，並告知醫師您近期的疫苗接種紀錄與症狀。'
    elif score >= TH_ORANGE:
        return '🟠', '高風險', '建議密切留意身體狀況，若症狀持續或加劇，請儘早就醫。'
    elif score >= TH_YELLOW:
        return '🟡', '中等風險', '請留意自身身體變化，若出現不適症狀請諮詢醫師。'
    else:
        return '🟢', '低風險', '目前評估風險較低，請維持正常作息並留意身體狀況。'


# ============================================================
# 5. SHAP 解釋
# ============================================================

def explain_with_shap(xgb_model, feature_df):
    """
    計算 SHAP 值並產生聚合後的白話文解釋資料。

    ClinVec_0 ~ ClinVec_127 的 SHAP 值加總為「用藥組合」單一項目，
    避免向民眾曝露無語義意義的向量維度編號。
    """
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(feature_df)

    raw_values = shap_values[0].values  # shape: (134,)

    # --- 分層聚合 ---
    interpretable_features = [
        'AGE_YRS', 'SEX_F', 'NUMDAYS_LOG',
        'DOSE_NUM', 'DRUG_COUNT', 'Elderly_Polypharmacy'
    ]
    aggregated = {}
    for i, name in enumerate(interpretable_features):
        aggregated[name] = float(raw_values[i])

    # 128 個 ClinVec 維度加總為單一項目
    clinvec_sum = float(np.sum(raw_values[6:]))
    aggregated['ClinVec_aggregated'] = clinvec_sum

    # 依絕對值排序
    sorted_items = sorted(aggregated.items(), key=lambda x: abs(x[1]), reverse=True)

    # 分類：推升 vs 降低
    risk_up = [(k, v) for k, v in sorted_items if v > 0][:3]
    risk_down = [(k, v) for k, v in sorted_items if v < 0][:3]

    return shap_values, risk_up, risk_down


def render_shap_text(risk_up, risk_down):
    """將 SHAP 聚合結果渲染為白話文條列。"""
    if risk_up:
        st.markdown("**⬆ 推升風險因素：**")
        for name, val in risk_up:
            display_name = FEATURE_NAME_MAP.get(name, name)
            st.error(f"• {display_name}（影響程度：{abs(val):.4f}）")

    if risk_down:
        st.markdown("**⬇ 降低風險因素：**")
        for name, val in risk_down:
            display_name = FEATURE_NAME_MAP.get(name, name)
            st.success(f"• {display_name}（影響程度：{abs(val):.4f}）")

    if not risk_up and not risk_down:
        st.info("此次分析中各項特徵的影響均不顯著。")


def render_shap_waterfall(shap_values):
    """在 expander 中渲染原始 134 維 SHAP Waterfall 圖。"""
    with st.markdown("特徵貢獻圖"):
        shap.plots.waterfall(shap_values[0], max_display=15, show=False)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.clf()
        plt.close('all')


# ============================================================
# 6. 主程式 (UI)
# ============================================================

def main():
    st.set_page_config(
        page_title="COVID-19 疫苗不良反應風險預警系統",
        layout="wide"
    )

    st.title("COVID-19 疫苗不良反應風險預警系統")
    st.markdown("本系統基於 VAERS 資料庫訓練之 XGBoost 模型，提供個體化風險評估與 SHAP 解釋。")
    st.caption(
        "⚠️ 免責聲明：本系統僅供參考，不構成醫療診斷。"
        "當前燈號切點為開發階段預設值，尚未經臨床驗證。如有不適，請儘速就醫。"
    )

    # 載入資源
    xgb_model, calibrated_model, thresholds = load_models()
    drug_vectors_df = load_drug_vectors()
    drug_names = sorted(drug_vectors_df['RXNORM_NAME'].dropna().unique().tolist())

    # ---- 基本資料輸入 ----
    st.subheader("基本資料輸入")

    age = st.number_input(
        "年齡", min_value=0, max_value=120, value=30, step=1
    ) # AGE_YRS
    sex = st.radio("生理性別", ["男性", "女性"], horizontal=True)
    sex_f = 1 if sex == "女性" else 0
    dose_num = st.number_input(
        "疫苗劑次（尚未施打者請填預計施打劑次）",
        min_value=1, max_value=10, value=1, step=1
    ) # DOSE_NUM

    if age < 5:
        st.info(
            "提醒：COVID-19 疫苗在多數國家建議 5 歲以上接種，"
            "低齡評估結果可能參考價值有限。"
        )

    st.divider()

    # ---- 症狀與用藥 ----
    symptom_status = st.radio(
        "施打疫苗後是否出現不適症狀？",
        ["是，已出現不適症狀", "否，已施打但無不適", "尚未施打疫苗"],
        horizontal=True
    )

    if symptom_status == "是，已出現不適症狀":
        numdays = st.number_input(
            "施打後幾天開始出現不適症狀？", min_value=1, max_value=365, value=1, step=1
        ) # NUMDAYS
    else:
        # 無症狀或尚未施打：以中位數 1 天作為預設值
        numdays = 1
        if symptom_status == "尚未施打疫苗":
            st.info(
                "此為施打前風險預估，系統將以您的基本資料與用藥狀況進行評估，"
                "結果僅供參考。"
            )

    st.subheader("用藥狀況")
    selected_drugs = st.multiselect(
        "請選擇您目前正在服用的藥物（可多選）",
        options=drug_names,
        help="輸入藥物名稱即可搜尋篩選"
    )
    if selected_drugs:
        st.info(f"已選擇 {len(selected_drugs)} 種藥物：{', '.join(selected_drugs)}")

    st.divider()

    # ---- 分析按鈕 ----
    if st.button("開始分析", type="primary"):
        with st.spinner("正在進行風險分析..."):
            # 特徵工程
            feature_df = build_features(
                age=age,
                sex_f=sex_f,
                numdays=numdays,
                dose_num=dose_num,
                selected_drugs=selected_drugs,
                drug_vectors_df=drug_vectors_df
            )

            # 預測
            risk_score = predict_risk(calibrated_model, feature_df)
            light, level, advice = get_traffic_light(risk_score)

            # 顯示結果
            col1, col2 = st.columns([1, 1])

            with col1:
                st.metric(label="風險分數", value=f"{risk_score} / 10")

            with col2:
                st.markdown(f"### {light} {level}")
                st.markdown(f"**臨床建議：** {advice}")

            st.divider()

            # SHAP 解釋
            st.subheader("風險因子解讀")
            st.caption(
                "風險因子分析基於模型原始預測邏輯，"
                "風險分數經過機率校準後可能略有差異。"
            )

            try:
                shap_values, risk_up, risk_down = explain_with_shap(
                    xgb_model, feature_df
                )
                render_shap_text(risk_up, risk_down)
                render_shap_waterfall(shap_values)
            except Exception as e:
                st.warning(f"SHAP 解釋功能暫時無法使用：{e}")


if __name__ == "__main__":
    main()
