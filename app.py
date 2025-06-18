# app.py
# 🚨 重要：st.set_page_config 必須是第一個 Streamlit 命令
import streamlit as st
st.set_page_config(page_title="行為預測工具", layout="wide")

import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from tensorflow.keras.models import load_model
import joblib
import os

# ========== 模型與前處理載入 ==========
@st.cache_resource
def load_model_and_preprocessor():
    log = []
    log.append("🔍 開始載入模型與前處理器...")

    model = load_model("best_model_weights_0615.h5")
    log.append("✅ 模型載入成功")

    preprocessor = joblib.load("sequence_preprocessor.pkl")
    log.append("✅ 前處理器載入成功")

    return model, preprocessor, log

# ========== 資料前處理與預測 ==========
def preprocess_and_predict(df, model, preprocessor):
    # 預處理：假設前處理器實作了 transform 方法
    X = preprocessor.transform(df)

    # 模型推論
    y_pred = model.predict(X)
    y_pred_action = np.argmax(y_pred[0], axis=1)
    y_pred_online = y_pred[1].flatten()
    y_pred_o2o = y_pred[2].flatten()
    y_pred_action_conf = np.max(y_pred[0], axis=1)

    # 還原編碼
    label_encoder = preprocessor.label_encoder_action_group
    pred_action_labels = label_encoder.inverse_transform(y_pred_action)

    df_result = df.copy()
    df_result["Top1_next_action_group"] = pred_action_labels
    df_result["Top1_confidence"] = y_pred_action_conf
    df_result["online_conversion_prob"] = y_pred_online
    df_result["o2o_conversion_prob"] = y_pred_o2o
    return df_result

# ========== 主應用邏輯 ==========
st.title("🧠 用戶行為預測工具")

# 載入模型與前處理器
with st.spinner("模型初始化中..."):
    model, preprocessor, logs = load_model_and_preprocessor()

with st.expander("🔧 載入記錄", expanded=False):
    for line in logs:
        st.markdown(f"- {line}")

# 上傳資料
uploaded_file = st.file_uploader("請上傳用戶行為資料 CSV", type=["csv"])

if uploaded_file:
    df_input = pd.read_csv(uploaded_file)
    st.success(f"✅ 成功載入 {len(df_input)} 筆資料")

    # 預測
    with st.spinner("進行預測中..."):
        df_pred = preprocess_and_predict(df_input, model, preprocessor)
        st.session_state["df_pred"] = df_pred

    st.markdown("## 📊 預測結果預覽")
    st.dataframe(df_pred.head(10), use_container_width=True)

    # 統計圖表
    st.markdown("## 📈 統計圖表")
    tab1, tab2, tab3 = st.tabs(["Top1 行為分佈", "信心分數分佈", "轉換機率"])

    with tab1:
        fig1 = px.bar(
            df_pred["Top1_next_action_group"].value_counts().reset_index(),
            x="index", y="Top1_next_action_group",
            labels={"index": "預測行為", "Top1_next_action_group": "人數"},
            title="Top1 預測行為分佈"
        )
        fig1.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        fig2 = px.histogram(
            df_pred, x="Top1_confidence", nbins=20,
            title="Top1 信心分數分佈"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig3 = px.histogram(df_pred, x="online_conversion_prob", nbins=20, title="網投轉換機率")
        fig4 = px.histogram(df_pred, x="o2o_conversion_prob", nbins=20, title="O2O 預約轉換機率")
        st.plotly_chart(fig3, use_container_width=True)
        st.plotly_chart(fig4, use_container_width=True)

    # 條件篩選與下載
    st.markdown("## 📥 結果篩選與下載")
    available_actions = df_pred["Top1_next_action_group"].unique().tolist()
    selected_actions = st.multiselect("篩選預測行為", options=available_actions, default=available_actions)
    conf_threshold = st.slider("信心分數下限", 0.0, 1.0, 0.3, step=0.05)

    df_filtered = df_pred[
        (df_pred["Top1_next_action_group"].isin(selected_actions)) &
        (df_pred["Top1_confidence"] >= conf_threshold)
    ]

    st.markdown(f"符合條件的用戶：{len(df_filtered)} 筆")
    st.dataframe(df_filtered.head(10), use_container_width=True)

    today_str = datetime.today().strftime("%Y%m%d")
    filename = st.text_input("輸出檔名", value=f"prediction_result_{today_str}.csv")
    st.download_button("下載結果 CSV", df_filtered.to_csv(index=False), file_name=filename, mime="text/csv")
else:
    st.info("請上傳行為資料以啟動預測流程")
