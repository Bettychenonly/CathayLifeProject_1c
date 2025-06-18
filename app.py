# app.py - 完整 UI 版本（含步驟導引與頁籤）
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import joblib
import os

st.set_page_config(page_title="國泰人壽 - 用戶行為預測工具", layout="centered", initial_sidebar_state="collapsed")

# ========== 初始化 Session State ==========
if "raw_uploaded_data" not in st.session_state:
    st.session_state.raw_uploaded_data = None
if "filtered_input_data" not in st.session_state:
    st.session_state.filtered_input_data = None
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None
if "filtered_prediction_data" not in st.session_state:
    st.session_state.filtered_prediction_data = None

# ========== 模型與前處理載入 ==========
@st.cache_resource
def load_model_and_preprocessor():
    log = []
    
    # 檢查模型檔案是否存在
    if not os.path.exists("model_0615"):
        log.append("❌ 模型檔案 'model_0615' 不存在")
        return None, None, log
    
    try:
        model = load_model("model_0615")
        log.append("✅ 模型載入成功")
    except Exception as e:
        log.append(f"❌ 模型載入失敗: {str(e)}")
        return None, None, log

    # 方法1: 使用 joblib 載入預處理器
    if os.path.exists("sequence_preprocessor.pkl"):
        try:
            preprocessor = joblib.load("sequence_preprocessor.pkl")
            log.append("✅ 前處理器載入成功")
            return model, preprocessor, log
        except Exception as e:
            log.append(f"❌ 前處理器載入失敗: {str(e)}")
    
    # 方法2: 如果 pkl 檔案不存在，使用內建類別
    log.append("⚠️ 使用內建前處理器類別")
    
    class SequencePreprocessor:
        def __init__(self, cat_features, num_features, seq_len=10):
            self.cat_features = cat_features
            self.num_features = num_features
            self.seq_len = seq_len
            self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            self.scaler = StandardScaler()
            self.num_categories = {}

        def transform(self, df):
            df = df.copy()
            df[self.cat_features] = self.ordinal_encoder.transform(df[self.cat_features].astype(str)) + 2
            df['staytime'] = np.log1p(df['staytime'].fillna(0))
            df['revisit_count'] = np.log1p(df['revisit_count'])
            df[['staytime', 'revisit_count']] = self.scaler.transform(df[['staytime', 'revisit_count']])
            return df

    # 初始化預設的前處理器
    cat_features = ['action', 'action_group', 'source', 'medium', 'platform']
    num_features = ['staytime', 'revisit_count']
    preprocessor = SequencePreprocessor(cat_features, num_features)
    log.append("✅ 內建前處理器初始化成功")
    
    return model, preprocessor, log

# ========== 前處理函式 ==========
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """清理和預處理資料框"""
    df = df.copy()
    df = df[~(df['action'].isna() & (df['action_group'] == '其他'))]
    df['action'].fillna(df['action_group'], inplace=True)
    df['source'] = df['source'].fillna('None')
    df['medium'] = df['medium'].fillna('None')
    df['staytime'] = df['staytime'].fillna(0)
    df['has_shared'] = df['has_shared'].fillna(False)
    return df

# ========== 預測函式 ==========
def preprocess_and_predict(df, model, preprocessor):
    """預處理資料並進行預測"""
    try:
        df = clean_dataframe(df)
        X = preprocessor.transform(df)
        y_pred = model.predict(X)
        
        # 處理預測結果
        y_pred_action = np.argmax(y_pred[0], axis=1)
        y_pred_online = y_pred[1].flatten()
        y_pred_o2o = y_pred[2].flatten()
        y_pred_action_conf = np.max(y_pred[0], axis=1)
        
        # 解碼預測標籤
        if hasattr(preprocessor, 'label_encoder_action_group'):
            label_encoder = preprocessor.label_encoder_action_group
            pred_action_labels = label_encoder.inverse_transform(y_pred_action)
        else:
            # 如果沒有標籤編碼器，使用數字標籤
            pred_action_labels = [f"Action_{i}" for i in y_pred_action]
        
        # 建立結果資料框
        df_result = df.copy()
        df_result["Top1_next_action_group"] = pred_action_labels
        df_result["Top1_confidence"] = y_pred_action_conf
        df_result["online_conversion_prob"] = y_pred_online
        df_result["o2o_conversion_prob"] = y_pred_o2o
        
        return df_result
        
    except Exception as e:
        st.error(f"預測過程發生錯誤: {str(e)}")
        return None

# ========== 欄位檢查函式 ==========
def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> list[str]:
    """檢查必要欄位是否存在"""
    missing = [col for col in required_columns if col not in df.columns]
    return missing

# ========== 主要應用程式 ==========
def main():
    st.title("國泰人壽 - 用戶行為預測工具")
    st.markdown("---")
    
    # ========== 步驟 1: 上傳資料 ==========
    st.markdown("### 步驟 1: 上傳資料")
    uploaded_file = st.file_uploader("請上傳用戶行為資料 (CSV 檔)", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.raw_uploaded_data = df
            
            required_columns = [
                "user_pseudo_id", "event_time", "action", "action_group", 
                "source", "medium", "platform", "staytime", "has_shared", "revisit_count"
            ]
            missing_cols = validate_columns(df, required_columns)
            
            if missing_cols:
                st.error(f"❌ 缺少必要欄位：{', '.join(missing_cols)}")
                st.stop()
                
            st.success(f"✅ 成功讀取 {len(df)} 筆資料，欄位完整")
            
            with st.expander("資料預覽", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ 讀取檔案時發生錯誤: {str(e)}")
            st.stop()
    else:
        st.info("請先上傳 CSV 檔案")
        st.stop()

    # ========== 步驟 2: 模型與資料載入 ==========
    st.markdown("### 步驟 2: 載入模型與前處理")
    
    with st.spinner("正在載入模型..."):
        model, preprocessor, logs = load_model_and_preprocessor()
    
    if model is None or preprocessor is None:
        st.error("❌ 模型或前處理器載入失敗")
        with st.expander("🧾 載入記錄", expanded=True):
            for line in logs:
                st.markdown(f"- {line}")
        st.stop()
    
    st.success("✅ 模型與前處理器載入成功")
    with st.expander("🧾 載入記錄", expanded=False):
        for line in logs:
            st.markdown(f"- {line}")

    # ========== 步驟 3: 開始預測 ==========
    st.markdown("### 步驟 3: 開始預測")
    
    if st.button("開始預測"):
        with st.spinner("預測中..."):
            df_pred = preprocess_and_predict(st.session_state.raw_uploaded_data, model, preprocessor)
            
            if df_pred is not None:
                st.session_state.prediction_data = df_pred
                st.success("✅ 預測完成！")
            else:
                st.error("❌ 預測失敗")
                st.stop()
    else:
        if "prediction_data" not in st.session_state or st.session_state.prediction_data is None:
            st.info("請點擊「開始預測」按鈕")
            st.stop()

    # ========== 步驟 4: 預測結果預覽 ==========
    st.markdown("### 步驟 4: 預測結果預覽")
    df_pred = st.session_state.prediction_data
    st.dataframe(df_pred.head(10), use_container_width=True)

    # ========== 步驟 5: 圖表統計 ==========
    st.markdown("### 步驟 5: 統計圖表")
    tab1, tab2, tab3, tab4 = st.tabs(["行為分佈", "信心分數", "轉換分析", "策略分佈"])

    with tab1:
        chart_df = df_pred["Top1_next_action_group"].value_counts().reset_index()
        chart_df.columns = ["action_group", "count"]
        fig1 = px.bar(chart_df, x="action_group", y="count", title="Top1 預測行為分佈")
        fig1.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        fig2 = px.histogram(
            df_pred,
            x="Top1_confidence",
            nbins=20,
            title="Top1 預測信心分數分佈（人數）",
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig3 = px.histogram(
            df_pred, x="online_conversion_prob", nbins=20, title="網投轉換機率分佈"
        )
        fig4 = px.histogram(
            df_pred, x="o2o_conversion_prob", nbins=20, title="O2O 預約轉換機率分佈"
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.plotly_chart(fig4, use_container_width=True)

    with tab4:
        st.info("可日後擴充策略推薦邏輯 (依 Top1 行為給定建議)")

    # ========== 步驟 6: 確認條件並下載 ==========
    st.markdown("### 步驟 6: 確認條件並下載")

    filtered_df = st.session_state.get("prediction_data", pd.DataFrame()).copy()
    st.markdown(f"**目前符合條件的用戶數量**：{len(filtered_df)} 人")

    if len(filtered_df) == 0:
        st.warning("⚠️ 目前條件下沒有符合的用戶，請調整條件後再試")
        st.stop()

    # 條件選擇
    available_actions = filtered_df["Top1_next_action_group"].unique().tolist()
    selected_actions = st.multiselect("篩選預測行為", options=available_actions, default=available_actions)
    conf_threshold = st.slider("信心分數下限", 0.0, 1.0, 0.3, step=0.05)

    # 應用篩選條件
    filtered_df = filtered_df[
        (filtered_df["Top1_next_action_group"].isin(selected_actions)) &
        (filtered_df["Top1_confidence"] >= conf_threshold)
    ]
    st.session_state.filtered_prediction_data = filtered_df
    st.markdown(f"**篩選後用戶數量**：{len(filtered_df)} 人")

    # 下載區塊
    today_str = datetime.now().strftime("%Y%m%d")
    default_filename = f"prediction_result_{len(filtered_df)}users_{today_str}"
    custom_filename = st.text_input(
        "自訂檔名（選填，系統會自動加上 .csv）",
        value=default_filename,
        placeholder="ex: 旅平險_Top3_信心0.3"
    )

    if st.button("下載結果"):
        filename = f"{custom_filename}.csv"
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="點擊下載 CSV",
            data=csv_data,
            file_name=filename,
            mime="text/csv"
        )

if __name__ == "__main__":
    main()





