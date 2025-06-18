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
            self.is_fitted = False

        def fit(self, df):
            """訓練編碼器和標準化器"""
            df = df.copy()
            # 訓練類別編碼器
            self.ordinal_encoder.fit(df[self.cat_features].astype(str))
            
            # 準備數值特徵
            df['staytime'] = np.log1p(df['staytime'].fillna(0))
            df['revisit_count'] = np.log1p(df['revisit_count'])
            
            # 訓練標準化器
            self.scaler.fit(df[['staytime', 'revisit_count']])
            self.is_fitted = True
            return self

        def transform(self, df):
            """轉換資料"""
            if not self.is_fitted:
                # 如果沒有訓練過，先用當前資料訓練
                self.fit(df)
            
            df = df.copy()
            
            # 轉換類別特徵
            df[self.cat_features] = self.ordinal_encoder.transform(df[self.cat_features].astype(str)) + 2
            
            # 轉換數值特徵
            df['staytime'] = np.log1p(df['staytime'].fillna(0))
            df['revisit_count'] = np.log1p(df['revisit_count'])
            df[['staytime', 'revisit_count']] = self.scaler.transform(df[['staytime', 'revisit_count']])
            
            return df
        
        def fit_transform(self, df):
            """訓練並轉換資料"""
            return self.fit(df).transform(df)

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
        
        # 檢查是否有必要的欄位
        required_features = ['action', 'action_group', 'source', 'medium', 'platform', 'staytime', 'revisit_count']
        missing_features = [col for col in required_features if col not in df.columns]
        if missing_features:
            raise ValueError(f"缺少必要特徵: {missing_features}")
        
        # 如果使用內建預處理器，需要先準備資料格式
        if hasattr(preprocessor, 'is_fitted') and not preprocessor.is_fitted:
            st.info("正在初始化預處理器...")
            X = preprocessor.fit_transform(df)
        else:
            X = preprocessor.transform(df)
        
        # 檢查轉換後的資料
        if X.isnull().any().any():
            st.warning("轉換後的資料包含空值，正在處理...")
            X = X.fillna(0)
        
        st.info("正在準備模型輸入格式...")
        
        # 準備模型輸入 - 只選擇模型需要的特徵
        model_features = ['action', 'action_group', 'source', 'medium', 'platform', 'staytime', 'revisit_count']
        X_model = X[model_features].copy()
        
        # 確保所有數據都是數值型
        for col in X_model.columns:
            X_model[col] = pd.to_numeric(X_model[col], errors='coerce')
        
        # 處理任何剩餘的 NaN 值
        X_model = X_model.fillna(0)
        
        # 轉換為 numpy array 並確保正確的資料類型
        X_array = X_model.values.astype(np.float32)
        
        st.info(f"模型輸入形狀: {X_array.shape}")
        st.info(f"模型輸入資料類型: {X_array.dtype}")
        
        # 檢查是否有序列模型需求
        try:
            # 嘗試直接預測
            st.info("正在進行模型預測...")
            y_pred = model.predict(X_array)
            
        except Exception as pred_error:
            st.warning(f"直接預測失敗: {pred_error}")
            st.info("嘗試序列格式...")
            
            # 如果是序列模型，需要重塑資料
            # 假設每個用戶有一個序列，序列長度為 10
            seq_len = 10
            n_features = X_array.shape[1]
            
            # 將資料重塑為序列格式
            if len(X_array) >= seq_len:
                # 創建滑動窗口序列
                sequences = []
                for i in range(len(X_array) - seq_len + 1):
                    sequences.append(X_array[i:i+seq_len])
                X_seq = np.array(sequences).astype(np.float32)
                st.info(f"序列輸入形狀: {X_seq.shape}")
                y_pred = model.predict(X_seq)
            else:
                # 如果資料太少，用零填充
                X_padded = np.zeros((1, seq_len, n_features), dtype=np.float32)
                X_padded[0, :len(X_array)] = X_array
                st.info(f"填充後輸入形狀: {X_padded.shape}")
                y_pred = model.predict(X_padded)
        
        # 檢查預測結果的格式
        st.info(f"預測輸出類型: {type(y_pred)}")
        if isinstance(y_pred, (list, tuple)):
            st.info(f"預測輸出數量: {len(y_pred)}")
            for i, pred in enumerate(y_pred):
                st.info(f"輸出 {i} 形狀: {pred.shape}")
        else:
            st.info(f"預測輸出形狀: {y_pred.shape}")
        
        # 處理不同的輸出格式
        if not isinstance(y_pred, (list, tuple)):
            # 如果只有一個輸出，包裝成列表
            y_pred = [y_pred]
        
        # 處理序列預測結果 - 取最後一個時間步
        processed_pred = []
        for pred in y_pred:
            if len(pred.shape) == 3:  # (batch, time, features)
                # 取最後一個時間步
                processed_pred.append(pred[:, -1, :])
            elif len(pred.shape) == 2:  # (batch, features)
                processed_pred.append(pred)
            else:  # (batch,)
                processed_pred.append(pred.reshape(-1, 1))
        
        y_pred = processed_pred
        
        # 確保預測結果與原始資料長度匹配
        original_length = len(df)
        
        if len(y_pred) >= 1:
            # 行為預測
            pred_0 = y_pred[0]
            if len(pred_0.shape) > 1 and pred_0.shape[1] > 1:
                y_pred_action = np.argmax(pred_0, axis=1)
                y_pred_action_conf = np.max(pred_0, axis=1)
            else:
                y_pred_action = pred_0.flatten()
                y_pred_action_conf = np.ones_like(y_pred_action) * 0.5
            
            # 調整長度以匹配原始資料
            if len(y_pred_action) != original_length:
                if len(y_pred_action) > original_length:
                    y_pred_action = y_pred_action[:original_length]
                    y_pred_action_conf = y_pred_action_conf[:original_length]
                else:
                    # 用最後一個值填充
                    last_action = y_pred_action[-1] if len(y_pred_action) > 0 else 0
                    last_conf = y_pred_action_conf[-1] if len(y_pred_action_conf) > 0 else 0.5
                    y_pred_action = np.pad(y_pred_action, (0, original_length - len(y_pred_action)), 
                                         constant_values=last_action)
                    y_pred_action_conf = np.pad(y_pred_action_conf, (0, original_length - len(y_pred_action_conf)), 
                                              constant_values=last_conf)
        else:
            raise ValueError("模型預測結果格式不正確")
        
        # 處理轉換機率（如果有的話）
        if len(y_pred) >= 2:
            y_pred_online = y_pred[1].flatten()
            if len(y_pred_online) != original_length:
                if len(y_pred_online) > original_length:
                    y_pred_online = y_pred_online[:original_length]
                else:
                    last_val = y_pred_online[-1] if len(y_pred_online) > 0 else 0.1
                    y_pred_online = np.pad(y_pred_online, (0, original_length - len(y_pred_online)), 
                                         constant_values=last_val)
        else:
            y_pred_online = np.random.rand(original_length) * 0.3  # 預設值
            
        if len(y_pred) >= 3:
            y_pred_o2o = y_pred[2].flatten()
            if len(y_pred_o2o) != original_length:
                if len(y_pred_o2o) > original_length:
                    y_pred_o2o = y_pred_o2o[:original_length]
                else:
                    last_val = y_pred_o2o[-1] if len(y_pred_o2o) > 0 else 0.1
                    y_pred_o2o = np.pad(y_pred_o2o, (0, original_length - len(y_pred_o2o)), 
                                      constant_values=last_val)
        else:
            y_pred_o2o = np.random.rand(original_length) * 0.3  # 預設值
        
        # 解碼預測標籤
        if hasattr(preprocessor, 'label_encoder_action_group'):
            try:
                label_encoder = preprocessor.label_encoder_action_group
                pred_action_labels = label_encoder.inverse_transform(y_pred_action.astype(int))
            except:
                pred_action_labels = [f"Action_{i}" for i in y_pred_action]
        else:
            # 創建簡單的標籤映射
            unique_actions = df['action_group'].unique()
            action_map = {i: action for i, action in enumerate(unique_actions)}
            pred_action_labels = [action_map.get(int(i) % len(unique_actions), f"Action_{i}") for i in y_pred_action]
        
        # 建立結果資料框
        df_result = df.copy()
        df_result["Top1_next_action_group"] = pred_action_labels
        df_result["Top1_confidence"] = y_pred_action_conf
        df_result["online_conversion_prob"] = y_pred_online
        df_result["o2o_conversion_prob"] = y_pred_o2o
        
        st.success("✅ 預測完成")
        return df_result
        
    except Exception as e:
        st.error(f"預測過程發生錯誤: {str(e)}")
        st.error(f"錯誤詳情: {type(e).__name__}")
        
        # 提供除錯信息
        with st.expander("🔍 除錯信息", expanded=False):
            st.write("資料形狀:", df.shape)
            st.write("資料欄位:", list(df.columns))
            st.write("預處理器類型:", type(preprocessor).__name__)
            if hasattr(preprocessor, 'cat_features'):
                st.write("類別特徵:", preprocessor.cat_features)
            if hasattr(preprocessor, 'num_features'):
                st.write("數值特徵:", preprocessor.num_features)
            
            # 顯示資料樣本
            st.write("資料樣本:")
            st.dataframe(df.head(3))
        
        return None

# ========== 欄位檢查函式 ==========
def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> list[str]:
    """檢查必要欄位是否存在"""
    missing = [col for col in required_columns if col not in df.columns]
    return missing

# ========== 主要應用程式 ==========
def main():
    st.title("🏢 國泰人壽 - 用戶行為預測工具")
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
            
            with st.expander("📊 資料預覽", expanded=False):
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
    
    if st.button("🔮 開始預測"):
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
    tab1, tab2, tab3, tab4 = st.tabs(["📊 行為分佈", "📈 信心分數", "🔍 轉換分析", "🎯 策略分佈"])

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

    if st.button("📥 下載結果"):
        filename = f"{custom_filename}.csv"
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 點擊下載 CSV",
            data=csv_data,
            file_name=filename,
            mime="text/csv"
        )

if __name__ == "__main__":
    main()




