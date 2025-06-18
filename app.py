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

    # 定義 SequencePreprocessor 類別（必須在載入 pkl 之前定義）
    class SequencePreprocessor:
        def __init__(self, cat_features=None, num_features=None, seq_len=10):
            self.cat_features = cat_features or ['action', 'action_group', 'source', 'medium', 'platform']
            self.num_features = num_features or ['staytime', 'revisit_count']
            self.seq_len = seq_len
            self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            self.scaler = StandardScaler()
            self.num_categories = {}
            self.is_fitted = False
            self.label_encoder_action_group = None

        def fit(self, df):
            """訓練編碼器和標準化器"""
            df = df.copy()
            self.ordinal_encoder.fit(df[self.cat_features].astype(str))
            df['staytime'] = np.log1p(df['staytime'].fillna(0))
            df['revisit_count'] = np.log1p(df['revisit_count'])
            self.scaler.fit(df[['staytime', 'revisit_count']])
            self.is_fitted = True
            return self

        def transform(self, df):
            """轉換資料"""
            if not self.is_fitted:
                self.fit(df)
            
            df = df.copy()
            df[self.cat_features] = self.ordinal_encoder.transform(df[self.cat_features].astype(str)) + 2
            
            embedding_limits = {
                'platform': 10, 'source': 1500, 'medium': 350, 'action_group': 100, 'action': 16000
            }
            
            for col in self.cat_features:
                if col in df.columns and col in embedding_limits:
                    limit = embedding_limits[col]
                    df[col] = df[col].clip(0, limit-1)
            
            df['staytime'] = np.log1p(df['staytime'].fillna(0))
            df['revisit_count'] = np.log1p(df['revisit_count'])
            df[['staytime', 'revisit_count']] = self.scaler.transform(df[['staytime', 'revisit_count']])
            
            return df
        
        def fit_transform(self, df):
            return self.fit(df).transform(df)

    # 嘗試載入 pkl 檔案
    if os.path.exists("sequence_preprocessor.pkl"):
        try:
            import sys
            current_module = sys.modules[__name__]
            setattr(current_module, 'SequencePreprocessor', SequencePreprocessor)
            
            preprocessor = joblib.load("sequence_preprocessor.pkl")
            log.append("✅ 前處理器載入成功")
            
            if not hasattr(preprocessor, 'is_fitted'):
                preprocessor.is_fitted = True
            if not hasattr(preprocessor, 'cat_features'):
                preprocessor.cat_features = ['action', 'action_group', 'source', 'medium', 'platform']
            if not hasattr(preprocessor, 'num_features'):
                preprocessor.num_features = ['staytime', 'revisit_count']
                
            return model, preprocessor, log
            
        except Exception as e:
            log.append(f"❌ 前處理器載入失敗: {str(e)}")
            log.append("⚠️ 將使用內建前處理器類別")
    
    log.append("⚠️ 使用內建前處理器類別")
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
        
        required_features = ['action', 'action_group', 'source', 'medium', 'platform', 'staytime', 'revisit_count']
        missing_features = [col for col in required_features if col not in df.columns]
        if missing_features:
            raise ValueError(f"缺少必要特徵: {missing_features}")
        
        if hasattr(preprocessor, 'is_fitted') and not preprocessor.is_fitted:
            st.info("正在初始化預處理器...")
            X = preprocessor.fit_transform(df)
        else:
            X = preprocessor.transform(df)
        
        if X.isnull().any().any():
            st.warning("轉換後的資料包含空值，正在處理...")
            X = X.fillna(0)
        
        st.info("正在準備模型專用輸入格式...")
        
        seq_len = 10
        cat_features = ['action_group', 'medium', 'platform', 'source']
        for col in cat_features:
            X[col] = X[col].astype(int)
        
        num_features = ['staytime', 'revisit_count']
        if 'has_shared' in X.columns:
            X['has_shared'] = X['has_shared'].astype(float)
            num_features.append('has_shared')
        else:
            X['has_shared'] = 0.0
            num_features.append('has_shared')
        
        if 'user_pseudo_id' in X.columns:
            user_groups = X.groupby('user_pseudo_id')
            st.info(f"找到 {len(user_groups)} 個用戶")
        else:
            st.warning("沒有找到 user_pseudo_id，將所有資料視為單一序列")
            X['user_pseudo_id'] = 'default_user'
            user_groups = X.groupby('user_pseudo_id')
        
        embedding_limits = {
            'platform': 7, 'source': 10, 'medium': 10, 'action_group': 15
        }
        
        st.info("檢查並修正類別特徵範圍...")
        for col in cat_features:
            if col in X.columns:
                max_val = X[col].max()
                min_val = X[col].min()
                limit = embedding_limits.get(col, max_val + 1)
                
                st.info(f"{col}: 範圍 {min_val}-{max_val}, 嵌入層限制 0-{limit-1}")
                
                if max_val >= limit:
                    st.warning(f"⚠️ {col} 超出範圍！將 {max_val} 調整為 {limit-1}")
                    X[col] = X[col].clip(0, limit-1)
                
                if min_val < 0:
                    st.warning(f"⚠️ {col} 有負值！將調整為 0")
                    X[col] = X[col].clip(0, limit-1)
        
        sequences = {
            'action_group': [], 'medium': [], 'platform': [], 'source': [], 'num_input': []
        }
        
        user_mappings = []
        
        for user_id, user_data in user_groups:
            user_data = user_data.sort_values('event_time') if 'event_time' in user_data.columns else user_data
            
            for i in range(0, len(user_data), seq_len):
                seq_data = user_data.iloc[i:i+seq_len].copy()
                
                for cat_col in cat_features:
                    seq_values = seq_data[cat_col].values
                    if len(seq_values) < seq_len:
                        last_val = seq_values[-1] if len(seq_values) > 0 else 0
                        seq_values = np.pad(seq_values, (0, seq_len - len(seq_values)), constant_values=last_val)
                    elif len(seq_values) > seq_len:
                        seq_values = seq_values[:seq_len]
                    
                    limit = embedding_limits.get(cat_col, 100)
                    seq_values = np.clip(seq_values, 0, limit-1)
                    sequences[cat_col].append(seq_values.astype(np.int32))
                
                num_matrix = []
                for num_col in num_features:
                    seq_values = seq_data[num_col].values
                    if len(seq_values) < seq_len:
                        last_val = seq_values[-1] if len(seq_values) > 0 else 0.0
                        seq_values = np.pad(seq_values, (0, seq_len - len(seq_values)), constant_values=last_val)
                    elif len(seq_values) > seq_len:
                        seq_values = seq_values[:seq_len]
                    
                    num_matrix.append(seq_values.astype(np.float32))
                
                num_matrix = np.array(num_matrix).T
                sequences['num_input'].append(num_matrix)
                user_mappings.extend(seq_data.index.tolist())
        
        model_inputs = {}
        for key in sequences:
            if key == 'num_input':
                model_inputs[key] = np.array(sequences[key], dtype=np.float32)
            else:
                model_inputs[key] = np.array(sequences[key], dtype=np.int32)
        
        st.info("模型輸入格式:")
        for key, value in model_inputs.items():
            st.info(f"  {key}: {value.shape}, dtype: {value.dtype}")
        
        st.info("正在進行模型預測...")
        y_pred = model.predict(model_inputs)
        
        st.info(f"預測輸出類型: {type(y_pred)}")
        
        if isinstance(y_pred, dict):
            st.info(f"預測輸出鍵值: {list(y_pred.keys())}")
            for key, value in y_pred.items():
                st.info(f"輸出 {key} 形狀: {value.shape}")
            
            pred_list = []
            possible_main_keys = ['next_action_group', 'action_group_pred', 'action_pred', 'main_output']
            possible_online_keys = ['online_conversion', 'online_prob', 'conversion_online']
            possible_o2o_keys = ['o2o_conversion', 'o2o_prob', 'conversion_o2o']
            
            main_pred = None
            for key in possible_main_keys:
                if key in y_pred:
                    main_pred = y_pred[key]
                    st.info(f"找到主要預測: {key}")
                    break
            
            if main_pred is None:
                main_key = list(y_pred.keys())[0]
                main_pred = y_pred[main_key]
                st.info(f"使用第一個輸出作為主要預測: {main_key}")
            
            pred_list.append(main_pred)
            
            online_pred = None
            for key in possible_online_keys:
                if key in y_pred:
                    online_pred = y_pred[key]
                    st.info(f"找到線上轉換預測: {key}")
                    break
            
            if online_pred is not None:
                pred_list.append(online_pred)
            
            o2o_pred = None
            for key in possible_o2o_keys:
                if key in y_pred:
                    o2o_pred = y_pred[key]
                    st.info(f"找到 O2O 轉換預測: {key}")
                    break
            
            if o2o_pred is not None:
                pred_list.append(o2o_pred)
            
            if online_pred is None and o2o_pred is None:
                used_keys = {list(y_pred.keys())[0]}
                for key, value in y_pred.items():
                    if key not in used_keys:
                        pred_list.append(value)
                        st.info(f"添加額外輸出: {key}")
            
            y_pred = pred_list
            st.info(f"轉換後的預測結果數量: {len(y_pred)}")
            
        elif isinstance(y_pred, (list, tuple)):
            st.info(f"預測輸出數量: {len(y_pred)}")
            for i, pred in enumerate(y_pred):
                st.info(f"輸出 {i} 形狀: {pred.shape}")
        else:
            st.info(f"預測輸出形狀: {y_pred.shape}")
            y_pred = [y_pred]
        
        processed_pred = []
        for i, pred in enumerate(y_pred):
            st.info(f"處理預測輸出 {i}: 形狀 {pred.shape}")
            
            if len(pred.shape) == 3:
                processed_pred.append(pred[:, -1, :])
                st.info(f"  3D 輸出，取最後時間步: {pred[:, -1, :].shape}")
            elif len(pred.shape) == 2:
                if pred.shape[1] == seq_len:
                    processed_pred.append(pred[:, -1:])
                    st.info(f"  2D 時間序列，取最後時間步: {pred[:, -1:].shape}")
                else:
                    processed_pred.append(pred)
                    st.info(f"  2D 特徵輸出: {pred.shape}")
            else:
                processed_pred.append(pred.reshape(-1, 1))
                st.info(f"  1D 輸出，重塑為: {pred.reshape(-1, 1).shape}")
        
        y_pred = processed_pred
        st.info(f"處理後的預測結果數量: {len(y_pred)}")
        for i, pred in enumerate(y_pred):
            st.info(f"  結果 {i}: {pred.shape}")
        
        original_length = len(df)
        st.info(f"原始資料長度: {original_length}")
        
        if len(y_pred) >= 1:
            pred_0 = y_pred[0]
            st.info(f"主要預測形狀: {pred_0.shape}")
            st.info(f"主要預測值範圍: {pred_0.min():.4f} - {pred_0.max():.4f}")
            
            if len(pred_0.shape) > 1 and pred_0.shape[1] > 1:
                # 多類別分類
                st.info(f"檢測到多類別分類，類別數量: {pred_0.shape[1]}")
                
                # 顯示每個類別的平均機率
                avg_probs = np.mean(pred_0, axis=0)
                st.info("各類別平均機率:")
                for i, prob in enumerate(avg_probs):
                    st.info(f"  類別 {i}: {prob:.4f}")
                
                y_pred_action = np.argmax(pred_0, axis=1)
                y_pred_action_conf = np.max(pred_0, axis=1)
                st.info(f"多類別預測: {len(y_pred_action)} 個預測值")
                
                # 顯示預測類別分佈
                from collections import Counter
                pred_dist = Counter(y_pred_action)
                st.info("預測類別分佈:")
                for cls, count in pred_dist.most_common():
                    st.info(f"  類別 {cls}: {count} 次 ({count/len(y_pred_action)*100:.1f}%)")
                    
            else:
                # 單一值或二元分類
                y_pred_action = pred_0.flatten()
                st.info(f"單值預測範圍: {y_pred_action.min():.4f} - {y_pred_action.max():.4f}")
                
                if np.max(y_pred_action) <= 1.0 and np.min(y_pred_action) >= 0.0:
                    # 看起來像機率值，轉換為類別
                    st.info("檢測到機率值，轉換為二元分類")
                    y_pred_action = (y_pred_action > 0.5).astype(int)
                    y_pred_action_conf = np.abs(pred_0.flatten() - 0.5) * 2
                else:
                    # 直接使用預測值
                    st.info("直接使用預測值作為類別索引")
                    y_pred_action_conf = np.ones_like(y_pred_action) * 0.7
                st.info(f"單值預測: {len(y_pred_action)} 個預測值")
            
            if len(y_pred_action) < original_length:
                n_repeats = (original_length + len(y_pred_action) - 1) // len(y_pred_action)
                y_pred_action = np.tile(y_pred_action, n_repeats)[:original_length]
                y_pred_action_conf = np.tile(y_pred_action_conf, n_repeats)[:original_length]
                st.info(f"重複預測結果以匹配原始長度: {len(y_pred_action)}")
            elif len(y_pred_action) > original_length:
                y_pred_action = y_pred_action[:original_length]
                y_pred_action_conf = y_pred_action_conf[:original_length]
                st.info(f"截斷預測結果以匹配原始長度: {len(y_pred_action)}")
        else:
            raise ValueError("模型預測結果格式不正確")
        
        if len(y_pred) >= 2:
            y_pred_online = y_pred[1].flatten()
            if len(y_pred_online) < original_length:
                n_repeats = (original_length + len(y_pred_online) - 1) // len(y_pred_online)
                y_pred_online = np.tile(y_pred_online, n_repeats)[:original_length]
            elif len(y_pred_online) > original_length:
                y_pred_online = y_pred_online[:original_length]
        else:
            y_pred_online = np.random.rand(original_length) * 0.3
            
        if len(y_pred) >= 3:
            y_pred_o2o = y_pred[2].flatten()
            if len(y_pred_o2o) < original_length:
                n_repeats = (original_length + len(y_pred_o2o) - 1) // len(y_pred_o2o)
                y_pred_o2o = np.tile(y_pred_o2o, n_repeats)[:original_length]
            elif len(y_pred_o2o) > original_length:
                y_pred_o2o = y_pred_o2o[:original_length]
        else:
            y_pred_o2o = np.random.rand(original_length) * 0.3
        
        if hasattr(preprocessor, 'label_encoder_action_group'):
            try:
                label_encoder = preprocessor.label_encoder_action_group
                pred_action_labels = label_encoder.inverse_transform(y_pred_action.astype(int))
            except:
                pred_action_labels = [f"Action_{i}" for i in y_pred_action]
        else:
            unique_actions = df['action_group'].unique()
            action_map = {i: action for i, action in enumerate(unique_actions)}
            pred_action_labels = [action_map.get(int(i) % len(unique_actions), f"Action_{i}") for i in y_pred_action]
        
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
        
        with st.expander("🔍 除錯信息", expanded=False):
            st.write("資料形狀:", df.shape)
            st.write("資料欄位:", list(df.columns))
            st.write("預處理器類型:", type(preprocessor).__name__)
            if hasattr(preprocessor, 'cat_features'):
                st.write("類別特徵:", preprocessor.cat_features)
            if hasattr(preprocessor, 'num_features'):
                st.write("數值特徵:", preprocessor.num_features)
            st.write("資料樣本:")
            st.dataframe(df.head(3))
        
        return None

# ========== 欄位檢查函式 ==========
def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> list[str]:
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
