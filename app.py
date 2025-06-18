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
            # 可能存在的其他屬性
            self.label_encoder_action_group = None

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
            
            # 根據模型嵌入層限制調整類別特徵範圍
            embedding_limits = {
                'platform': 7,     # platform 嵌入層大小
                'source': 15,      # 估計值
                'medium': 15,      # 估計值
                'action_group': 20, # 估計值
                'action': 25       # 估計值（如果有的話）
            }
            
            for col in self.cat_features:
                if col in df.columns and col in embedding_limits:
                    limit = embedding_limits[col]
                    # 將值限制在 [0, limit-1] 範圍內
                    df[col] = df[col].clip(0, limit-1)
            
            # 轉換數值特徵
            df['staytime'] = np.log1p(df['staytime'].fillna(0))
            df['revisit_count'] = np.log1p(df['revisit_count'])
            df[['staytime', 'revisit_count']] = self.scaler.transform(df[['staytime', 'revisit_count']])
            
            return df
        
        def fit_transform(self, df):
            """訓練並轉換資料"""
            return self.fit(df).transform(df)

    # 嘗試載入 pkl 檔案
    if os.path.exists("sequence_preprocessor.pkl"):
        try:
            # 先將 SequencePreprocessor 加入到全域命名空間，讓 pickle 能找到
            import sys
            current_module = sys.modules[__name__]
            setattr(current_module, 'SequencePreprocessor', SequencePreprocessor)
            
            preprocessor = joblib.load("sequence_preprocessor.pkl")
            log.append("✅ 前處理器載入成功")
            
            # 檢查載入的前處理器是否有必要的屬性
            if not hasattr(preprocessor, 'is_fitted'):
                preprocessor.is_fitted = True  # 假設已經訓練過
            if not hasattr(preprocessor, 'cat_features'):
                preprocessor.cat_features = ['action', 'action_group', 'source', 'medium', 'platform']
            if not hasattr(preprocessor, 'num_features'):
                preprocessor.num_features = ['staytime', 'revisit_count']
                
            return model, preprocessor, log
            
        except Exception as e:
            log.append(f"❌ 前處理器載入失敗: {str(e)}")
            log.append("⚠️ 將使用內建前處理器類別")
    
    # 如果 pkl 檔案不存在或載入失敗，使用內建類別
    log.append("⚠️ 使用內建前處理器類別")
    
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
        
        st.info("正在準備模型專用輸入格式...")
        
        # 根據模型需求準備輸入格式
        seq_len = 10  # 模型期望的序列長度
        
        # 確保所有類別特徵都是整數
        cat_features = ['action_group', 'medium', 'platform', 'source']
        for col in cat_features:
            X[col] = X[col].astype(int)
        
        # 準備數值特徵 (staytime, revisit_count, 可能還需要 has_shared)
        num_features = ['staytime', 'revisit_count']
        if 'has_shared' in X.columns:
            X['has_shared'] = X['has_shared'].astype(float)
            num_features.append('has_shared')
        else:
            # 如果沒有 has_shared，創建一個預設值
            X['has_shared'] = 0.0
            num_features.append('has_shared')
        
        # 按用戶分組準備序列資料
        if 'user_pseudo_id' in X.columns:
            # 按用戶分組
            user_groups = X.groupby('user_pseudo_id')
            st.info(f"找到 {len(user_groups)} 個用戶")
        else:
            # 如果沒有用戶ID，假設所有資料是一個序列
            st.warning("沒有找到 user_pseudo_id，將所有資料視為單一序列")
            X['user_pseudo_id'] = 'default_user'
            user_groups = X.groupby('user_pseudo_id')
        
        # 檢查並修正類別特徵的範圍
        # 模型的嵌入層大小限制（根據錯誤訊息推斷）
        embedding_limits = {
            'platform': 7,    # platform 嵌入層大小為 7 (索引 0-6)
            'source': 10,     # 估計值，可能需要調整
            'medium': 10,     # 估計值，可能需要調整  
            'action_group': 15 # 估計值，可能需要調整
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
                    # 將超出範圍的值映射到有效範圍內
                    X[col] = X[col].clip(0, limit-1)
                
                if min_val < 0:
                    st.warning(f"⚠️ {col} 有負值！將調整為 0")
                    X[col] = X[col].clip(0, limit-1)
        
        # 為每個用戶創建序列
        sequences = {
            'action_group': [],
            'medium': [],
            'platform': [],
            'source': [],
            'num_input': []
        }
        
        user_mappings = []  # 記錄每個序列對應的原始資料索引
        
        for user_id, user_data in user_groups:
            user_data = user_data.sort_values('event_time') if 'event_time' in user_data.columns else user_data
            
            # 如果用戶資料少於序列長度，用最後一個值填充
            for i in range(0, len(user_data), seq_len):
                seq_data = user_data.iloc[i:i+seq_len].copy()
                
                # 準備序列資料
                seq_dict = {}
                
                # 處理類別特徵
                for cat_col in cat_features:
                    seq_values = seq_data[cat_col].values
                    if len(seq_values) < seq_len:
                        # 填充到指定長度
                        last_val = seq_values[-1] if len(seq_values) > 0 else 0
                        seq_values = np.pad(seq_values, (0, seq_len - len(seq_values)), 
                                          constant_values=last_val)
                    elif len(seq_values) > seq_len:
                        seq_values = seq_values[:seq_len]
                    
                    # 再次確保範圍正確
                    limit = embedding_limits.get(cat_col, 100)
                    seq_values = np.clip(seq_values, 0, limit-1)
                    
                    sequences[cat_col].append(seq_values.astype(np.int32))
                
                # 處理數值特徵
                num_matrix = []
                for num_col in num_features:
                    seq_values = seq_data[num_col].values
                    if len(seq_values) < seq_len:
                        last_val = seq_values[-1] if len(seq_values) > 0 else 0.0
                        seq_values = np.pad(seq_values, (0, seq_len - len(seq_values)), 
                                          constant_values=last_val)
                    elif len(seq_values) > seq_len:
                        seq_values = seq_values[:seq_len]
                    
                    num_matrix.append(seq_values.astype(np.float32))
                
                # 轉置數值矩陣: (seq_len, num_features)
                num_matrix = np.array(num_matrix).T
                sequences['num_input'].append(num_matrix)
                
                # 記錄映射
                user_mappings.extend(seq_data.index.tolist())
        
        # 轉換為 numpy arrays
        model_inputs = {}
        for key in sequences:
            if key == 'num_input':
                model_inputs[key] = np.array(sequences[key], dtype=np.float32)
            else:
                model_inputs[key] = np.array(sequences[key], dtype=np.int32)
        
        # 顯示輸入格式信息
        st.info("模型輸入格式:")
        for key, value in model_inputs.items():
            st.info(f"  {key}: {value.shape}, dtype: {value.dtype}")
        
        # 進行預測
        st.info("正在進行模型預測...")
        y_pred = model.predict(model_inputs)
        
        # === 重要：處理字典格式的預測結果 ===
        st.info(f"預測輸出類型: {type(y_pred)}")
        
        if isinstance(y_pred, dict):
            st.info(f"預測輸出鍵值: {list(y_pred.keys())}")
            for key, value in y_pred.items():
                st.info(f"輸出 {key} 形狀: {value.shape}")
            
            # 將字典轉換為列表格式以便後續處理
            pred_list = []
            
            # 常見的輸出鍵名
            possible_main_keys = ['next_action_group', 'action_group_pred', 'action_pred', 'main_output']
            possible_online_keys = ['online_conversion', 'online_prob', 'conversion_online']
            possible_o2o_keys = ['o2o_conversion', 'o2o_prob', 'conversion_o2o']
            
            # 查找主要預測結果
            main_pred = None
            for key in possible_main_keys:
                if key in y_pred:
                    main_pred = y_pred[key]
                    st.info(f"找到主要預測: {key}")
                    break
            
            if main_pred is None:
                # 如果沒找到預期的鍵，使用第一個
                main_key = list(y_pred.keys())[0]
                main_pred = y_pred[main_key]
                st.info(f"使用第一個輸出作為主要預測: {main_key}")
            
            pred_list.append(main_pred)
            
            # 查找線上轉換預測
            online_pred = None
            for key in possible_online_keys:
                if key in y_pred:
                    online_pred = y_pred[key]
                    st.info(f"找到線上轉換預測: {key}")
                    break
            
            if online_pred is not None:
                pred_list.append(online_pred)
            
            # 查找 O2O 轉換預測
            o2o_pred = None
            for key in possible_o2o_keys:
                if key in y_pred:
                    o2o_pred = y_pred[key]
                    st.info(f"找到 O2O 轉換預測: {key}")
                    break
            
            if o2o_pred is not None:
                pred_list.append(o2o_pred)
            
            # 如果沒找到轉換預測，添加剩餘的輸出
            if online_pred is None and o2o_pred is None:
                used_keys = {list(y_pred.keys())[0]}  # 已使用的主要預測鍵
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
        
        # 處理序列預測結果 - 取最後一個時間步或平均
        processed_pred = []
        for i, pred in enumerate(y_pred):
            st.info(f"處理預測輸出 {i}: 形狀 {pred.shape}")
            
            if len(pred.shape) == 3:  # (batch, time, features)
                # 取最後一個時間步
                processed_pred.append(pred[:, -1, :])
                st.info(f"  3D 輸出，取最後時間步: {pred[:, -1, :].shape}")
            elif len(pred.shape) == 2:  # (batch, features) 或 (batch, time)
                if pred.shape[1] == seq_len:  # 可能是 (batch, time) 格式
                    # 取最後一個時間步
                    processed_pred.append(pred[:, -1:])
                    st.info(f"  2D 時間序列，取最後時間步: {pred[:, -1:].shape}")
                else:  # 正常的 (batch, features)
                    processed_pred.append(pred)
                    st.info(f"  2D 特徵輸出: {pred.shape}")
            else:  # (batch,)
                processed_pred.append(pred.reshape(-1, 1))
                st.info(f"  1D 輸出，重塑為: {pred.reshape(-1, 1).shape}")
        
        y_pred = processed_pred
        st.info(f"處理後的預測結果數量: {len(y_pred)}")
        for i, pred in enumerate(y_pred):
            st.info(f"  結果 {i}: {pred.shape}")
        
        # 展開預測結果以匹配原始資料
        original_length = len(df)
        st.info(f"原始資料長度: {original_length}")
        
        if len(y_pred) >= 1:
            # 行為預測
            pred_0 = y_pred[0]
            st.info(f"主要預測形狀: {pred_0.shape}")
            
            if len(pred_0.shape) > 1 and pred_0.shape[1] > 1:
                # 多類別分類
                y_pred_action = np.argmax(pred_0, axis=1)
                y_pred_action_conf = np.max(pred_0, axis=1)
                st.info(f"多類別預測: {len(y_pred_action)} 個預測值")
            else:
                # 單一值或二元分類
                y_pred_action = pred_0.flatten()
                if np.max(y_pred_action) <= 1.0 and np.min(y_pred_action) >= 0.0:
                    # 看起來像機率值，轉換為類別
                    y_pred_action = (y_pred_action > 0.5).astype(int)
                    y_pre
