import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Set up page layout
st.set_page_config(page_title="AI Network Guard", page_icon="🛡️", layout="wide")

st.title("🛡️ Network Intrusion Detection System Using Machine Learning")
st.markdown("Enter network packet details on the left sidebar to test live traffic evaluation.")
st.markdown("---")

@st.cache_resource
def load_and_train_system():
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level'
    ]
    
    # Read your dataset
    df = pd.read_csv('kdd_train.csv', names=columns, skiprows=1, low_memory=False)
    
    # Preprocessing
    df['label'] = df['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)
    df = df.drop(['attack_type', 'difficulty_level'], axis=1)
    
    text_columns = ['protocol_type', 'service', 'flag']
    df_encoded = pd.get_dummies(df, columns=text_columns, drop_first=True)
    
    X_raw = df_encoded.drop(['label'], axis=1)
    y_raw = df_encoded['label']
    X_raw = X_raw.astype(float)
    
    scaler_obj = StandardScaler()
    X_scaled_data = scaler_obj.fit_transform(X_raw)
    
    model_obj = DecisionTreeClassifier(max_depth=5, random_state=42)
    model_obj.fit(X_scaled_data, y_raw)
    
    return model_obj, scaler_obj, X_raw.columns

with st.spinner("🧠 Booting up the AI Guard brains..."):
    ai_guard, scaler, trained_feature_columns = load_and_train_system()

# Sidebar Layout
st.sidebar.header("🕹️ Live Packet Data Customizer")
input_protocol = st.sidebar.selectbox("Protocol Type", ["tcp", "udp", "icmp"])
input_service = st.sidebar.selectbox("Network Service Type", ["http", "private", "ftp", "smtp", "other"])
input_flag = st.sidebar.selectbox("Connection Status Flag", ["SF", "S0", "REJ", "RSTR"])

input_duration = st.sidebar.slider("Connection Duration (seconds)", 0, 60, 0)
input_src_bytes = st.sidebar.number_input("Source Bytes (Sent Data)", min_value=0, max_value=100000, value=250)
input_dst_bytes = st.sidebar.number_input("Destination Bytes (Received Data)", min_value=0, max_value=100000, value=1200)
input_count = st.sidebar.slider("Simultaneous Host Connection Count", 1, 500, 1)
input_serror_rate = st.sidebar.slider("SYN Connection Error Rate", 0.0, 1.0, 0.0, step=0.1)

# Format sample input
encoded_sample = pd.DataFrame(0.0, index=[0], columns=trained_feature_columns)
encoded_sample['duration'] = float(input_duration)
encoded_sample['src_bytes'] = float(input_src_bytes)
encoded_sample['dst_bytes'] = float(input_dst_bytes)
encoded_sample['count'] = float(input_count)
encoded_sample['serror_rate'] = float(input_serror_rate)

proto_col = f"protocol_type_{input_protocol}"
if proto_col in encoded_sample.columns:
    encoded_sample[proto_col] = 1.0

serv_col = f"service_{input_service}"
if serv_col in encoded_sample.columns:
    encoded_sample[serv_col] = 1.0

flag_col = f"flag_{input_flag}"
if flag_col in encoded_sample.columns:
    encoded_sample[flag_col] = 1.0

# Scale and predict
scaled_sample = scaler.transform(encoded_sample)
prediction = ai_guard.predict(scaled_sample)[0]
prediction_probability = ai_guard.predict_proba(scaled_sample)[0]

# UI Outputs
col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Live Input Monitor Summary")
    st.write(f"**Protocol Active:** `{input_protocol.upper()}` | **Service Route:** `{input_service}`")
    st.write(f"**Data Window:** Sent `{input_src_bytes} bytes` | Received `{input_dst_bytes} bytes`")
    st.write(f"**Burst Connection Level:** `{input_count}`")

with col2:
    st.subheader("🎯 Real-Time Verdict")
    if prediction == 0:
        st.success("✅ **SAFE TRAFFIC**")
        st.info(f"System confidence rating: **{prediction_probability[0] * 100:.1f}%**")
    else:
        st.error("🚨 **INTRUSION ATTACK DETECTED**")
        st.warning(f"System confidence rating: **{prediction_probability[1] * 100:.1f}%**")