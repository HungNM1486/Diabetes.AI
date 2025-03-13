import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from streamlit_extras.colored_header import colored_header

# Load các mô hình
@st.cache_resource
def load_models():
    return {
        "kNN": joblib.load("models/knn_model_k19.pkl"),
        "SVM (Linear)": joblib.load("models/svm_linear_C60.pkl"),
        "SVM (RBF)": joblib.load("models/svm_rbf_C55_G1.pkl"),
        "Logistic Regression": joblib.load("models/logistic_reg_C1.pkl"),
        "Deep Learning": tf.keras.models.load_model("models/deep_learning_model.h5")
    }

models = load_models()

# ========== Cấu hình giao diện ==========
st.set_page_config(
    page_title="Diabetes.AI - Chẩn đoán thông minh",
    # page_icon="🩺",
    layout="wide"
)

# ========== Phần Header ==========
colored_header(
    label="🩺 DIABETES.AI - HỆ THỐNG CHẨN ĐOÁN TIỂU ĐƯỜNG",
    color_name="red-70",
    description="Ứng dụng AI hỗ trợ chẩn đoán bệnh tiểu đường dựa trên các chỉ số lâm sàng"
)

# ========== Phần nhập liệu ==========
with st.sidebar:
    st.header("⚙️ THÔNG TIN BỆNH NHÂN")
    with st.form("input_form"):
        # Tạo 2 cột với tỉ lệ 1:1
        col1, col2 = st.columns(2, gap="medium")
        
        with col1:
            pregnancies = st.number_input("Số lần mang thai", 0, 20, 1, 
                                       help="Số lần mang thai (nếu là nữ)")
            glucose = st.number_input("Glucose (mg/dL)", 0, 200, 100,
                                   help="Nồng độ glucose huyết tương sau 2 giờ")
            blood_pressure = st.number_input("Huyết áp (mmHg)", 0, 200, 80,
                                           help="Huyết áp tâm trương (mm Hg)")
            skin_thickness = st.number_input("Độ dày da (mm)", 0, 100, 20,
                                          help="Độ dày nếp gấp da cơ tam đầu (mm)")
            
        with col2:
            insulin = st.number_input("Insulin (μU/mL)", 0, 500, 30,
                                    help="Nồng độ insulin huyết thanh 2 giờ (μU/mL)")
            bmi = st.number_input("BMI", 0.0, 50.0, 25.0, step=0.1,
                                help="Chỉ số khối cơ thể (kg/m²)")
            diabetes_pedigree = st.number_input("Diabetes Pedigree", 0.0, 2.5, 0.5, step=0.01,
                                               help="Chức năng phả hệ tiểu đường")
            age = st.number_input("Tuổi", 0, 120, 30)
        
        # Phần chọn model và nút submit
        model_choice = st.selectbox("Mô hình AI", list(models.keys()),
                                  help="Lựa chọn thuật toán AI cho chẩn đoán")
        
        submitted = st.form_submit_button("🩺 CHẨN ĐOÁN NGAY", use_container_width=True)

# Update CSS
st.markdown(f"""
    <style>
    /* Điều chỉnh chiều rộng cột */
    [data-testid="column"] {{
        width: calc(50% - 1rem) !important;
        flex: 1 1 calc(50% - 1rem) !important;
        min-width: 300px;
    }}
    
    /* Responsive design cho mobile */
    @media (max-width: 768px) {{
        [data-testid="column"] {{
            width: 100% !important;
            flex: 1 1 100% !important;
        }}
    }}
    
    /* Điều chỉnh khoảng cách giữa các input */
    .stNumberInput, .stSelectbox {{
        margin-bottom: 1.2rem;
    }}
    
    /* Fix chiều rộng input */
    .stNumberInput > div, .stSelectbox > div {{
        width: 100% !important;
    }}
    </style>
""", unsafe_allow_html=True)

# ========== Phần kết quả ==========
if submitted:
    with st.spinner("🔄 Hệ thống đang phân tích dữ liệu..."):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                              insulin, bmi, diabetes_pedigree, age]])
        
        if model_choice == "Deep Learning":
            prediction = (models[model_choice].predict(input_data) > 0.5).astype(int)
        else:
            prediction = models[model_choice].predict(input_data)
        
        result = prediction[0]
        result_text = "DƯƠNG TÍNH - CÓ NGUY CƠ TIỂU ĐƯỜNG" if result == 1 else "ÂM TÍNH - KHÔNG CÓ NGUY CƠ"
        result_color = "positive" if result == 1 else "negative"
        
    st.markdown(f"""
        <div class="result-box {result_color}">
            <h3 style='margin:0'>KẾT QUẢ CHẨN ĐOÁN:</h3>
            <h1 style='margin:0; color:{"#e63946" if result ==1 else "#28a745"}'>▶ {result_text}</h1>
            <p style='margin:0.5rem 0; font-size:0.9rem'>Mô hình sử dụng: <strong>{model_choice}</strong></p>
        </div>
    """, unsafe_allow_html=True)

# ========== Phần giới thiệu mô hình ==========
st.markdown("---")
st.header("📊 BỘ THUẬT TOÁN AI", anchor="models")

model_info = {
    "kNN": {
        "color": "#4CC9F0",
        "icon": "🧮",
        "desc": "Thuật toán học máy truyền thống dựa trên khoảng cách dữ liệu",
        "params": "K = 19 neighbors\nMetric: Minkowski",
        "accuracy": "78.2%"
    },
    "SVM (Linear)": {
        "color": "#7209B7",
        "icon": "⚡",
        "desc": "Phân lớp bằng siêu phẳng tuyến tính",
        "params": "C = 60\nKernel: Linear",
        "accuracy": "81.5%"
    },
    "SVM (RBF)": {
        "color": "#F72585",
        "icon": "🌀",
        "desc": "Phân lớp phi tuyến với kernel Gaussian",
        "params": "C = 55\nGamma = 1",
        "accuracy": "83.1%"
    },
    "Logistic Regression": {
        "color": "#3A0CA3",
        "icon": "📈",
        "desc": "Mô hình thống kê cho phân loại nhị phân",
        "params": "C = 1\nSolver: lbfgs",
        "accuracy": "79.8%"
    },
    "Deep Learning": {
        "color": "#FF006E",
        "icon": "🧠",
        "desc": "Mạng neural tích chập đa lớp",
        "params": "4 Hidden layers\nDropout: 0.2",
        "accuracy": "85.6%"
    }
}

# Tạo layout dạng lưới responsive
cols = st.columns(3, gap="medium")

for idx, (name, info) in enumerate(model_info.items()):
    with cols[idx % 3]:  # Tạo layout 3 cột
        with st.container():
            st.markdown(f"""
                <div class="model-card" style="border-left: 4px solid {info['color']}">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
                        <span style="font-size: 24px">{info['icon']}</span>
                        <h4 style="margin:0; color:{info['color']}">{name}</h4>
                    </div>
                    <p style="font-size: 0.95rem">{info['desc']}</p>
                    <div style="background: {info['color']}10; padding: 12px; border-radius: 8px; margin: 10px 0">
                        <p style="margin:0; font-size:0.9rem; color:{info['color']}">⚙️ Tham số</p>
                        <pre style="margin:5px 0; font-size:0.85rem">{info['params']}</pre>
                        <p style="margin:0; font-size:0.9rem; color:{info['color']}">📊 Độ chính xác</p>
                        <p style="margin:5px 0; font-size:1.1rem; font-weight:500">{info['accuracy']}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# CSS bổ sung
st.markdown("""
    <style>
    .model-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: white;
        margin: 1rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    pre {
        white-space: pre-wrap;
        font-family: inherit;
    }
    @media (max-width: 768px) {
        [data-testid="column"] {
            width: 100% !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ========== Footer ==========
st.markdown("""
    <div style='text-align: center; margin-top: 4rem; color: #6c757d'>
        <hr style='margin-bottom: 1rem'>
        <p>🩺 Hệ thống chẩn đoán AI - Phiên bản 2.0</p>
        <p>© 2025 Diabetes.AI | Được phát triển bởi HungNM19PT</p>
    </div>
""", unsafe_allow_html=True)
