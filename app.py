import streamlit as st
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from src.config.config import MODEL_SAVE_PATH, IMG_HEIGHT, IMG_WIDTH
from src.data.data_loader import DataLoader
import unicodedata

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="HỆ THỐNG PHÂN LOẠI TÌNH TRẠNG TRÁI CÂY",
    page_icon="🍎",
    layout="wide"
)

# Load model
@st.cache_resource
def load_trained_model():
    try:
        model = load_model(MODEL_SAVE_PATH)
        return model
    except Exception as e:
        st.error(f"Không thể tải model: {str(e)}")
        return None

# Hàm tiền xử lý ảnh
def preprocess_image(img):
    img = image.load_img(img, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Đường dẫn đến thư mục dataset
SCRIPT_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(SCRIPT_DIR, 'dataset')

# Hàm đếm số lượng ảnh dựa trên loại trái cây và tình trạng
def count_images_by_fruit_type_and_status(fruit_type):
    fresh_count = 0
    rotten_count = 0
    for root_dir in ['train']:
        # Convert fruit_type to lowercase and remove Vietnamese diacritics
        fruit_type_normalized = unicodedata.normalize('NFKD', fruit_type).encode('ascii', 'ignore').decode('utf-8')
        fruit_type_lower = fruit_type_normalized.lower()

        fresh_dir = os.path.join(DATASET_PATH, root_dir, f'{fruit_type_lower}_tuoi')
        rotten_dir = os.path.join(DATASET_PATH, root_dir, f'{fruit_type_lower}_hong')
        
        if os.path.exists(fresh_dir):
            fresh_count += len(os.listdir(fresh_dir))
        if os.path.exists(rotten_dir):
            rotten_count += len(os.listdir(rotten_dir))
    return fresh_count, rotten_count

# Danh sách tên các lớp
data_loader = DataLoader()
class_names = data_loader.get_class_names()

# Đọc và áp dụng CSS
def load_css():
    with open("static/style.css", encoding="utf-8") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Kiểm tra và tạo thư mục images nếu chưa tồn tại
if not os.path.exists('images'):
    os.makedirs('images')

# Header section
st.markdown('<div class="header-container">', unsafe_allow_html=True)
col1, col2 = st.columns([0.5, 3])
with col1:
    try:
        st.image('pic_logo/logo.png', width=100)
    except:
        st.warning("Không thể tải logo. Vui lòng kiểm tra đường dẫn ảnh.")
with col2:
    st.markdown('<h1 class="header-title">HỆ THỐNG PHÂN LOẠI TÌNH TRẠNG TRÁI CÂY</h1>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Welcome message
st.markdown('<div class="chart-container" style="background-color: white; border-radius: 10px; margin: 10px auto; padding: 20px; display: flex; align-items: center; justify-content: center;"><p style="font-size: 25px; font-weight: bold; color: #2c3e50; text-align: center; margin: 0;">Chào mừng đến với hệ thống phân loại trái cây! Hãy tải lên một bức ảnh để mô hình giúp bạn phân loại tình trạng trái cây.</p></div>', unsafe_allow_html=True)

# Hiển thị hình ảnh 
st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.image('images/train_val_acc.jpg', caption='Biểu đồ 1 ', use_container_width=True)
with col2:
    st.image('images/train_val_loss.jpg', caption='Biểu đồ 2', use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar section
with st.sidebar:
    st.markdown('<div class="metric-card" style="height: 50px; background-color: white; border-radius: 10px; margin: 10px auto; display: flex; align-items: center; justify-content: left;"><p style="font-size: 20px; text-align: left; margin: 0; font-weight: bold;">Tùy chọn</p></div>', unsafe_allow_html=True)
    selected_option = st.selectbox("Số lượng ảnh để huấn luyện mô hình cho các loại trái cây", ["Chuối", "Cam", "Táo"])
    
    # Display image count for selected fruit type
    if selected_option:
        fresh_count, rotten_count = count_images_by_fruit_type_and_status(selected_option)
        st.markdown(f'<div class="metric-card" style="height: 50px; background-color: white; border-radius: 10px; margin: 10px auto; display: flex; align-items: center; justify-content: center;"><p style="font-size: 18px; text-align: center; margin: 0; font-weight: bold;">Số lượng ảnh {selected_option}:</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card" style="height: 50px; background-color: white; border-radius: 10px; margin: 5px auto; display: flex; align-items: center; justify-content: center;"><p style="font-size: 16px; text-align: center; margin: 0;">Tươi: {fresh_count} ảnh</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card" style="height: 50px; background-color: white; border-radius: 10px; margin: 5px auto; display: flex; align-items: center; justify-content: center;"><p style="font-size: 16px; text-align: center; margin: 0;">Hỏng: {rotten_count} ảnh</p></div>', unsafe_allow_html=True)

    # Thống kê
    st.markdown("---")
    st.markdown('<div class="metric-card" style="height: 50px; background-color: white; border-radius: 10px; margin: 10px auto; display: flex; align-items: center; justify-content: center;"><p style="font-size: 20px; text-align: center; margin: 0; font-weight: bold;">Dataset: 13,599 ảnh</p></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Main content area
st.markdown('<div class="main">', unsafe_allow_html=True)

# Upload section
st.markdown('<div class="upload-area">', unsafe_allow_html=True)
st.markdown("### Tải lên ảnh trái cây")
uploaded_file = st.file_uploader("Chọn hình ảnh trái cây", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

# Load model
model = load_trained_model()

# Xử lý khi người dùng đã tải lên ảnh
if uploaded_file is not None and model is not None:
    try:
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            img = st.image(uploaded_file, caption='Ảnh đã tải lên', use_container_width=True)
        with col2:
            st.write("Ảnh đã được tải lên thành công!")
            st.write("Đang phân tích hình ảnh...")
            
            # Nút phân loại
            if st.button("Phân loại trái cây", type="primary"):
                st.write("Đang thực hiện phân loại...")
                
                # Thanh tiến trình
                st.markdown('<div class="progress-container">', unsafe_allow_html=True)
                progress_bar = st.progress(0)
                for i in range(100):
                    progress_bar.progress(i + 1)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Lưu file tạm thời
                temp_path = "temp_image.jpg"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Tiền xử lý và dự đoán
                img_array = preprocess_image(temp_path)
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class])
                
                # Xóa file tạm
                os.remove(temp_path)
                
                st.success("Phân loại hoàn tất!")
                
                # Kết quả phân loại
                st.markdown("### Kết quả phân loại")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown('<div class="metric-card" style="height: 50px; background-color: white; border-radius: 10px; margin: 10px auto; display: flex; align-items: center; justify-content: center;"><p style="font-size: 20px; text-align: center; margin: 0; font-weight: bold;">Loại trái cây</p></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="metric-card" style="height: 50px; background-color: white; border-radius: 10px; margin: 10px auto; display: flex; align-items: center; justify-content: center;"><p style="font-size: 20px; text-align: center; margin: 0; font-weight: bold;">Tình trạng</p></div>', unsafe_allow_html=True)
                with col3:
                    st.markdown('<div class="metric-card" style="height: 50px; background-color: white; border-radius: 10px; margin: 10px auto; display: flex; align-items: center; justify-content: center;"><p style="font-size: 20px; text-align: center; margin: 0; font-weight: bold;">Độ tin cậy</p></div>', unsafe_allow_html=True)
                
                col1_val, col2_val, col3_val = st.columns(3)
                with col1_val:
                    loai_trai_cay_value = class_names[predicted_class].split()[0]
                    st.markdown(f'<div class="metric-card" style="height: 50px; background-color: white; border-radius: 10px; margin: 10px auto; display: flex; align-items: center; justify-content: center;"><p style="font-size: 20px; text-align: center; margin: 0; font-weight: bold;">{loai_trai_cay_value}</p></div>', unsafe_allow_html=True)
                with col2_val:
                    tinh_trang_value = 'Tốt' if 'tươi' in class_names[predicted_class] else 'Hỏng'
                    st.markdown(f'<div class="metric-card" style="height: 50px; background-color: white; border-radius: 10px; margin: 10px auto; display: flex; align-items: center; justify-content: center;"><p style="font-size: 20px; text-align: center; margin: 0; font-weight: bold;">{tinh_trang_value}</p></div>', unsafe_allow_html=True)
                with col3_val:
                    st.markdown(f'<div class="metric-card" style="height: 50px; background-color: white; border-radius: 10px; margin: 10px auto; display: flex; align-items: center; justify-content: center;"><p style="font-size: 20px; text-align: center; margin: 0; font-weight: bold;">{confidence*100:.2f}%</p></div>', unsafe_allow_html=True)
                
                # Hiển thị chi tiết dự đoán
                st.markdown("### Chi tiết dự đoán")
                for i, (class_name, conf) in enumerate(zip(class_names, predictions[0])):
                    st.progress(float(conf), text=f"{class_name}: {conf*100:.2f}%")
                
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi xử lý ảnh: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)
