# -*- coding: utf-8 -*-
"""comvi_fruits_final.py

Mô hình phân loại trái cây sử dụng MobileNetV2
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Khai báo đường dẫn dataset
train_dir = "dataset/train"
test_dir = "dataset/test"

# Kiểm tra sự tồn tại của thư mục
print("Thư mục Train tồn tại:", os.path.exists(train_dir))
print("Thư mục Test tồn tại:", os.path.exists(test_dir))

# Kiểm tra các lớp trong thư mục train
print("Các lớp trong thư mục train:", os.listdir(train_dir))

# Data preprocessing
img_size = (244, 244)
batch_size = 32

# Tạo ImageDataGenerator với các phép biến đổi tăng cường dữ liệu
train_datagen = ImageDataGenerator(
    rescale=1./255,                    # Chuẩn hóa giá trị ảnh về phạm vi [0, 1]
    rotation_range=15,                 # Xoay ảnh ngẫu nhiên trong phạm vi 15 độ
    zoom_range=0.1,                    # Phóng to ảnh ngẫu nhiên trong phạm vi 10%
    width_shift_range=0.1,             # Dịch chuyển ảnh theo chiều ngang trong phạm vi 10%
    height_shift_range=0.1,            # Dịch chuyển ảnh theo chiều dọc trong phạm vi 10%
    shear_range=0.1,                   # Cắt ảnh theo một góc ngẫu nhiên
    horizontal_flip=True,              # Lật ảnh theo chiều ngang
    fill_mode='nearest'                # Phương pháp điền các giá trị bị thiếu khi biến đổi ảnh
)

test_datagen = ImageDataGenerator(
    rescale=1./255                      # Chỉ chuẩn hóa giá trị ảnh cho tập kiểm tra
)

# Đọc dữ liệu từ thư mục train và test
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Xây dựng mô hình
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(244, 244, 3))

# Đóng băng các lớp của MobileNetV2
base_model.trainable = True

# Chỉ mở khóa các lớp sau cùng của MobileNetV2
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Xây dựng mô hình mới
model = tf.keras.models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_generator.class_indices), activation='softmax')  # Số lớp tự động theo dataset
])

# Tóm tắt mô hình
model.summary()

# Compile mô hình
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Huấn luyện mô hình
EPOCHS = 80
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    validation_data=test_generator,
    validation_steps=len(test_generator)
)

# Vẽ biểu đồ accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Vẽ biểu đồ loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Lưu mô hình
model.save('fruit_classification_model.h5')
print("Mô hình đã được lưu thành công!")

# Hàm dự đoán ảnh mới
def predict_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array/255.0, axis=0)
    
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_labels = list(train_generator.class_indices.keys())
    return class_labels[class_idx], prediction[0][class_idx]

# Ví dụ sử dụng hàm dự đoán
if __name__ == "__main__":
    # Kiểm tra với một ảnh mẫu
    test_image_path = "path_to_your_test_image.jpg"  # Thay đổi đường dẫn này
    if os.path.exists(test_image_path):
        label, confidence = predict_image(test_image_path)
        print(f"\nDự đoán: {label} (Độ tin cậy: {confidence:.2f})")
    else:
        print("Không tìm thấy ảnh test!")