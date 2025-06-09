import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from config.config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    LEARNING_RATE,
    MODEL_SAVE_PATH
)

class FruitClassifier:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = None
        self.img_height = IMG_HEIGHT
        self.img_width = IMG_WIDTH
        self.learning_rate = LEARNING_RATE
        self.model_save_path = MODEL_SAVE_PATH
    
    def build_model(self):
        """
        Xây dựng mô hình MobileNetV2
        """
        # Tải mô hình MobileNetV2 đã được huấn luyện trước
        base_model = MobileNetV2(
            input_shape=(self.img_height, self.img_width, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Đóng băng các lớp của mô hình cơ sở
        base_model.trainable = False
        
        # Xây dựng mô hình
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Biên dịch mô hình
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Đã xây dựng mô hình thành công!")
        self.model.summary()
    
    def train(self, train_generator, test_generator, epochs, steps_per_epoch, validation_steps):
        """
        Huấn luyện mô hình
        """
        print("\nBắt đầu huấn luyện...")
        
        # Huấn luyện mô hình
        history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=test_generator,
            validation_steps=validation_steps,
            verbose=1
        )
        
        print("\nHuấn luyện hoàn tất!")
        return history
    
    def save_model(self, path=None):
        """
        Lưu mô hình
        """
        if path is None:
            path = self.model_save_path
        
        self.model.save(path)
        print(f"\nĐã lưu mô hình tại: {path}")
    
    def load_model(self, path=None):
        """
        Tải mô hình đã lưu
        """
        if path is None:
            path = self.model_save_path
        
        self.model = models.load_model(path)
        print(f"\nĐã tải mô hình từ: {path}")
        self.model.summary()
    
    def predict(self, image_path):
        """
        Dự đoán ảnh mới
        """
        if self.model is None:
            raise ValueError("Chưa xây dựng mô hình. Hãy gọi build_model() trước.")
            
        img = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=(self.img_height, self.img_width)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array/255.0, axis=0)
        
        prediction = self.model.predict(img_array)
        return prediction 