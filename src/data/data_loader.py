import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config.config import (
    TRAIN_DATA_DIR,
    TEST_DATA_DIR,
    IMG_HEIGHT,
    IMG_WIDTH,
    BATCH_SIZE,
    CLASS_MODE
)

class DataLoader:
    def __init__(self):
        self.train_data_dir = TRAIN_DATA_DIR
        self.test_data_dir = TEST_DATA_DIR
        self.img_height = IMG_HEIGHT
        self.img_width = IMG_WIDTH
        self.batch_size = BATCH_SIZE
        self.class_mode = CLASS_MODE
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(self.train_data_dir, exist_ok=True)
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Tạo các thư mục con cho từng lớp (6 classes)
        for class_name in ['tao_tuoi', 'chuoi_tuoi', 'cam_tuoi', 'tao_hong', 'chuoi_hong', 'cam_hong']:
            os.makedirs(os.path.join(self.train_data_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(self.test_data_dir, class_name), exist_ok=True)
        
        # Ánh xạ tên thư mục cũ sang tên mới (6 classes)
        self.class_mapping = {
            'freshapples': 'tao_tuoi',
            'freshbanana': 'chuoi_tuoi',
            'freshoranges': 'cam_tuoi',
            'rottenapples': 'tao_hong',
            'rottenbanana': 'chuoi_hong',
            'rottenoranges': 'cam_hong',
            'tao': 'tao_tuoi',
            'chuoi': 'chuoi_tuoi',
            'cam': 'cam_tuoi'
        }
    
    def load_data(self):
        """
        Tải và tiền xử lý dữ liệu
        """
        # Tạo data generator cho training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Tạo data generator cho testing
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Tải dữ liệu training
        train_generator = train_datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode=self.class_mode
        )
        
        # Tải dữ liệu testing
        test_generator = test_datagen.flow_from_directory(
            self.test_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode=self.class_mode
        )
        
        return train_generator, test_generator
    
    def get_class_names(self):
        """
        Lấy tên các lớp
        """
        class_names = sorted(os.listdir(self.train_data_dir))
        return [self.get_vietnamese_name(name) for name in class_names]
    
    def get_vietnamese_name(self, class_name):
        """
        Chuyển đổi tên lớp sang tiếng Việt
        """
        vietnamese_names = {
            'tao_tuoi': 'Táo tươi',
            'chuoi_tuoi': 'Chuối tươi',
            'cam_tuoi': 'Cam tươi',
            'tao_hong': 'Táo hỏng',
            'chuoi_hong': 'Chuối hỏng',
            'cam_hong': 'Cam hỏng'
        }
        return vietnamese_names.get(class_name, class_name)
    
    def get_steps_per_epoch(self):
        """
        Tính số bước cho mỗi epoch
        """
        return len(os.listdir(self.train_data_dir)) * len(os.listdir(os.path.join(self.train_data_dir, os.listdir(self.train_data_dir)[0]))) // self.batch_size
    
    def get_validation_steps(self):
        """
        Tính số bước cho validation
        """
        return len(os.listdir(self.test_data_dir)) * len(os.listdir(os.path.join(self.test_data_dir, os.listdir(self.test_data_dir)[0]))) // self.batch_size
    
    def reorganize_data(self):
        """
        Tổ chức lại dữ liệu thành 6 classes
        """
        import shutil
        
        # Tạo thư mục tạm thời để tránh xung đột
        temp_train_dir = os.path.join(os.path.dirname(self.train_data_dir), 'temp_train')
        temp_test_dir = os.path.join(os.path.dirname(self.test_data_dir), 'temp_test')
        
        # Di chuyển dữ liệu từ thư mục cũ sang thư mục mới
        for old_name, new_name in self.class_mapping.items():
            # Xử lý dữ liệu training
            old_train_path = os.path.join(self.train_data_dir, old_name)
            new_train_path = os.path.join(temp_train_dir, new_name)
            if os.path.exists(old_train_path):
                os.makedirs(new_train_path, exist_ok=True)
                for file in os.listdir(old_train_path):
                    shutil.copy2(
                        os.path.join(old_train_path, file),
                        os.path.join(new_train_path, file)
                    )
            
            # Xử lý dữ liệu testing
            old_test_path = os.path.join(self.test_data_dir, old_name)
            new_test_path = os.path.join(temp_test_dir, new_name)
            if os.path.exists(old_test_path):
                os.makedirs(new_test_path, exist_ok=True)
                for file in os.listdir(old_test_path):
                    shutil.copy2(
                        os.path.join(old_test_path, file),
                        os.path.join(new_test_path, file)
                    )
        
        # Xóa thư mục cũ và đổi tên thư mục tạm
        shutil.rmtree(self.train_data_dir)
        shutil.rmtree(self.test_data_dir)
        os.rename(temp_train_dir, self.train_data_dir)
        os.rename(temp_test_dir, self.test_data_dir) 