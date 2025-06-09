import os

# Đường dẫn dữ liệu
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'train')
TEST_DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'test')

# Cấu hình ảnh
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
CLASS_MODE = 'categorical'

# Cấu hình GPU
GPU_MEMORY_LIMIT = 4096  # Giới hạn bộ nhớ GPU (MB)
ALLOW_GROWTH = True      # Cho phép tăng bộ nhớ GPU

# Cấu hình huấn luyện
EPOCHS = 50
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models_trained', 'fruit_classifier.h5')

# Model Configuration
BASE_MODEL = "MobileNetV2"
FINE_TUNE_AT = 100
DENSE_LAYER_SIZE = 512
DROPOUT_RATE = 0.5

# Training Configuration
LEARNING_RATE = 0.0001
STEPS_PER_EPOCH = None  # Will be set automatically
VALIDATION_STEPS = None  # Will be set automatically

# Data Augmentation Configuration
TRAIN_AUGMENTATION = {
    'rescale': 1./255,
    'rotation_range': 15,
    'zoom_range': 0.1,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'shear_range': 0.1,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

TEST_AUGMENTATION = {
    'rescale': 1./255
} 