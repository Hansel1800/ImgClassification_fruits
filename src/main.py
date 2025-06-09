import os
import matplotlib.pyplot as plt
from utils.gpu_utils import setup_gpu, get_device_info
from data.data_loader import DataLoader
from models.model import FruitClassifier
from config.config import EPOCHS, MODEL_SAVE_PATH

def plot_training_history(history):
    """
    Vẽ biểu đồ quá trình huấn luyện
    """
    plt.figure(figsize=(12, 4))
    
    # Biểu đồ accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Biểu đồ loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Cấu hình GPU
    setup_gpu()
    get_device_info()
    
    # # Tạo thư mục models nếu chưa tồn tại
    # os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # # Tải dữ liệu
    # data_loader = DataLoader()
    
    # # Tổ chức lại dữ liệu thành 6 classes
    # data_loader.reorganize_data()
    
    # train_generator, test_generator = data_loader.load_data()
    
    # # Xây dựng và huấn luyện mô hình
    # num_classes = len(data_loader.get_class_names())
    # model = FruitClassifier(num_classes)
    # model.build_model()
    
    # # Huấn luyện
    # history = model.train(
    #     train_generator,
    #     test_generator,
    #     epochs=EPOCHS,
    #     steps_per_epoch=data_loader.get_steps_per_epoch(),
    #     validation_steps=data_loader.get_validation_steps()
    # )
    
    # # Vẽ biểu đồ
    plot_training_history(history)
    
    # # Lưu mô hình
    # model.save_model(MODEL_SAVE_PATH)

if __name__ == "__main__":
    main() 