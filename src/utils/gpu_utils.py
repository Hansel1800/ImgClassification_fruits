import tensorflow as tf
from config.config import GPU_MEMORY_LIMIT, ALLOW_GROWTH

def setup_gpu():
    """
    Cấu hình GPU cho TensorFlow
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, ALLOW_GROWTH)
                if GPU_MEMORY_LIMIT > 0:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=GPU_MEMORY_LIMIT)]
                    )
            print("GPU đã được cấu hình thành công")
        except RuntimeError as e:
            print(f"Lỗi khi cấu hình GPU: {e}")
    else:
        print("Không tìm thấy GPU, sẽ sử dụng CPU")

def get_device_info():
    """
    In thông tin chi tiết về thiết bị đang sử dụng
    """
    print("\n=== Thông tin thiết bị ===")
    print("TensorFlow version:", tf.__version__)
    
    # Kiểm tra GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("\nGPU được tìm thấy:")
        for gpu in gpus:
            print(f"- {gpu}")
        
        # Kiểm tra GPU có thể sử dụng được không
        try:
            with tf.device('/GPU:0'):
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])
                c = tf.matmul(a, b)
                print("\nGPU hoạt động bình thường!")
        except RuntimeError as e:
            print(f"\nLỗi khi kiểm tra GPU: {e}")
    else:
        print("\nKhông tìm thấy GPU, TensorFlow sẽ sử dụng CPU")
    
    # In thông tin về thiết bị đang sử dụng
    print("\nThiết bị đang sử dụng:", tf.config.get_visible_devices())
    
    # Kiểm tra xem TensorFlow có thể truy cập GPU không
    print("\nTensorFlow có thể truy cập GPU:", tf.test.is_built_with_cuda())
    print("TensorFlow có thể truy cập GPU (kiểm tra khác):", tf.test.is_gpu_available())
    print("========================\n") 