import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if tf.test.is_gpu_available():
    print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
else:
    print("TensorFlow GPU not available.")
