import tensorflow as tf
import numpy as np

# Tải dữ liệu Fashion MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Tiền xử lý dữ liệu
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Định nghĩa lớp custom batch normalization
class CustomBatchNorm(tf.keras.layers.Layer):
    def __init__(self, momentum=0.99, epsilon=1e-5):
        super(CustomBatchNorm, self).__init__()
        self.batch_norm = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)

    def call(self, X, training=True):
        return self.batch_norm(X, training=training)

# Xây dựng mô hình
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(6, kernel_size=5, padding='same', input_shape=(28, 28, 1)),
    CustomBatchNorm(),
    tf.keras.layers.Activation('sigmoid'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Conv2D(16, kernel_size=5, padding='same'),
    CustomBatchNorm(),
    tf.keras.layers.Activation('sigmoid'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120),
    CustomBatchNorm(),
    tf.keras.layers.Activation('sigmoid'),
    tf.keras.layers.Dense(84),
    CustomBatchNorm(),
    tf.keras.layers.Activation('sigmoid'),
    tf.keras.layers.Dense(10)
])
lr, num_epochs, batch_size = 1.0, 10, 256
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_test, y_test))

# Đánh giá mô hình trên tập kiểm tra
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')
