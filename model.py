import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

# MNIST veri setini yükle
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Veriyi normalize et
x_train = x_train / 255.0
x_test = x_test / 255.0

# Model mimarisini tanımla
model = Sequential([
    Flatten(input_shape=(28, 28)),  # 28x28 boyutundaki resimleri düzleştir
    Dense(256, activation='relu'),  # 256 nörondan oluşan gizli katman
    Dropout(0.3),                   # Aşırı öğrenmeyi önlemek için Dropout
    Dense(128, activation='relu'),  # 128 nörondan oluşan ek gizli katman
    Dropout(0.3),                   # Aşırı öğrenmeyi önlemek için Dropout
    Dense(64, activation='relu'),   # 64 nörondan oluşan ek gizli katman
    Dense(10, activation='softmax') # 10 sınıf için çıkış katmanı (MNIST)
])

# Modeli derle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğit
history = model.fit(x_train, y_train, 
                    epochs=10, 
                    batch_size=64,           # Daha hızlı eğitim için batch boyutunu belirle
                    validation_data=(x_test, y_test))

# Eğitim sürecini görselleştir (doğruluk ve kayıp)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.show()

# Modeli kaydet
model.save('mnist_model_updated.h5')
print("Model 'mnist_model_updated.h5' olarak kaydedildi.")

# Modeli test et
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Doğruluğu: {test_acc * 100:.2f}%")

# Test verisinden bir örnek al ve tahmin yap
test_image = x_test[5].reshape(1, 28, 28)
prediction = model.predict(test_image)
predicted_class = np.argmax(prediction, axis=1)[0]

# Tahmin edilen sınıfı yazdır
print(f"Tahmin Edilen Sayı: {predicted_class}")

# Görüntüyü ve tahmin sonucunu göster
plt.imshow(x_test[5], cmap='gray')
plt.title(f'Tahmin Edilen Sayı: {predicted_class}')
plt.show()
