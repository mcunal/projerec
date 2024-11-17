import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense

# MNIST veri setini yükle
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Veriyi normalize et
x_train, x_test = x_train / 255.0, x_test / 255.0

# Model mimarisini tanımla
model = Sequential([
    Flatten(input_shape=(28, 28)),  # 28x28 boyutundaki resimleri düzleştir
    Dense(128, activation='relu'),  # 128 nörondan oluşan gizli katman
    Dense(10, activation='softmax')  # 10 sınıf (0-9) için çıkış katmanı
])

# Modeli derle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğit
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

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
model.save('mnist_model.h5')
print("Model 'mnist_model.h5' olarak kaydedildi.")

# Modeli yükle (Modeli kaydettikten sonra tekrar yükleme)
model = tf.keras.models.load_model('mnist_model.h5')

# Test verisinden bir örnek al
test_image = x_test[5]  # Burada 5. test verisini alıyoruz, diğer indexler de denenebilir.

# Görüntüyü 28x28 boyutunda yeniden şekillendir ve normalize et
test_image = test_image.reshape(1, 28, 28)
test_image = test_image / 255.0

# Model ile tahmin yap
prediction = model.predict(test_image)

# Tahmin edilen sınıfı (rakamı) al
predicted_class = np.argmax(prediction, axis=1)[0]

# Tahmin edilen sınıfı yazdır
print(f"Tahmin Edilen Sayı: {predicted_class}")

# Görüntüyü ve tahmin sonucunu göster
plt.imshow(x_test[5], cmap='gray')
plt.title(f'Tahmin Edilen Sayı: {predicted_class}')
plt.show()

# Test veri seti üzerinde modelin performansını değerlendir
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Doğruluğu: {test_acc * 100:.2f}%")
