import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
import tensorflow_datasets as tfds

# EMNIST veri setini yükle
(x_train, y_train), (x_test, y_test) = tfds.load('emnist', as_supervised=True, split=['train[:80%]', 'test[:20%]'], download=True)

# Veriyi normalize et
x_train = np.array([image.numpy() for image, label in x_train]) / 255.0
y_train = np.array([label.numpy() for image, label in x_train])
x_test = np.array([image.numpy() for image, label in x_test]) / 255.0
y_test = np.array([label.numpy() for image, label in x_test])

# Model mimarisini tanımla
model = Sequential([
    Flatten(input_shape=(28, 28)),  # 28x28 boyutundaki resimleri düzleştir
    Dense(512, activation='relu'),   # Daha büyük bir gizli katman
    Dropout(0.4),                   # Aşırı öğrenmeyi önlemek için Dropout
    Dense(256, activation='relu'),  # Ek bir gizli katman
    Dropout(0.3),                   # Dropout
    Dense(128, activation='relu'),  # Yine bir gizli katman
    Dense(47, activation='softmax') # EMNIST için 47 sınıf
])

# Modeli derle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğit
history = model.fit(x_train, y_train, 
                    epochs=20, 
                    batch_size=64,            # Hızlı eğitim için batch boyutu
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
model.save('emnist_model_updated.h5')
print("Model 'emnist_model_updated.h5' olarak kaydedildi.")

# Modeli test et
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Doğruluğu: {test_acc * 100:.2f}%")

# Test verisinden bir örnek al ve tahmin yap
test_image = x_test[1].reshape(1, 28, 28)
prediction = model.predict(test_image)
predicted_class = np.argmax(prediction, axis=1)[0]

# Tahmin edilen sınıfı yazdır
print(f"Tahmin Edilen Sınıf: {predicted_class}")

# Görüntüyü ve tahmin sonucunu göster
plt.imshow(x_test[5], cmap='gray')
plt.title(f'Tahmin Edilen Sınıf: {predicted_class}')
plt.show()
