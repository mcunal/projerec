import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense

# MNIST veri setini yükle
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Verileri normalize et
x_train, x_test = x_train / 255.0, x_test / 255.0

# Modeli tanımlama
model = Sequential([
    Flatten(input_shape=(28, 28)),  # 28x28 giriş verisini düzleştirir
    Dense(128, activation='relu'),  # 128 nöronlu gizli katman
    Dense(10, activation='softmax') # Çıkış katmanı (10 sınıf için)
])

# Modeli derleme
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğitme
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Eğitim sürecini görselleştirme (Doğruluk ve Kayıp)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend(loc='lower right')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.show()

plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend(loc='upper right')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.show()

# Modelin test verisi üzerinde değerlendirilmesi
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test Doğruluğu: {test_acc}')
print(f'Test Kaybı: {test_loss}')

# Modeli kaydetme
model.save('mnist_model.h5')
print("Model kaydedildi!")

# Modeli tekrar yükleme (yeniden kullanmak için)
loaded_model = tf.keras.models.load_model('mnist_model.h5')

# Test verisi ile modelin yeniden tahmin yapması
loaded_model_pred = loaded_model.predict(x_test)
loaded_model_pred_classes = np.argmax(loaded_model_pred, axis=1)

# Yüklenen modelin test verisi üzerindeki doğruluğunu kontrol etme
loaded_model_test_acc = np.mean(loaded_model_pred_classes == y_test)
print(f'Yüklenen Modelin Test Doğruluğu: {loaded_model_test_acc}')

# Confusion Matrix görselleştirme
cm = confusion_matrix(y_test, loaded_model_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()
