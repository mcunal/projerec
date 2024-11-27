import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist



# Veri setinin boyutlarını yazdır
print(f"Training data shape: {x_train.shape}")
print(f"Testing data shape: {x_test.shape}")

# İlk 5 görüntüyü görselleştir
for i in range(5):
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.show()
