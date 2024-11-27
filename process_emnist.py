import pandas as pd
import numpy as np

# Veriyi yükleyin
data = pd.read_csv('/Users/cetinunal/Desktop/archive/emnist-balanced-train.csv')
y = data.iloc[:, 0].values
X = data.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0  # 28x28'lik görüntüler

# Numpy dosyalarını yükleme
x_data = np.load('x.npy')
y_data = np.load('y.npy')

print(f"x_data shape: {x_data.shape}")
print(f"y_data shape: {y_data.shape}")

# Dönüştürülmüş veriyi kaydedin
np.save('X.npy', X)
np.save('y.npy', y)
