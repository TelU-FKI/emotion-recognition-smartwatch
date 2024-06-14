import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Memuat data dari file CSV
data_1 = pd.read_csv('features/features_mo_ew2_accdata_21_10_139-1336.csv', header=None)
data_2 = pd.read_csv('features/features_mu_ew6_accdata_25_1_1912-1939.csv', header=None)
data_3 = pd.read_csv('features/features_mw_ew20_accdata_7_2_1713-1732.csv', header=None)

# Menggabungkan semua data
data = pd.concat([data_1, data_2, data_3], ignore_index=True)

# Memisahkan fitur (x_data) dan label (y_data)
x_data = data.iloc[:, :-1].values
y_data = data.iloc[:, -1].values

# Membagi data menjadi data pelatihan dan data pengujian
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Mengubah bentuk data untuk RNN
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)

# Membuat model RNN
model = Sequential()
model.add(SimpleRNN(64, input_shape=(x_train.shape[1], 1), return_sequences=True))
model.add(SimpleRNN(64))
model.add(Dense(1, activation='sigmoid'))

# Mengompilasi model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Melatih model
history = model.fit(x_train, y_train, epochs=20, validation_split=0.2, batch_size=32)

# Evaluasi model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
