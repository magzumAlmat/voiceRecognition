import numpy as np
import librosa
import sounddevice as sd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 1. Функция для записи аудио с микрофона
def record_audio(duration=2, sample_rate=22050):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Ждем завершения записи
    print("Recording complete.")
    return audio.flatten()

# 2. Функция для извлечения признаков MFCC из аудиосигнала
def extract_features(audio, sample_rate=22050):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# 3. Создание данных команд для обучения
commands = ["start", "stop", "left", "right"]
X = []  # Хранилище признаков
y = []  # Метки команд

# Имитация данных для обучения (запишите несколько аудиофайлов и извлеките их признаки)
for command in commands:
    print(f"Запись команды '{command}'...")
    audio = record_audio()
    features = extract_features(audio)
    X.append(features)
    y.append(command)

X = np.array(X)
y = np.array(y)

# Преобразование меток команд в числа
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 4. Обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Прогнозирование команды
def predict_command(audio):
    features = extract_features(audio)
    prediction = model.predict([features])
    command = label_encoder.inverse_transform([int(round(prediction[0]))])[0]
    return command

# Тестирование модели с новой записью
audio = record_audio()
predicted_command = predict_command(audio)
print(f"Предсказанная команда: {predicted_command}")