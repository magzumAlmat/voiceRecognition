import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from sklearn.model_selection import train_test_split
import json
import os
from tqdm import tqdm
import pyttsx3
import datetime
import pytz
import re
from fuzzywuzzy import fuzz
import threading
import queue

class VoiceAssistant:
    def __init__(self, model_path):
        # Инициализация распознавания речи
        self.vosk_model = Model(model_path)
        self.recorder = None
        self.speech_queue = queue.Queue()
        
        # Инициализация синтеза речи
        self.tts_engine = pyttsx3.init()
        # Настройка голоса
        voices = self.tts_engine.getProperty('voices')
        for voice in voices:
            if 'russian' in voice.languages[0].lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
        self.tts_engine.setProperty('rate', 150)
        
        # Команды и ответы
        self.commands = {
            'время': self.tell_time,
            'дата': self.tell_date,
            'привет': self.greet,
            'пока': self.goodbye,
            'погода': self.tell_weather,
            'как дела': self.how_are_you,
        }

    def speak(self, text):
        """Озвучивание текста"""
        print(f"Ассистент: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def listen(self):
        """Запись и распознавание речи"""
        sample_rate = 16000
        duration = 5  # длительность записи в секундах
        
        # Инициализация распознавателя
        rec = KaldiRecognizer(self.vosk_model, sample_rate)
        
        print("Слушаю...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        
        # Преобразование аудио в формат, понятный Vosk
        audio_data = (audio.flatten() * 32767).astype(np.int16).tobytes()
        
        if rec.AcceptWaveform(audio_data):
            result = json.loads(rec.Result())
            return result.get('text', '').lower()
        return ''

    def process_command(self, command_text):
        """Обработка распознанной команды"""
        if not command_text:
            return
        
        # Поиск наиболее похожей команды
        best_match = None
        best_ratio = 0
        
        for cmd in self.commands.keys():
            ratio = fuzz.ratio(command_text, cmd)
            if ratio > best_ratio and ratio > 60:  # порог схожести
                best_ratio = ratio
                best_match = cmd
        
        if best_match:
            self.commands[best_match]()
        else:
            self.speak("Извините, я не поняла команду. Повторите, пожалуйста.")

    # Команды
    def tell_time(self):
        """Сообщает текущее время"""
        moscow_tz = pytz.timezone('Europe/Moscow')
        current_time = datetime.datetime.now(moscow_tz)
        time_str = current_time.strftime('%H:%M')
        self.speak(f"Сейчас {time_str}")

    def tell_date(self):
        """Сообщает текущую дату"""
        current_date = datetime.datetime.now()
        # Словарь для перевода месяцев на русский
        months = {
            1: 'января', 2: 'февраля', 3: 'марта', 4: 'апреля',
            5: 'мая', 6: 'июня', 7: 'июля', 8: 'августа',
            9: 'сентября', 10: 'октября', 11: 'ноября', 12: 'декабря'
        }
        date_str = f"{current_date.day} {months[current_date.month]} {current_date.year} года"
        self.speak(f"Сегодня {date_str}")

    def greet(self):
        """Приветствие"""
        current_hour = datetime.datetime.now().hour
        if 4 <= current_hour < 12:
            greeting = "Доброе утро"
        elif 12 <= current_hour < 17:
            greeting = "Добрый день"
        elif 17 <= current_hour < 23:
            greeting = "Добрый вечер"
        else:
            greeting = "Доброй ночи"
        self.speak(f"{greeting}! Чем могу помочь?")

    def goodbye(self):
        """Прощание"""
        self.speak("До свидания! Буду рада помочь вам снова.")

    def tell_weather(self):
        """Заглушка для прогноза погоды"""
        self.speak("Извините, функция прогноза погоды пока не реализована.")

    def how_are_you(self):
        """Ответ на вопрос о делах"""
        self.speak("Спасибо, у меня всё хорошо! Готова помочь вам.")

    def run(self):
        """Основной цикл работы ассистента"""
        self.speak("Привет! Я ваш голосовой помощник. Чем могу помочь?")
        
        while True:
            try:
                command_text = self.listen()
                if command_text:
                    print(f"Распознано: {command_text}")
                    
                    # Проверка на команду выхода
                    if any(word in command_text for word in ['выход', 'пока', 'до свидания']):
                        self.goodbye()
                        break
                    
                    self.process_command(command_text)
                    
            except Exception as e:
                print(f"Произошла ошибка: {e}")
                self.speak("Произошла ошибка. Попробуйте еще раз.")

def main():
    # Путь к модели Vosk
    model_path = "vosk-model-small-ru-0.22"  # Укажите правильный путь
    
    try:
        assistant = VoiceAssistant(model_path)
        assistant.run()
    except Exception as e:
        print(f"Критическая ошибка: {e}")

if __name__ == "__main__":
    main()