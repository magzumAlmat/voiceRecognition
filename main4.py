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
        # Инициализация как раньше
        self.vosk_model = Model(model_path)
        self.recorder = None
        self.speech_queue = queue.Queue()
        
        self.tts_engine = pyttsx3.init()
        voices = self.tts_engine.getProperty('voices')
        for voice in voices:
            if 'russian' in voice.languages[0].lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
        self.tts_engine.setProperty('rate', 150)
        
        self.tts_engine.setProperty('rate', 175)  # Увеличиваем скорость речи (было 150)
        self.tts_engine.setProperty('volume', 1.0)  # Максимальная громкость
        
        # Параметры записи звука
        self.sample_rate = 16000
        self.record_duration = 2.5  # Уменьшаем длительность записи (было 5 секунд)
        self.silence_threshold = 0.01  # Порог тишины для определения конца речи
        self.min_speech_duration = 0.3  # Минимальная длительность речи
        
        # Добавляем состояние контекста
        self.context = {
            'awaiting_number': False,
            'last_command': None
        }
        
        self.number_mapping = {
            'ноль': 0, 'один': 1, 'два': 2, 'три': 3, 'четыре': 4,
            'пять': 5, 'шесть': 6, 'семь': 7, 'восемь': 8, 'девять': 9,
            'десять': 10, 'первый': 1, 'второй': 2, 'третий': 3,
            'четвертый': 4, 'пятый': 5, 'шестой': 6, 'седьмой': 7,
            'восьмой': 8, 'девятый': 9, 'десятый': 10
        }
        
        self.commands = {
            'привет': self.greet,
            'пока': self.goodbye,
            'заявка': self.request_info,
            'заявка номер': self.request_info,
        }
        
        self.requests_data = {
            5: {"status": "в работе", "date": "вчера", "time": "16:00"},
            1: {"status": "завершена", "date": "сегодня", "time": "10:00"},
            2: {"status": "отменена", "date": "вчера", "time": "14:30"},
        }

    def extract_number(self, text):
        """Извлекает номер из текста, поддерживая как цифры, так и словесные числительные"""
        # Сначала проверяем наличие цифр
        numeric_match = re.search(r'\b(\d+)\b', text)
        if numeric_match:
            return int(numeric_match.group(1))
        
        # Если цифр нет, ищем словесные числительные
        words = text.lower().split()
        for word in words:
            if word in self.number_mapping:
                return self.number_mapping[word]
        
        return None

    def process_command(self, command_text):
        """Обработка распознанной команды с поддержкой контекста"""
        if not command_text:
            return
            
        # Проверяем, ожидаем ли мы номер заявки
        if self.context['awaiting_number']:
            number = self.extract_number(command_text)
            if number is not None:
                self.context['awaiting_number'] = False
                self.request_info(number)
                return
            
        # Извлекаем номер заявки из текста
        request_number = self.extract_number(command_text)
        
        # Поиск наиболее похожей команды
        best_match = None
        best_ratio = 0
        
        for cmd in self.commands.keys():
            ratio = fuzz.ratio(command_text, cmd)
            if ratio > best_ratio and ratio > 60:
                best_ratio = ratio
                best_match = cmd
        
        if best_match:
            if "заявк" in command_text.lower():
                if request_number is not None:
                    self.request_info(request_number)
                else:
                    self.context['awaiting_number'] = True
                    self.context['last_command'] = 'заявка'
                    self.speak("Пожалуйста, укажите номер заявки")
            else:
                self.context['awaiting_number'] = False
                self.commands[best_match]()
        else:
            # Если не нашли команду, но есть номер и ожидаем номер заявки
            if self.context['awaiting_number'] and self.extract_number(command_text) is not None:
                self.request_info(self.extract_number(command_text))
                self.context['awaiting_number'] = False
            else:
                self.speak("Извините, я не поняла команду. Повторите, пожалуйста.")



    def speak(self, text):
        """Озвучивание текста"""
        print(f"Ассистент: {text}")
         # Используем более быстрые настройки для коротких фраз
        if len(text) < 50:
            self.tts_engine.setProperty('rate', 190)
        else:
            self.tts_engine.setProperty('rate', 175)
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def listen(self):
        """Запись и распознавание речи"""
        sample_rate = 16000
        duration = 5
        
        rec = KaldiRecognizer(self.vosk_model, sample_rate)
        
        print("Слушаю...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        
        audio_data = (audio.flatten() * 32767).astype(np.int16).tobytes()
        
        if rec.AcceptWaveform(audio_data):
            result = json.loads(rec.Result())
            return result.get('text', '').lower()
        return ''

    def request_info(self, request_number=None):
        """Предоставляет информацию по конкретной заявке"""
        if request_number is None:
            self.context['awaiting_number'] = True
            self.speak("Укажите номер заявки.")
            return

        request = self.requests_data.get(request_number)
        if request:
            status = request["status"]
            date = request["date"]
            time = request["time"]
            self.speak(f"Заявка номер {request_number} была принята {date} в {time} и сейчас {status}.")
        else:
            self.speak("Заявка с указанным номером не найдена.")
        
        # Сбрасываем контекст после обработки заявки
        self.context['awaiting_number'] = False



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
    model_path = "vosk-model-small-ru-0.22"
    
    try:
        assistant = VoiceAssistant(model_path)
        assistant.run()
    except Exception as e:
        print(f"Критическая ошибка: {e}")

if __name__ == "__main__":
    main()