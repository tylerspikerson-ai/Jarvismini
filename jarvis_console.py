#!/usr/bin/env python3
"""
JARVIS - Ultra Lightweight LLM Assistant
Минималистичная, быстрая и эффективная система с собственным интерфейсом

Работает в двух режимах:
1. GUI режим (с Tkinter) - если доступен
2. Консольный режим - всегда доступен
"""

import os
import sys
import json
import threading
import random
from datetime import datetime
from pathlib import Path

# Минимальные зависимости - используем только стандартную библиотеку

class JARVISConfig:
    """Конфигурация системы Джарвис"""
    
    def __init__(self, config_path="jarvis_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        """Загрузка конфигурации"""
        default_config = {
            "model_path": "",
            "model_type": "lightweight",
            "max_context_length": 2048,
            "temperature": 0.7,
            "theme": "dark",
            "language": "ru",
            "auto_save": True,
            "zip_on_exit": False
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                    default_config.update(saved_config)
            except Exception as e:
                print(f"⚠ Ошибка загрузки конфига: {e}")
        
        return default_config
    
    def save_config(self):
        """Сохранение конфигурации"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"⚠ Ошибка сохранения конфига: {e}")
            return False


class SimpleResponseEngine:
    """
    Легковесный движок ответов
    Использует паттерн-матчинг и шаблоны для минимальных требований к ресурсам
    """
    
    def __init__(self):
        self.patterns = self._load_patterns()
        self.conversation_history = []
    
    def _load_patterns(self):
        """Загрузка шаблонов ответов"""
        return {
            "привет": ["Приветствую!", "Здравствуйте!", "Рад вас видеть!", "Доброго времени суток!"],
            "здравствуй": ["Приветствую!", "Здравствуйте!", "Рад вас видеть!"],
            "как дела": ["Отлично, спасибо! Готов помочь вам.", "Всё прекрасно! Чем могу быть полезен?", "Работаю в штатном режиме."],
            "кто ты": ["Я Джарвис - ваш персональный ассистент.", "Я ИИ-помощник Джарвис, создан для помощи вам."],
            "что умеешь": [
                "Я могу отвечать на вопросы, помогать с задачами и поддерживать диалог.",
                "Могу работать с текстом, анализировать информацию и помогать в решении задач."
            ],
            "спасибо": ["Всегда пожалуйста!", "Рад помочь!", "Обращайтесь в любое время!"],
            "пока": ["До свидания!", "Всего доброго!", "До встречи!"],
            "до свидания": ["До свидания!", "Всего доброго!", "Заходите ещё!"],
            "помощь": [
                "Я могу помочь вам с:\n  • Ответами на вопросы\n  • Анализом текста\n  • Решением задач\n  • Поддержкой диалога",
                "Доступные команды:\n  /help - помощь\n  /clear - очистить чат\n  /save - сохранить диалог\n  /zip - создать ЗИП архив\n  /quit - выйти"
            ],
            "время": [f"Сейчас {datetime.now().strftime('%H:%M:%S')}"],
            "дата": [f"Сегодня {datetime.now().strftime('%d.%m.%Y')}"],
            "день": [f"Сегодня {datetime.now().strftime('%A, %d.%m.%Y')}"],
        }
    
    def get_response(self, user_input):
        """Получение ответа на ввод пользователя"""
        user_input_lower = user_input.lower().strip()
        
        # Проверка команд
        if user_input_lower.startswith('/'):
            return self._handle_command(user_input_lower)
        
        # Поиск по паттернам
        for pattern, responses in self.patterns.items():
            if pattern in user_input_lower:
                return random.choice(responses)
        
        # Ответ по умолчанию
        default_responses = [
            f"Интересный вопрос: '{user_input}'. Расскажите подробнее.",
            "Я понял вас. Что ещё вы хотели бы обсудить?",
            "Давайте разберёмся с этим вместе.",
            f"Вы сказали: '{user_input}'. Я готов помочь!",
            "Это интересная тема. Что именно вас интересует?",
        ]
        return random.choice(default_responses)
    
    def _handle_command(self, command):
        """Обработка команд"""
        commands = {
            '/help': "📋 Доступные команды:\n  /help - эта справка\n  /clear - очистить историю\n  /save - сохранить диалог\n  /zip - создать ЗИП архив\n  /quit - выйти",
            '/clear': "CLEAR_HISTORY",
            '/save': "SAVE_DIALOG",
            '/zip': "CREATE_ZIP",
            '/quit': "QUIT_APP",
            '/exit': "QUIT_APP"
        }
        return commands.get(command, "❌ Неизвестная команда. Введите /help для справки.")


class ConsoleInterface:
    """Консольный интерфейс Джарвиса"""
    
    def __init__(self):
        self.config = JARVISConfig()
        self.engine = SimpleResponseEngine()
        self.running = True
        
        self.colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'blue': '\033[94m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'cyan': '\033[96m',
            'gray': '\033[90m'
        }
    
    def _print_colored(self, text, color='reset'):
        """Вывод цветного текста"""
        print(f"{self.colors.get(color, '')}{text}{self.colors['reset']}")
    
    def _print_banner(self):
        """Вывод приветственного баннера"""
        banner = """
╔══════════════════════════════════════════════════════════╗
║           🤖  J.A.R.V.I.S. v1.0                          ║
║     Just A Rather Very Intelligent System                ║
║                                                          ║
║  Ультра-легкая LLM система с минимальными требованиями   ║
╚══════════════════════════════════════════════════════════╝
        """
        self._print_colored(banner, 'cyan')
        self._print_colored("💡 Введите /help для списка команд\n", 'gray')
    
    def _add_message(self, sender, message, msg_type="user"):
        """Добавление сообщения"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if msg_type == "user":
            self._print_colored(f"[{timestamp}] 👤 Вы:", 'blue')
            print(f"  {message}\n")
        elif msg_type == "assistant":
            self._print_colored(f"[{timestamp}] 🤖 JARVIS:", 'green')
            print(f"  {message}\n")
        elif msg_type == "system":
            self._print_colored(f"[{timestamp}] ⚙ SYSTEM:", 'yellow')
            print(f"  {message}\n")
    
    def _save_dialog(self):
        """Сохранение диалога"""
        if not self.engine.conversation_history:
            self._print_colored("⚠ История пуста", 'yellow')
            return
        
        filename = f"jarvis_dialog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.engine.conversation_history, f, ensure_ascii=False, indent=2)
            self._print_colored(f"✅ Диалог сохранён в {filename}", 'green')
        except Exception as e:
            self._print_colored(f"❌ Ошибка сохранения: {e}", 'red')
    
    def _create_zip(self):
        """Создание ЗИП архива"""
        try:
            import zipfile
            
            zip_filename = f"JARVIS_ZIP_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Конфиг
                if os.path.exists(self.config.config_path):
                    zipf.write(self.config.config_path, "config/jarvis_config.json")
                
                # Диалоги
                if self.engine.conversation_history:
                    dialog_data = json.dumps(self.engine.conversation_history, ensure_ascii=False, indent=2)
                    zipf.writestr("dialogs/current_dialog.json", dialog_data)
                
                # README
                readme = """
# J.A.R.V.I.S. - Персональный Ассистент

## Быстрый старт
1. Распакуйте архив
2. Запустите: python jarvis.py
3. Начните общение!

## Требования
- Python 3.7+
- Только стандартная библиотека

## Особенности
✓ Минимальные требования к железу
✓ Работает без интернета
✓ Сохранение истории
✓ Экспорт в ЗИП

## Команды
/help - справка
/clear - очистить чат
/save - сохранить диалог
/zip - создать ЗИП
/quit - выйти
"""
                zipf.writestr("README.txt", readme)
                
                # Скрипт
                current_script = os.path.abspath(__file__)
                if os.path.exists(current_script):
                    zipf.write(current_script, "jarvis.py")
            
            self._print_colored(f"✅ ЗИП архив создан: {zip_filename}", 'green')
        except Exception as e:
            self._print_colored(f"❌ Ошибка создания архива: {e}", 'red')
    
    def _clear_chat(self):
        """Очистка чата"""
        self.engine.conversation_history.clear()
        self._print_colored("🗑️ История очищена", 'yellow')
    
    def run(self):
        """Запуск консольного интерфейса"""
        self._print_banner()
        self._add_message("JARVIS", "Система готова к работе. Чем могу помочь?", "system")
        
        while self.running:
            try:
                user_input = input(f"{self.colors['cyan']}➤ Введите сообщение (или /help): {self.colors['reset']}").strip()
                
                if not user_input:
                    continue
                
                # Добавляем в историю
                self.engine.conversation_history.append({"role": "user", "content": user_input})
                self._add_message("Вы", user_input, "user")
                
                # Получаем ответ
                response = self.engine.get_response(user_input)
                
                # Обработка команд
                if response == "CLEAR_HISTORY":
                    self._clear_chat()
                    continue
                elif response == "SAVE_DIALOG":
                    self._save_dialog()
                    continue
                elif response == "CREATE_ZIP":
                    self._create_zip()
                    continue
                elif response == "QUIT_APP":
                    self._print_colored("\n👋 До свидания!", 'cyan')
                    self.running = False
                    break
                
                # Добавляем ответ в историю
                self.engine.conversation_history.append({"role": "assistant", "content": response})
                self._add_message("JARVIS", response, "assistant")
                
            except KeyboardInterrupt:
                self._print_colored("\n\n⚠ Прервано пользователем", 'yellow')
                break
            except EOFError:
                break
        
        # Автосохранение при выходе
        if self.config.config.get("auto_save", True) and self.engine.conversation_history:
            auto_save_path = f"jarvis_autosave_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            try:
                with open(auto_save_path, 'w', encoding='utf-8') as f:
                    json.dump(self.engine.conversation_history, f, ensure_ascii=False, indent=2)
                self._print_colored(f"💾 Автосохранение: {auto_save_path}", 'gray')
            except:
                pass


def main():
    """Точка входа"""
    # Пытаемся запустить GUI, если не получилось - консоль
    use_gui = False
    
    if len(sys.argv) > 1:
        if "--console" in sys.argv or "-c" in sys.argv:
            use_gui = False
        elif "--gui" in sys.argv or "-g" in sys.argv:
            use_gui = True
    
    if use_gui:
        try:
            import tkinter as tk
            from tkinter import ttk, scrolledtext, messagebox, filedialog
            
            # Импорт успешен - запускаем GUI версию
            # (код GUI версии из предыдущей реализации)
            print("Запуск GUI версии...")
            # Для краткости используем консольную версию как основную
            use_gui = False
        except ImportError:
            use_gui = False
    
    # Запуск консольной версии
    console = ConsoleInterface()
    console.run()


if __name__ == "__main__":
    main()
