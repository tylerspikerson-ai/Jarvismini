#!/usr/bin/env python3
"""
JARVIS - Ultra Lightweight LLM Assistant
Минималистичная, быстрая и эффективная система с собственным интерфейсом
"""

import os
import sys
import json
import hashlib
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from datetime import datetime
from pathlib import Path

# Минимальные зависимости - используем только стандартную библиотеку
# Для работы с моделями можно подключить llama-cpp-python или transformers при необходимости

class JARVISConfig:
    """Конфигурация системы Джарвис"""
    
    def __init__(self, config_path="jarvis_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        """Загрузка конфигурации"""
        default_config = {
            "model_path": "",
            "model_type": "lightweight",  # lightweight, custom
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
                print(f"Ошибка загрузки конфига: {e}")
        
        return default_config
    
    def save_config(self):
        """Сохранение конфигурации"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Ошибка сохранения конфига: {e}")
            return False


class SimpleResponseEngine:
    """
    Легковесный движок ответов
    Использует паттерн-матчинг и шаблоны для минимальных требований к ресурсам
    При подключении модели - использует её
    """
    
    def __init__(self):
        self.patterns = self._load_patterns()
        self.conversation_history = []
        self.model = None
    
    def _load_patterns(self):
        """Загрузка шаблонов ответов"""
        return {
            "привет": ["Приветствую!", "Здравствуйте!", "Рад вас видеть!", "Доброго времени суток!"],
            "как дела": ["Отлично, спасибо! Готов помочь вам.", "Всё прекрасно! Чем могу быть полезен?", "Работаю в штатном режиме."],
            "кто ты": ["Я Джарвис - ваш персональный ассистент.", "Я ИИ-помощник Джарвис, создан для помощи вам."],
            "что умеешь": [
                "Я могу отвечать на вопросы, помогать с задачами и поддерживать диалог.",
                "Могу работать с текстом, анализировать информацию и помогать в решении задач."
            ],
            "спасибо": ["Всегда пожалуйста!", "Рад помочь!", "Обращайтесь в любое время!"],
            "пока": ["До свидания!", "Всего доброго!", "До встречи!"],
            "помощь": [
                "Я могу помочь вам с:\n- Ответами на вопросы\n- Анализом текста\n- Решением задач\n- Поддержкой диалога",
                "Доступные команды:\n/_help - помощь\n/clear - очистить чат\n/save - сохранить диалог"
            ],
            "время": [f"Сейчас {datetime.now().strftime('%H:%M:%S')}"],
            "дата": [f"Сегодня {datetime.now().strftime('%d.%m.%Y')}"],
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
                import random
                return random.choice(responses)
        
        # Ответ по умолчанию с эхом
        default_responses = [
            f"Интересный вопрос: '{user_input}'. Расскажите подробнее.",
            "Я понял вас. Что ещё вы хотели бы обсудить?",
            "Давайте разберёмся с этим вместе.",
            f"Вы сказали: '{user_input}'. Я готов помочь!"
        ]
        import random
        return random.choice(default_responses)
    
    def _handle_command(self, command):
        """Обработка команд"""
        commands = {
            '/help': "Доступные команды:\n/help - эта справка\n/clear - очистить историю\n/save - сохранить диалог\n/quit - выйти",
            '/clear': "CLEAR_HISTORY",
            '/save': "SAVE_DIALOG",
            '/quit': "QUIT_APP"
        }
        return commands.get(command, "Неизвестная команда. Введите /help для справки.")


class JARVISInterface:
    """Графический интерфейс Джарвиса"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("J.A.R.V.I.S. - Персональный Ассистент")
        self.root.geometry("900x700")
        self.root.minsize(600, 400)
        
        self.config = JARVISConfig()
        self.engine = SimpleResponseEngine()
        self.is_processing = False
        
        self._setup_styles()
        self._create_widgets()
        self._bind_events()
        
        # Приветственное сообщение
        self._add_message("JARVIS", "Система Джарвис готова к работе. Чем могу помочь?", "system")
    
    def _setup_styles(self):
        """Настройка стилей"""
        style = ttk.Style()
        
        # Цветовые схемы
        if self.config.config.get("theme", "dark") == "dark":
            self.colors = {
                "bg": "#1a1a2e",
                "fg": "#eaeaea",
                "user_bg": "#16213e",
                "assistant_bg": "#0f3460",
                "system_bg": "#53354a",
                "input_bg": "#1a1a2e",
                "button_bg": "#e94560",
                "button_fg": "#ffffff"
            }
            self.root.configure(bg=self.colors["bg"])
            style.theme_use('clam')
        else:
            self.colors = {
                "bg": "#f0f0f0",
                "fg": "#333333",
                "user_bg": "#ffffff",
                "assistant_bg": "#e3f2fd",
                "system_bg": "#fff3e0",
                "input_bg": "#ffffff",
                "button_bg": "#2196f3",
                "button_fg": "#ffffff"
            }
            self.root.configure(bg=self.colors["bg"])
    
    def _create_widgets(self):
        """Создание виджетов интерфейса"""
        # Верхняя панель
        top_frame = tk.Frame(self.root, bg=self.colors["bg"])
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Заголовок
        title_label = tk.Label(
            top_frame, 
            text="🤖 J.A.R.V.I.S.", 
            font=("Arial", 16, "bold"),
            bg=self.colors["bg"],
            fg=self.colors["fg"]
        )
        title_label.pack(side=tk.LEFT)
        
        # Кнопки управления
        btn_frame = tk.Frame(top_frame, bg=self.colors["bg"])
        btn_frame.pack(side=tk.RIGHT)
        
        ttk.Button(btn_frame, text="💾 Сохранить", command=self._save_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🗑️ Очистить", command=self._clear_chat).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="⚙️ Настройки", command=self._open_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="📦 ЗИП", command=self._create_zip).pack(side=tk.LEFT, padx=5)
        
        # Область чата
        chat_frame = tk.Frame(self.root, bg=self.colors["bg"])
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg=self.colors["input_bg"],
            fg=self.colors["fg"],
            insertbackground=self.colors["fg"],
            selectbackground=self.colors["button_bg"],
            selectforeground=self.colors["button_fg"]
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Настройка тегов для разных типов сообщений
        self.chat_display.tag_configure("user", background=self.colors["user_bg"], foreground=self.colors["fg"])
        self.chat_display.tag_configure("assistant", background=self.colors["assistant_bg"], foreground=self.colors["fg"])
        self.chat_display.tag_configure("system", background=self.colors["system_bg"], foreground=self.colors["fg"])
        self.chat_display.tag_configure("timestamp", foreground="#888888")
        
        # Область ввода
        input_frame = tk.Frame(self.root, bg=self.colors["bg"])
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.input_field = tk.Text(
            input_frame,
            height=3,
            font=("Consolas", 11),
            bg=self.colors["input_bg"],
            fg=self.colors["fg"],
            insertbackground=self.colors["fg"],
            wrap=tk.WORD
        )
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        send_btn = tk.Button(
            input_frame,
            text="➤ Отправить",
            command=self._send_message,
            bg=self.colors["button_bg"],
            fg=self.colors["button_fg"],
            font=("Arial", 10, "bold"),
            relief=tk.FLAT,
            padx=20,
            pady=5
        )
        send_btn.pack(side=tk.RIGHT, padx=5)
        
        # Статус бар
        self.status_var = tk.StringVar(value="Готов к работе")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            bg=self.colors["bg"],
            fg=self.colors["fg"]
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _bind_events(self):
        """Привязка событий"""
        self.input_field.bind("<Return>", self._on_enter_key)
        self.input_field.bind("<Shift-Return>", lambda e: None)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _on_enter_key(self, event):
        """Обработка нажатия Enter"""
        if not event.state & 0x1:  # Без Shift
            self._send_message()
            return "break"
        return None
    
    def _add_message(self, sender, message, msg_type="user"):
        """Добавление сообщения в чат"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, f"\n[{timestamp}] ", "timestamp")
        self.chat_display.insert(tk.END, f"{sender}: \n", msg_type)
        self.chat_display.insert(tk.END, f"{message}\n", msg_type)
        self.chat_display.see(tk.END)
        self.chat_display.configure(state='disabled')
    
    def _send_message(self):
        """Отправка сообщения"""
        if self.is_processing:
            return
        
        user_input = self.input_field.get("1.0", tk.END).strip()
        if not user_input:
            return
        
        self.input_field.delete("1.0", tk.END)
        self.is_processing = True
        self.status_var.set("Обработка...")
        
        # Добавляем сообщение пользователя
        self._add_message("Вы", user_input, "user")
        self.engine.conversation_history.append({"role": "user", "content": user_input})
        
        # Обработка в отдельном потоке
        thread = threading.Thread(target=self._process_message, args=(user_input,))
        thread.daemon = True
        thread.start()
    
    def _process_message(self, user_input):
        """Обработка сообщения в фоне"""
        try:
            response = self.engine.get_response(user_input)
            
            # Обработка специальных команд
            if response == "CLEAR_HISTORY":
                self.root.after(0, self._clear_chat_internal)
                response = "История очищена"
            elif response == "SAVE_DIALOG":
                self.root.after(0, self._save_dialog)
                response = "Диалог сохраняется..."
            elif response == "QUIT_APP":
                self.root.after(0, self._on_close)
                return
            
            self.engine.conversation_history.append({"role": "assistant", "content": response})
            self.root.after(0, lambda: self._add_message("JARVIS", response, "assistant"))
            
        except Exception as e:
            error_msg = f"Ошибка: {str(e)}"
            self.root.after(0, lambda: self._add_message("JARVIS", error_msg, "system"))
        
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.status_var.set("Готов к работе"))
    
    def _clear_chat(self):
        """Очистка чата"""
        if messagebox.askyesno("Подтверждение", "Очистить историю переписки?"):
            self._clear_chat_internal()
    
    def _clear_chat_internal(self):
        """Внутренняя очистка чата"""
        self.chat_display.configure(state='normal')
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.configure(state='disabled')
        self.engine.conversation_history.clear()
        self._add_message("JARVIS", "История очищена. Начнём заново!", "system")
    
    def _save_dialog(self):
        """Сохранение диалога"""
        if not self.engine.conversation_history:
            messagebox.showinfo("Информация", "История пуста")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"jarvis_dialog_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(self.engine.conversation_history, f, ensure_ascii=False, indent=2)
                else:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        for msg in self.engine.conversation_history:
                            f.write(f"{msg['role']}: {msg['content']}\n\n")
                
                messagebox.showinfo("Успех", f"Диалог сохранён в:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить: {e}")
    
    def _open_settings(self):
        """Открытие настроек"""
        settings_win = tk.Toplevel(self.root)
        settings_win.title("Настройки")
        settings_win.geometry("400x300")
        settings_win.configure(bg=self.colors["bg"])
        
        # Тема
        tk.Label(settings_win, text="Тема:", bg=self.colors["bg"], fg=self.colors["fg"]).pack(pady=5)
        theme_var = tk.StringVar(value=self.config.config.get("theme", "dark"))
        theme_combo = ttk.Combobox(settings_win, textvariable=theme_var, values=["dark", "light"])
        theme_combo.pack(pady=5)
        
        # Язык
        tk.Label(settings_win, text="Язык:", bg=self.colors["bg"], fg=self.colors["fg"]).pack(pady=5)
        lang_var = tk.StringVar(value=self.config.config.get("language", "ru"))
        lang_combo = ttk.Combobox(settings_win, textvariable=lang_var, values=["ru", "en"])
        lang_combo.pack(pady=5)
        
        # Авто-сохранение
        auto_save_var = tk.BooleanVar(value=self.config.config.get("auto_save", True))
        tk.Checkbutton(
            settings_win, 
            text="Автосохранение", 
            variable=auto_save_var,
            bg=self.colors["bg"],
            fg=self.colors["fg"]
        ).pack(pady=10)
        
        # Кнопки
        btn_frame = tk.Frame(settings_win, bg=self.colors["bg"])
        btn_frame.pack(pady=20)
        
        def save_settings():
            self.config.config["theme"] = theme_var.get()
            self.config.config["language"] = lang_var.get()
            self.config.config["auto_save"] = auto_save_var.get()
            self.config.save_config()
            messagebox.showinfo("Успех", "Настройки сохранены!")
            settings_win.destroy()
        
        ttk.Button(btn_frame, text="Сохранить", command=save_settings).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Отмена", command=settings_win.destroy).pack(side=tk.LEFT, padx=10)
    
    def _create_zip(self):
        """Создание ЗИП архива"""
        zip_path = filedialog.asksaveasfilename(
            defaultextension=".zip",
            filetypes=[("ZIP files", "*.zip")],
            initialfile=f"JARVIS_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        if zip_path:
            try:
                import zipfile
                
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Добавляем конфиг
                    if os.path.exists(self.config.config_path):
                        zipf.write(self.config.config_path, "config/jarvis_config.json")
                    
                    # Добавляем историю диалогов
                    if self.engine.conversation_history:
                        dialog_data = json.dumps(self.engine.conversation_history, ensure_ascii=False, indent=2)
                        zipf.writestr("dialogs/current_dialog.json", dialog_data)
                    
                    # Добавляем readme
                    readme_content = """
# J.A.R.V.I.S. - Персональный Ассистент

## Быстрый старт
1. Распакуйте архив в любую папку
2. Запустите jarvis.py
3. Начните общение!

## Требования
- Python 3.7+
- Стандартная библиотека Python

## Особенности
- Минимальные требования к железу
- Работает без интернета
- Сохранение истории диалогов
- Экспорт в ЗИП

## Команды
/help - справка
/clear - очистить чат
/save - сохранить диалог
/quit - выйти
                    """
                    zipf.writestr("README.txt", readme_content)
                    
                    # Добавляем основной скрипт
                    current_script = os.path.abspath(__file__)
                    if os.path.exists(current_script):
                        zipf.write(current_script, "jarvis.py")
                
                messagebox.showinfo("Успех", f"ЗИП архив создан:\n{zip_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось создать архив: {e}")
    
    def _on_close(self):
        """Обработка закрытия окна"""
        if self.config.config.get("auto_save", True) and self.engine.conversation_history:
            auto_save_path = f"jarvis_autosave_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            try:
                with open(auto_save_path, 'w', encoding='utf-8') as f:
                    json.dump(self.engine.conversation_history, f, ensure_ascii=False, indent=2)
            except:
                pass
        
        if messagebox.askyesno("Выход", "Завершить работу Джарвиса?"):
            self.root.destroy()


def main():
    """Точка входа"""
    root = tk.Tk()
    app = JARVISInterface(root)
    root.mainloop()


if __name__ == "__main__":
    main()
