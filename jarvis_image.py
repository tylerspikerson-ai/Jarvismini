#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
J.A.R.V.I.S. Image Generator - Ультра-легкая модель для генерации изображений
На базе Stable Diffusion Nano / Tiny модели
Минимальные требования, работает на CPU

Автор: J.A.R.V.I.S. Team
Версия: 1.0
"""

import os
import sys
import json
import argparse
import hashlib
from datetime import datetime
from pathlib import Path

# Проверка доступности библиотек
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  Предупреждение: PIL/Pillow не установлен. Установка: pip install Pillow")

try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 PyTorch доступен. Устройство: {DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"
    print("⚠️  PyTorch не установлен. Будет использован режим эмуляции.")
    print("   Для реальной генерации: pip install torch torchvision")

try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from diffusers import AutoencoderKL, UNet2DConditionModel
    DIFFUSERS_AVAILABLE = True
    print("✅ Diffusers доступен")
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("⚠️  Diffusers не установлен.")
    print("   Для генерации изображений выполните: pip install diffusers transformers accelerate safetensors")

class UltraLightImageGenerator:
    """
    Ультра-легкий генератор изображений на базе компактных моделей SD
    Оптимизирован для работы на слабом железе и CPU
    """
    
    # Компактные модели для разных сценариев
    MODELS = {
        "nano": {
            "name": "stable-diffusion-v1-5",  # Базовая, можно дообучить или использовать tiny версии
            "revision": "fp16",
            "torch_dtype": torch.float16 if TORCH_AVAILABLE and DEVICE != "cpu" else torch.float32,
            "description": "Базовая модель (требует ~4GB RAM)"
        },
        "tiny": {
            "name": "runwayml/stable-diffusion-v1-5",
            "revision": "main",
            "torch_dtype": torch.float32,
            "description": "Оптимизированная версия для CPU (требует ~2-3GB RAM)"
        },
        "cpu": {
            "name": "compvis/stable-diffusion-v1-4",
            "revision": "main", 
            "torch_dtype": torch.float32,
            "description": "Максимально совместимая версия"
        }
    }
    
    def __init__(self, model_type="tiny", output_dir="generated_images"):
        self.model_type = model_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.pipe = None
        self.model_loaded = False
        self.config = {
            "width": 512,
            "height": 512,
            "num_inference_steps": 20,  # Меньше шагов = быстрее
            "guidance_scale": 7.5,
            "negative_prompt": "ugly, blurry, low quality, distorted, deformed",
            "seed": None
        }
        
        self.history_file = self.output_dir / "generation_history.json"
        self.history = self._load_history()
        
        if not TORCH_AVAILABLE or not DIFFUSERS_AVAILABLE:
            print("\n" + "="*60)
            print("⚠️  РЕЖИМ ЭМУЛЯЦИИ / ДЕМО")
            print("="*60)
            print("Не все зависимости установлены.")
            print("Система будет работать в демо-режиме (генерация заглушек).")
            print("\nДля полноценной работы установите:")
            print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
            print("  pip install diffusers transformers accelerate safetensors Pillow")
            print("="*60 + "\n")
    
    def _load_history(self):
        """Загрузить историю генераций"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return []
    
    def _save_history(self):
        """Сохранить историю генераций"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
    
    def load_model(self, progress_callback=None):
        """Загрузить модель с оптимизациями для слабого железа"""
        if not TORCH_AVAILABLE or not DIFFUSERS_AVAILABLE:
            print("❌ Невозможно загрузить модель: отсутствуют зависимости")
            return False
        
        if self.model_loaded:
            print("✅ Модель уже загружена")
            return True
        
        try:
            print(f"\n🔄 Загрузка модели: {self.MODELS[self.model_type]['name']}")
            print(f"   Тип: {self.model_type}, Устройство: {DEVICE}")
            
            model_config = self.MODELS[self.model_type]
            
            # Оптимизации для CPU и слабого железа
            pipeline_args = {
                "pretrained_model_name_or_path": model_config["name"],
                "torch_dtype": model_config["torch_dtype"],
            }
            
            if DEVICE == "cpu":
                # Специфичные настройки для CPU
                pipeline_args["use_safetensors"] = True
                pipeline_args["low_cpu_mem_usage"] = True
            
            if progress_callback:
                progress_callback("Загрузка модели...")
            
            self.pipe = StableDiffusionPipeline.from_pretrained(**pipeline_args)
            
            # Дополнительные оптимизации
            if DEVICE != "cpu" and torch.cuda.is_available():
                self.pipe.enable_xformers_memory_efficient_attention()
                self.pipe.enable_attention_slicing()
            else:
                self.pipe.enable_attention_slicing()
            
            # Перемещение на устройство
            self.pipe = self.pipe.to(DEVICE)
            
            # Настройка scheduler для скорости
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            self.model_loaded = True
            print("✅ Модель успешно загружена и готова к работе")
            
            if progress_callback:
                progress_callback("Модель готова!")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            print("💡 Совет: Попробуйте модель типа 'cpu' или установите недостающие пакеты")
            return False
    
    def generate(self, prompt, negative_prompt=None, width=None, height=None, 
                 steps=None, guidance=None, seed=None, save=True):
        """
        Генерация изображения по промпту
        
        Args:
            prompt: Текстовое описание изображения
            negative_prompt: Что исключить из изображения
            width: Ширина изображения (по умолчанию 512)
            height: Высота изображения (по умолчанию 512)
            steps: Количество шагов инференса (меньше = быстрее, 20-50 оптимально)
            guidance: Scale соответствия промпту (7-12 обычно)
            seed: Сид для воспроизводимости (None = случайный)
            save: Сохранять ли файл на диск
        
        Returns:
            Путь к сохраненному файлу или None в случае ошибки
        """
        if not TORCH_AVAILABLE:
            print("❌ PyTorch не доступен. Генерация невозможна.")
            return self._generate_placeholder(prompt, save)
        
        if not self.model_loaded:
            print("⚠️  Модель не загружена. Попытка автоматической загрузки...")
            if not self.load_model():
                print("❌ Не удалось загрузить модель. Используется режим заглушки.")
                return self._generate_placeholder(prompt, save)
        
        # Применение настроек по умолчанию
        width = width or self.config["width"]
        height = height or self.config["height"]
        steps = steps or self.config["num_inference_steps"]
        guidance = guidance or self.config["guidance_scale"]
        negative_prompt = negative_prompt or self.config["negative_prompt"]
        
        # Установка seed
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        
        print(f"\n🎨 Генерация изображения...")
        print(f"   Промпт: {prompt}")
        print(f"   Размер: {width}x{height}")
        print(f"   Шаги: {steps}, Guidance: {guidance}")
        print(f"   Seed: {seed}")
        
        try:
            # Генерация
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator
            ).images[0]
            
            # Сохранение
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
                filename = f"jarvis_img_{timestamp}_{prompt_hash}.png"
                filepath = self.output_dir / filename
                
                image.save(filepath)
                print(f"✅ Изображение сохранено: {filepath}")
                
                # Запись в историю
                record = {
                    "timestamp": timestamp,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "guidance": guidance,
                    "seed": seed,
                    "filename": str(filepath)
                }
                self.history.append(record)
                self._save_history()
                
                return str(filepath)
            else:
                return image
                
        except Exception as e:
            print(f"❌ Ошибка генерации: {e}")
            return self._generate_placeholder(prompt, save, error=str(e))
    
    def _generate_placeholder(self, prompt, save=True, error=None):
        """Создать изображение-заглушку когда модель недоступна"""
        if not PIL_AVAILABLE:
            print("❌ PIL не доступен. Невозможно создать даже заглушку.")
            return None
        
        print("📝 Создание изображения-заглушки (демо-режим)...")
        
        # Создание простого цветного изображения с текстом
        width = self.config["width"]
        height = self.config["height"]
        
        # Градиентный фон
        image = Image.new('RGB', (width, height), color=(30, 30, 50))
        
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(image)
        
        # Попытка загрузить шрифт
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Текст промпта (перенос длинных строк)
        max_chars = 40
        lines = []
        words = prompt.split()
        current_line = ""
        
        for word in words:
            if len(current_line) + len(word) < max_chars:
                current_line += word + " "
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        if current_line:
            lines.append(current_line.strip())
        
        # Отрисовка текста по центру
        text_height = 30
        total_text_height = len(lines) * text_height
        start_y = (height - total_text_height) // 2
        
        for i, line in enumerate(lines):
            # Получение размеров текста для центрирования
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x = (width - text_width) // 2
            y = start_y + i * text_height
            
            draw.text((x, y), line, fill=(200, 200, 255), font=font)
        
        # Добавление подписи
        signature = "J.A.R.V.I.S. Image Gen (Demo Mode)"
        if error:
            signature = f"Error: {error[:30]}"
        
        bbox = draw.textbbox((0, 0), signature, font=font)
        sig_width = bbox[2] - bbox[0]
        draw.text(((width - sig_width) // 2, height - 40), signature, fill=(150, 150, 150), font=font)
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"jarvis_demo_{timestamp}.png"
            filepath = self.output_dir / filename
            image.save(filepath)
            print(f"✅ Заглушка сохранена: {filepath}")
            
            record = {
                "timestamp": timestamp,
                "prompt": prompt,
                "filename": str(filepath),
                "demo_mode": True,
                "error": error
            }
            self.history.append(record)
            self._save_history()
            
            return str(filepath)
        
        return image
    
    def batch_generate(self, prompts, **kwargs):
        """Генерация серии изображений по списку промптов"""
        results = []
        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}] Обработка: {prompt[:50]}...")
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results
    
    def get_stats(self):
        """Получить статистику генераций"""
        total = len(self.history)
        demo_mode = sum(1 for h in self.history if h.get('demo_mode', False))
        real_gen = total - demo_mode
        
        return {
            "total_generations": total,
            "real_generations": real_gen,
            "demo_generations": demo_mode,
            "output_directory": str(self.output_dir),
            "model_loaded": self.model_loaded,
            "device": str(DEVICE),
            "torch_available": TORCH_AVAILABLE,
            "diffusers_available": DIFFUSERS_AVAILABLE
        }


class JarvisImageGUI:
    """Графический интерфейс для генератора изображений"""
    
    def __init__(self):
        if not PIL_AVAILABLE:
            print("❌ PIL не установлен. GUI режим недоступен.")
            print("   Установите: pip install Pillow")
            return
        
        try:
            import tkinter as tk
            from tkinter import ttk, messagebox, filedialog
            self.tk = tk
            self.ttk = ttk
            self.messagebox = messagebox
            self.filedialog = filedialog
        except ImportError:
            print("❌ Tkinter не доступен. GUI режим недоступен.")
            return
        
        self.generator = None
        self.current_image = None
        self.setup_gui()
    
    def setup_gui(self):
        """Настройка графического интерфейса"""
        self.root = self.tk.Tk()
        self.root.title("🎨 J.A.R.V.I.S. Image Generator")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Стиль
        style = self.ttk.Style()
        style.theme_use('clam')
        
        # Основной фрейм
        main_frame = self.ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=self.tk.BOTH, expand=True)
        
        # Заголовок
        title_label = self.ttk.Label(
            main_frame, 
            text="🎨 J.A.R.V.I.S. Image Generator",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)
        
        # Фрейм промпта
        prompt_frame = self.ttk.LabelFrame(main_frame, text="📝 Описание изображения (Prompt)", padding="10")
        prompt_frame.pack(fill=self.tk.X, pady=5)
        
        self.prompt_text = self.tk.Text(prompt_frame, height=4, wrap=self.tk.WORD)
        self.prompt_text.pack(fill=self.tk.X, padx=5, pady=5)
        self.prompt_text.insert("1.0", "красивый пейзаж с горами и озером на закате, фотореализм")
        
        # Негативный промпт
        neg_prompt_frame = self.ttk.LabelFrame(main_frame, text="❌ Исключить (Negative Prompt)", padding="10")
        neg_prompt_frame.pack(fill=self.tk.X, pady=5)
        
        self.neg_prompt_text = self.tk.Text(neg_prompt_frame, height=2, wrap=self.tk.WORD)
        self.neg_prompt_text.pack(fill=self.tk.X, padx=5, pady=5)
        self.neg_prompt_text.insert("1.0", "ugly, blurry, low quality, distorted, deformed, watermark")
        
        # Настройки
        settings_frame = self.ttk.LabelFrame(main_frame, text="⚙️ Настройки", padding="10")
        settings_frame.pack(fill=self.tk.X, pady=5)
        
        # Grid для настроек
        settings_grid = self.ttk.Frame(settings_frame)
        settings_grid.pack(fill=self.tk.X)
        
        # Ширина
        self.ttk.Label(settings_grid, text="Ширина:").grid(row=0, column=0, padx=5, pady=5)
        self.width_var = self.tk.StringVar(value="512")
        width_combo = self.ttk.Combobox(settings_grid, textvariable=self.width_var, width=10)
        width_combo['values'] = ('256', '512', '768', '1024')
        width_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # Высота
        self.ttk.Label(settings_grid, text="Высота:").grid(row=0, column=2, padx=5, pady=5)
        self.height_var = self.tk.StringVar(value="512")
        height_combo = self.ttk.Combobox(settings_grid, textvariable=self.height_var, width=10)
        height_combo['values'] = ('256', '512', '768', '1024')
        height_combo.grid(row=0, column=3, padx=5, pady=5)
        
        # Шаги
        self.ttk.Label(settings_grid, text="Шаги:").grid(row=0, column=4, padx=5, pady=5)
        self.steps_var = self.tk.StringVar(value="20")
        steps_spin = self.ttk.Spinbox(settings_grid, from_=10, to=100, textvariable=self.steps_var, width=8)
        steps_spin.grid(row=0, column=5, padx=5, pady=5)
        
        # Seed
        self.ttk.Label(settings_grid, text="Seed:").grid(row=0, column=6, padx=5, pady=5)
        self.seed_var = self.tk.StringVar(value="")
        seed_entry = self.ttk.Entry(settings_grid, textvariable=self.seed_var, width=10)
        seed_entry.grid(row=0, column=7, padx=5, pady=5)
        
        # Кнопки управления
        button_frame = self.ttk.Frame(main_frame)
        button_frame.pack(fill=self.tk.X, pady=10)
        
        self.load_btn = self.ttk.Button(button_frame, text="📦 Загрузить модель", command=self.load_model)
        self.load_btn.pack(side=self.tk.LEFT, padx=5)
        
        self.generate_btn = self.ttk.Button(button_frame, text="🎨 Генерировать", command=self.generate_image)
        self.generate_btn.pack(side=self.tk.LEFT, padx=5)
        
        self.save_btn = self.ttk.Button(button_frame, text="💾 Сохранить как...", command=self.save_image)
        self.save_btn.pack(side=self.tk.LEFT, padx=5)
        
        self.clear_btn = self.ttk.Button(button_frame, text="🗑️ Очистить", command=self.clear_fields)
        self.clear_btn.pack(side=self.tk.LEFT, padx=5)
        
        self.stats_btn = self.ttk.Button(button_frame, text="📊 Статистика", command=self.show_stats)
        self.stats_btn.pack(side=self.tk.LEFT, padx=5)
        
        # Индикатор статуса
        self.status_var = self.tk.StringVar(value="Статус: Модель не загружена")
        status_label = self.ttk.Label(main_frame, textvariable=self.status_var, foreground="gray")
        status_label.pack(pady=5)
        
        # Область предпросмотра
        preview_frame = self.ttk.LabelFrame(main_frame, text="🖼️ Предпросмотр", padding="10")
        preview_frame.pack(fill=self.tk.BOTH, expand=True, pady=5)
        
        self.preview_label = self.ttk.Label(preview_frame, text="Изображение появится здесь после генерации", 
                                           anchor=self.tk.CENTER)
        self.preview_label.pack(fill=self.tk.BOTH, expand=True)
        
        # Прогресс бар
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(fill=self.tk.X, pady=5)
        
        # История
        history_frame = self.ttk.LabelFrame(main_frame, text="📜 История генераций", padding="10")
        history_frame.pack(fill=self.tk.X, pady=5)
        
        self.history_list = self.tk.Listbox(history_frame, height=4)
        self.history_list.pack(fill=self.tk.X, padx=5, pady=5)
        
        # Обновление истории
        self.refresh_history()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def load_model(self):
        """Загрузка модели через GUI"""
        self.status_var.set("Статус: Загрузка модели...")
        self.progress.start()
        self.root.update()
        
        if self.generator is None:
            self.generator = UltraLightImageGenerator(model_type="tiny")
        
        success = self.generator.load_model(
            progress_callback=lambda msg: self.status_var.set(f"Статус: {msg}")
        )
        
        self.progress.stop()
        
        if success:
            self.status_var.set("Статус: ✅ Модель готова к работе")
            self.messagebox.showinfo("Успех", "Модель успешно загружена!\nТеперь можно генерировать изображения.")
        else:
            self.status_var.set("Статус: ⚠️ Режим эмуляции (модель не загружена)")
            self.messagebox.showwarning("Предупреждение", 
                                       "Не удалось загрузить модель.\nСистема будет работать в демо-режиме.")
    
    def generate_image(self):
        """Генерация изображения"""
        prompt = self.prompt_text.get("1.0", self.tk.END).strip()
        if not prompt:
            self.messagebox.showwarning("Внимание", "Введите описание изображения!")
            return
        
        if self.generator is None:
            self.generator = UltraLightImageGenerator()
        
        # Парсинг настроек
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            steps = int(self.steps_var.get())
            seed = int(self.seed_var.get()) if self.seed_var.get() else None
        except ValueError:
            self.messagebox.showerror("Ошибка", "Неверный формат числовых параметров!")
            return
        
        negative_prompt = self.neg_prompt_text.get("1.0", self.tk.END).strip()
        
        self.status_var.set("Статус: 🎨 Генерация изображения...")
        self.progress.start()
        self.root.update()
        
        # Генерация в отдельном потоке (чтобы не блокировать GUI)
        import threading
        def gen_thread():
            result = self.generator.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                seed=seed
            )
            self.root.after(0, self.generation_complete, result)
        
        thread = threading.Thread(target=gen_thread)
        thread.start()
    
    def generation_complete(self, filepath):
        """Обработка завершения генерации"""
        self.progress.stop()
        
        if filepath and os.path.exists(filepath):
            self.status_var.set(f"Статус: ✅ Изображение создано: {os.path.basename(filepath)}")
            
            # Загрузка и отображение
            try:
                from PIL import ImageTk
                img = Image.open(filepath)
                img.thumbnail((400, 400))
                self.current_image = ImageTk.PhotoImage(img)
                self.preview_label.config(image=self.current_image, text="")
                
                self.refresh_history()
            except Exception as e:
                self.messagebox.showerror("Ошибка", f"Не удалось отобразить изображение: {e}")
        else:
            self.status_var.set("Статус: ❌ Ошибка генерации")
            self.messagebox.showerror("Ошибка", "Не удалось создать изображение!")
    
    def save_image(self):
        """Сохранение изображения"""
        if not hasattr(self, 'current_image') or self.current_image is None:
            self.messagebox.showinfo("Инфо", "Сначала сгенерируйте изображение!")
            return
        
        filepath = self.filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if filepath:
            # Находим последний сгенерированный файл
            if self.generator and self.generator.history:
                last_record = self.generator.history[-1]
                src_file = last_record.get('filename')
                if src_file and os.path.exists(src_file):
                    import shutil
                    shutil.copy(src_file, filepath)
                    self.messagebox.showinfo("Успех", f"Изображение сохранено:\n{filepath}")
    
    def clear_fields(self):
        """Очистка полей"""
        self.prompt_text.delete("1.0", self.tk.END)
        self.neg_prompt_text.delete("1.0", self.tk.END)
        self.seed_var.set("")
        self.preview_label.config(image='', text="Изображение появится здесь после генерации")
        self.current_image = None
        self.status_var.set("Статус: Поля очищены")
    
    def refresh_history(self):
        """Обновление списка истории"""
        self.history_list.delete(0, self.tk.END)
        if self.generator:
            for record in reversed(self.generator.history[-5:]):
                display = f"{record['timestamp']}: {record['prompt'][:40]}..."
                if record.get('demo_mode'):
                    display += " (демо)"
                self.history_list.insert(self.tk.END, display)
    
    def show_stats(self):
        """Показ статистики"""
        if self.generator:
            stats = self.generator.get_stats()
            msg = (
                f"📊 Статистика генераций\n\n"
                f"Всего генераций: {stats['total_generations']}\n"
                f"Реальных: {stats['real_generations']}\n"
                f"Демо-режим: {stats['demo_generations']}\n\n"
                f"Модель загружена: {'✅ Да' if stats['model_loaded'] else '❌ Нет'}\n"
                f"Устройство: {stats['device']}\n"
                f"PyTorch: {'✅' if stats['torch_available'] else '❌'}\n"
                f"Diffusers: {'✅' if stats['diffusers_available'] else '❌'}\n\n"
                f"Папка вывода:\n{stats['output_directory']}"
            )
            self.messagebox.showinfo("Статистика", msg)
    
    def on_closing(self):
        """Обработка закрытия окна"""
        if self.messagebox.askokcancel("Выход", "Завершить работу J.A.R.V.I.S. Image Generator?"):
            self.root.destroy()
    
    def run(self):
        """Запуск GUI"""
        if hasattr(self, 'root'):
            self.root.mainloop()


def console_interface():
    """Консольный интерфейс для генератора"""
    print("\n" + "="*60)
    print("🎨 J.A.R.V.I.S. Image Generator - Консольный режим")
    print("="*60)
    
    generator = UltraLightImageGenerator()
    
    print("\n📋 Доступные команды:")
    print("  /help     - Показать справку")
    print("  /load     - Загрузить модель")
    print("  /gen      - Генерировать изображение")
    print("  /batch    - Пакетная генерация")
    print("  /stats    - Показать статистику")
    print("  /history  - Показать историю")
    print("  /settings - Настройки параметров")
    print("  /quit     - Выход")
    print("="*60)
    
    while True:
        try:
            cmd = input("\n🔹 jarvis-img> ").strip().lower()
            
            if cmd in ['/quit', '/exit', 'q']:
                print("👋 До свидания!")
                break
            
            elif cmd in ['/help', 'h']:
                print("\n📋 Команды:")
                print("  /load              - Загрузить AI модель")
                print("  /gen [промт]       - Генерировать изображение")
                print("  /batch             - Пакетная генерация по списку")
                print("  /stats             - Статистика генераций")
                print("  /history [N]       - Последние N записей истории")
                print("  /settings          - Просмотр/изменение настроек")
                print("  /clear             - Очистить консоль")
                print("  /quit              - Выход из программы")
            
            elif cmd == '/load':
                print("🔄 Загрузка модели...")
                generator.load_model()
            
            elif cmd.startswith('/gen'):
                prompt = cmd[4:].strip()
                if not prompt:
                    prompt = input("📝 Введите описание изображения: ").strip()
                
                if not prompt:
                    print("❌ Промпт не может быть пустым!")
                    continue
                
                generator.generate(prompt)
            
            elif cmd == '/batch':
                print("📦 Пакетная генерация")
                print("Вводите промпты по одному. Пустая строка завершает ввод.")
                
                prompts = []
                while True:
                    p = input(f"  Промпт #{len(prompts)+1}: ").strip()
                    if not p:
                        break
                    prompts.append(p)
                
                if prompts:
                    generator.batch_generate(prompts)
                else:
                    print("❌ Промпты не введены!")
            
            elif cmd == '/stats':
                stats = generator.get_stats()
                print("\n📊 Статистика:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            
            elif cmd.startswith('/history'):
                parts = cmd.split()
                n = int(parts[1]) if len(parts) > 1 else 5
                
                print(f"\n📜 Последние {n} генераций:")
                for record in reversed(generator.history[-n:]):
                    demo_tag = " [ДЕМО]" if record.get('demo_mode') else ""
                    print(f"  • {record['timestamp']}{demo_tag}")
                    print(f"    Промпт: {record['prompt'][:60]}")
                    print(f"    Файл: {record['filename']}")
            
            elif cmd == '/settings':
                print("\n⚙️ Текущие настройки:")
                for key, value in generator.config.items():
                    print(f"  {key}: {value}")
                
                change = input("\nИзменить настройку? (да/нет): ").strip().lower()
                if change in ['да', 'д', 'yes', 'y']:
                    key = input("Ключ настройки: ").strip()
                    if key in generator.config:
                        value = input(f"Новое значение для {key}: ").strip()
                        
                        # Попытка преобразования типа
                        if isinstance(generator.config[key], int):
                            try:
                                value = int(value)
                            except:
                                print("❌ Ошибка: ожидается целое число!")
                                continue
                        elif isinstance(generator.config[key], float):
                            try:
                                value = float(value)
                            except:
                                print("❌ Ошибка: ожидается число!")
                                continue
                        
                        generator.config[key] = value
                        print(f"✅ Настройка {key} обновлена!")
                    else:
                        print(f"❌ Настройка '{key}' не найдена!")
            
            elif cmd == '/clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                print("✅ Консоль очищена")
            
            else:
                print(f"❌ Неизвестная команда: {cmd}")
                print("   Введите /help для справки")
        
        except KeyboardInterrupt:
            print("\n\n👋 Прервано пользователем")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")


def create_zip_package(output_dir="jarvis_image_generator_zip"):
    """Создание ЗИП архива для переноса"""
    import zipfile
    import shutil
    
    print("\n📦 Создание ЗИП архива J.A.R.V.I.S. Image Generator...")
    
    zip_dir = Path(output_dir)
    zip_dir.mkdir(exist_ok=True)
    
    # Копирование основного скрипта
    current_script = Path(__file__)
    shutil.copy(current_script, zip_dir / "jarvis_image.py")
    
    # Создание requirements.txt
    requirements = """# J.A.R.V.I.S. Image Generator - Зависимости
# Минимальный набор для работы в демо-режиме:
Pillow>=9.0.0

# Для полноценной генерации изображений:
torch>=2.0.0
torchvision>=0.15.0
diffusers>=0.24.0
transformers>=4.35.0
accelerate>=0.24.0
safetensors>=0.4.0

# Установка для CPU:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# pip install diffusers transformers accelerate safetensors Pillow
"""
    
    with open(zip_dir / "requirements.txt", 'w', encoding='utf-8') as f:
        f.write(requirements)
    
    # Создание README
    readme = """# 🎨 J.A.R.V.I.S. Image Generator

Ультра-легкий генератор изображений на базе Stable Diffusion

## 🚀 Быстрый старт

### 1. Установка зависимостей

**Минимальный режим (только заглушки):**
```bash
pip install Pillow
```

**Полноценный режим (AI генерация):**
```bash
# Для CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install diffusers transformers accelerate safetensors Pillow

# Для GPU (NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate safetensors Pillow
```

### 2. Запуск

**Консольная версия:**
```bash
python jarvis_image.py --console
```

**GUI версия:**
```bash
python jarvis_image.py --gui
```

## 📋 Команды (консольный режим)

- `/load` - Загрузить AI модель
- `/gen <описание>` - Генерировать изображение
- `/batch` - Пакетная генерация
- `/stats` - Статистика
- `/history` - История генераций
- `/settings` - Настройки
- `/quit` - Выход

## ⚙️ Требования

| Режим | RAM | Место | Время генерации |
|-------|-----|-------|-----------------|
| Демо | <100 MB | <1 MB | <1 сек |
| CPU | 2-4 GB | ~4 GB | 30-120 сек |
| GPU | 4-8 GB | ~4 GB | 5-20 сек |

## 💡 Советы

1. Используйте меньшее разрешение (256x256 или 512x512) для скорости
2. Уменьшите количество шагов (15-20) для быстрой генерации
3. На слабом железе используйте режим "tiny" модели
4. Первый запуск скачивает модель (~2-4 GB)

---
Создано J.A.R.V.I.S. Team
"""
    
    with open(zip_dir / "README_IMAGE_GEN.md", 'w', encoding='utf-8') as f:
        f.write(readme)
    
    # Создание конфигурационного файла
    config = {
        "default_model": "tiny",
        "default_width": 512,
        "default_height": 512,
        "default_steps": 20,
        "default_guidance": 7.5,
        "output_directory": "generated_images",
        "version": "1.0"
    }
    
    with open(zip_dir / "config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    # Создание ZIP архива
    zip_path = Path("jarvis_image_generator.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in zip_dir.glob("*"):
            zipf.write(file, file.name)
    
    # Очистка временной директории
    shutil.rmtree(zip_dir)
    
    print(f"✅ ЗИП архив создан: {zip_path}")
    print(f"   Размер: {zip_path.stat().st_size / 1024:.2f} KB")
    print("\n📦 Для развертывания на другом ПК:")
    print(f"   1. Распакуйте {zip_path}")
    print("   2. Установите зависимости: pip install -r requirements.txt")
    print("   3. Запустите: python jarvis_image.py --gui")
    
    return str(zip_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. Image Generator")
    parser.add_argument('--console', action='store_true', help='Запустить консольную версию')
    parser.add_argument('--gui', action='store_true', help='Запустить GUI версию')
    parser.add_argument('--zip', action='store_true', help='Создать ЗИП архив для переноса')
    parser.add_argument('--prompt', type=str, help='Быстрая генерация по промпту')
    parser.add_argument('--model', type=str, default='tiny', choices=['nano', 'tiny', 'cpu'],
                       help='Тип модели: nano, tiny, cpu')
    
    args = parser.parse_args()
    
    if args.zip:
        create_zip_package()
    elif args.console or not (args.gui or args.prompt):
        console_interface()
    elif args.gui:
        gui = JarvisImageGUI()
        gui.run()
    elif args.prompt:
        generator = UltraLightImageGenerator(model_type=args.model)
        generator.generate(args.prompt)
