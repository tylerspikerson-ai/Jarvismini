#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
J.A.R.V.I.S. - НАСТОЯЩАЯ легковесная LLM с использованием трансформеров
Минимальные требования к железу, работает на CPU
Модель: Qwen2.5-0.5B-Instruct (всего 0.5 млрд параметров!)
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datetime import datetime
import threading
import queue

# === КОНФИГУРАЦИЯ ===
VERSION = "2.0.0-LLM"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"

print("Загрузка J.A.R.V.I.S. с настоящей LLM...", flush=True)

class LLMEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.loaded = False
        self.loading = False
        self.error = None
        
    def load_model(self, progress_callback=None):
        if self.loaded or self.loading:
            return
        self.loading = True
        try:
            if progress_callback:
                progress_callback("Загрузка токенизатора...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True
            )
            
            if progress_callback:
                progress_callback("Загрузка модели (может занять время)...")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            if not torch.cuda.is_available():
                self.model = self.model.to('cpu')
            
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            self.loaded = True
            if progress_callback:
                progress_callback("Model loaded!")
        except Exception as e:
            self.error = str(e)
            if progress_callback:
                progress_callback(f"Error: {e}")
        finally:
            self.loading = False
    
    def generate(self, user_message, history=None):
        if not self.loaded:
            return "Model is still loading. Please wait..."
        if self.error:
            return f"Model error: {self.error}"
        try:
            system_prompt = "You are Jarvis, a smart and helpful AI assistant. Respond briefly in Russian."
            messages = [{"role": "system", "content": system_prompt}]
            if history:
                for msg in history[-5:]:
                    messages.append(msg)
            messages.append({"role": "user", "content": user_message})
            
            result = self.pipe(
                messages,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1
            )
            
            full_text = result[0]['generated_text']
            assistant_message = full_text[-1]['content']
            return assistant_message.strip()
        except Exception as e:
            return f"Generation error: {e}"

class DialogManager:
    def __init__(self):
        self.history = []
        self.llm = LLMEngine()
        self.load_history()
    
    def load_history(self):
        history_file = "jarvis_llm_history.json"
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.history = data.get('history', [])
                    print(f"History loaded ({len(self.history)} entries)")
            except:
                pass
    
    def save_history(self):
        history_file = "jarvis_llm_history.json"
        data = {'history': self.history[-50:], 'updated': datetime.now().isoformat()}
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def add_message(self, role, text):
        self.history.append({'role': role, 'content': text, 'time': datetime.now().isoformat()})
        self.save_history()
    
    def get_response(self, user_text):
        llm_history = [{'role': msg['role'], 'content': msg['content']} for msg in self.history]
        response = self.llm.generate(user_text, llm_history)
        self.add_message("user", user_text)
        self.add_message("assistant", response)
        return response
    
    def clear(self):
        self.history = []
        self.save_history()

class Colors:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_colored(text, color=Colors.ENDC, end='\n'):
    print(f"{color}{text}{Colors.ENDC}", end=end)

def background_load(dm, status_queue):
    dm.llm.load_model(lambda msg: status_queue.put(msg))
    status_queue.put("DONE")

def main():
    print_colored("\n" + "="*60, Colors.CYAN)
    print_colored(f"  J.A.R.V.I.S. v{VERSION} - REAL LLM", Colors.BOLD + Colors.BLUE)
    print_colored(f"  Model: Qwen2.5-0.5B-Instruct (0.5B parameters)", Colors.CYAN)
    print_colored("="*60, Colors.CYAN)
    print_colored("\nInitializing neural network...", Colors.YELLOW)
    print_colored("First run may take 2-5 minutes (downloading model)", Colors.YELLOW)
    print_colored("Commands: /help, /clear, /status, /quit\n", Colors.YELLOW)
    
    dm = DialogManager()
    status_queue = queue.Queue()
    load_thread = threading.Thread(target=background_load, args=(dm, status_queue), daemon=True)
    load_thread.start()
    
    loading_chars = ['|', '/', '-', '\\']
    char_idx = 0
    
    while True:
        try:
            try:
                status = status_queue.get_nowait()
                if status == "DONE":
                    print_colored("\nNeural network ready!\n", Colors.GREEN)
                    break
                else:
                    print_colored(f"\r{loading_chars[char_idx % 4]} {status}", Colors.CYAN, end='')
                    char_idx += 1
            except queue.Empty:
                pass
            import time
            time.sleep(0.3)
        except KeyboardInterrupt:
            print_colored("\nInterrupted", Colors.RED)
            sys.exit(1)
    
    while True:
        try:
            user_input = input(f"{Colors.GREEN}You:{Colors.ENDC} ").strip()
            if not user_input:
                continue
            if user_input.startswith('/'):
                cmd = user_input.lower()[1:]
                if cmd in ['quit', 'exit']:
                    print_colored("\nGoodbye!", Colors.CYAN)
                    break
                elif cmd == 'help':
                    print_colored("\nCommands: /help, /clear, /status, /quit", Colors.BOLD)
                elif cmd == 'clear':
                    dm.clear()
                    print_colored("History cleared", Colors.YELLOW)
                elif cmd == 'status':
                    if dm.llm.loaded:
                        print_colored("Model ready", Colors.GREEN)
                    elif dm.llm.loading:
                        print_colored("Model loading...", Colors.YELLOW)
                    else:
                        print_colored("Model error", Colors.RED)
                continue
            
            print_colored("\rThinking...", Colors.CYAN, end='')
            response = dm.get_response(user_input)
            print_colored("\r" + " "*30 + "\r", end='')
            print_colored(f"{Colors.BLUE}Jarvis:{Colors.ENDC} {response}\n")
        except KeyboardInterrupt:
            print_colored("\nBye!", Colors.CYAN)
            break
        except EOFError:
            break

if __name__ == "__main__":
    main()
