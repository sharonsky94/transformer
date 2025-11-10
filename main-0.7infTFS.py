import tensorflow as tf
from tokenizers import Tokenizer
import tkinter as tk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
from threading import Thread
import time
from tensorflow.keras import layers
import numpy as np

# --- Настройки ---
dir_path = ''
tokenizer = Tokenizer.from_file(dir_path + 'bpe_tokenizer_15k268files.json')
model_path = dir_path + "output/sl128_b156_as10/model_checkpoint_3.53.keras"

max_generations = 100000
temperature = 1
tail_free_threshold = 0.99
stop = True
gui_scale = 2.0
tokens_per_generation = 20

# --- Определение Transformer блока (для загрузки модели) ---
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, trainable=True, dtype='float32', rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=False)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=False)
        return self.layernorm2(out1 + ffn_output)

    def build(self, input_shape):
        super().build(input_shape)
tf.keras.utils.get_custom_objects().update({'TransformerBlock': TransformerBlock})

# --- Загрузка модели асинхронно ---
def load_model_async(path, callback):
    def load_model():
        # Загружаем без compile, чтобы не тратить время
        model = tf.keras.models.load_model(path, compile=False)
        # Принудительно компилируем для JIT ускорения (XLA)
        model.compile(jit_compile=True)
        callback(model)
    Thread(target=load_model, daemon=True).start()

# --- Окно загрузки ---
def on_model_loaded(loaded_model):
    global model
    model = loaded_model
    progress_window.destroy()
    messagebox.showinfo("Готово", "Модель успешно загружена и\nоптимизирована для быстрого\nинференса.")

# --- Генерация --- 
def process_text():
    global stop
    stop = False
    text = text_box.get("1.0", tk.END).rstrip()
    temp = float(temp_entry.get() or temperature)
    tfs = float(tfs_entry.get() or tail_free_threshold)

    for i in range(max_generations):
        if stop:
            break
        new_text = generate_text(model, tokenizer, text, max_length=tokens_per_generation, temperature=temp, tail_free_threshold=tfs)
        text_box.insert(tk.END, new_text)
        text_box.see(tk.END)
        text += new_text  # <— добавляем в контекст

    progress_window.destroy()

# --- Обработчик кнопки генерации ---
def on_button_click():
    global stop
    if not stop:
        stop = True  # остановить
    else:
        stop = False  # сбросить флаг
        Thread(target=process_text, daemon=True).start()

# Функция. Софтмакс
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# Функция. Сэмплирование TFS
def tail_free_sampling(logits, tail_free_threshold=0.9, temperature=1.0):
    # Apply temperature scaling
    logits = logits / temperature

    # Convert logits into probabilities using softmax
    probs = softmax(logits)

    # Sort probabilities in descending order
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = np.sort(probs)[::-1]

    # Calculate the first derivative
    first_derivative = np.diff(sorted_probs)

    # Calculate the second derivative
    second_derivative = np.diff(first_derivative)

    # Calculate the absolute values of the second derivative
    abs_second_derivative = np.abs(second_derivative)

    # Normalize the second derivative
    normalized_second_derivative = abs_second_derivative / np.sum(abs_second_derivative)

    # Find the smallest subset of second derivative magnitudes that surpass the threshold
    cumulative_sum = np.cumsum(normalized_second_derivative)
    cutoff_index = np.searchsorted(cumulative_sum, tail_free_threshold, side='right')

    # Keep only tokens up to the cutoff index
    if cutoff_index < len(sorted_probs):
        sorted_indices = sorted_indices[:cutoff_index + 1]
        sorted_probs = sorted_probs[:cutoff_index + 1]
    else:
        sorted_indices = sorted_indices[:1]
        sorted_probs = sorted_probs[:1]

    # Renormalize probabilities
    normalized_probs = sorted_probs / np.sum(sorted_probs)

    # Sample next token based on the normalized probabilities
    next_token = np.random.choice(sorted_indices, p=normalized_probs)

    return next_token

# Функция. Генерация текста через TFS
queue5 = [0, 0] # глоб парам для отслеживания остановы
def generate_text(model, tokenizer, input_text, max_length=100, temperature=1.0, tail_free_threshold=0.9):
    tokens = tokenizer.encode(input_text).ids
    
    for _ in range(max_length):
        token_array = np.array(tokens)[None, :]  # Add batch dimension
        predictions = model(token_array, training=False)
        next_token_logits = predictions[0, -1, :]
        
        next_token = tail_free_sampling(next_token_logits, tail_free_threshold=tail_free_threshold, temperature=temperature)
        tokens.append(next_token)        
        queue5.pop(0)
        queue5.append(next_token)
        if next_token == tokenizer.token_to_id("[END]"):
            break
        if '\n\n' in tokenizer.decode(queue5):       
            break  
    
    return tokenizer.decode(tokens[-max_length:])

# --- Интерфейс ---
root = tk.Tk()
root.title("Диалог")

text_box = ScrolledText(root, height=int(25*gui_scale), width=int(40*gui_scale))
text_box.pack(pady=30)

tk.Label(root, text="Температура").pack()
temp_entry = tk.Entry(root, width=5)
temp_entry.insert(0, str(temperature))
temp_entry.pack()

tk.Label(root, text="Tail Free").pack()
tfs_entry = tk.Entry(root, width=8)
tfs_entry.insert(0, str(tail_free_threshold))
tfs_entry.pack()

tk.Button(root, text="Отправить", command=on_button_click).pack(pady=10)

# --- Загрузка модели ---
progress_window = tk.Toplevel(root)
progress_window.title("Загрузка модели")
progress_label = tk.Label(progress_window, text="Загрузка модели...")
progress_label.pack(padx=20, pady=20)

load_model_async(model_path, on_model_loaded)

root.mainloop()
