import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os



# Функция. Токенизация и запись токенов в файл
def tokenize_and_save(file_path, token_file_path, tokenizer, chunk_size=50000):
    print("Загрузка BPE-токенизатора")
    token_count = 0
    with open(file_path, 'r') as f:
        if os.path.exists(token_file_path):
            tokens = list(np.load(token_file_path))
            token_count = len(tokens)
        else:
            tokens = []

        while True:
            text_chunk = f.read(chunk_size)
            if not text_chunk:
                break
            tokens_chunk = tokenizer.encode(text_chunk).ids
            tokens.extend(tokens_chunk)
            token_count += len(tokens_chunk)
            print(f"\rТокенов: {token_count}", end='', flush=True)

        print()
        np.save(token_file_path, tokens)
        return token_count



# Функция. Загрузка токенов из файла
def load_tokens(token_file_path):
    tokens = np.load(token_file_path)
    return tokens
 
       
    
# Disk
# Функция. Генерация батчей из файла
def generate_batch(file_path, batch_size, sequence_length, vocab_size, token_count):
    indices = np.random.randint(0, token_count - sequence_length - 1, batch_size)
    X = np.zeros((batch_size, sequence_length), dtype=np.int64)
    Y = np.zeros((batch_size, sequence_length), dtype=np.int64)

    with open(file_path, 'rb') as f:
            for i, idx in enumerate(indices):
                # Чтение X
                f.seek(idx * 8)  # np.int64 занимает 8 байт
                x_seq = np.fromfile(f, dtype=np.int64, count=sequence_length)

                # Чтение Y
                f.seek((idx + 1) * 8)  # смещение на 1 элемент
                y_seq = np.fromfile(f, dtype=np.int64, count=sequence_length)
                
                if np.any(x_seq >= vocab_size) or np.any(y_seq >= vocab_size):
                    print(f"Found token with index >= {vocab_size}")
                    continue

                X[i] = x_seq
                Y[i] = y_seq

    return X, Y
    
    
    
# Disk  
# Функция. Генерация датасета из файла
# TODO: Не хватает функции с yield
def generate_dataset(dir_path, batch_size, sequence_length, vocab_size, token_count):    
    dataset = tf.data.Dataset.from_generator(
        lambda: iter([generate_batch(dir_path + 'tokens.npy', batch_size, sequence_length, vocab_size, token_count)]),
        output_signature=(
            tf.TensorSpec(shape=(batch_size, sequence_length), dtype=tf.int64),
            tf.TensorSpec(shape=(batch_size, sequence_length), dtype=tf.int64),
        )
    )
    dataset = dataset.repeat().prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
    


# Класс Трансформер
class TransformerBlock_full(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock_full, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerBlock_full, self).get_config()
        config.update({
            "embed_dim": self.att._key_dim,
            "num_heads": self.att._num_heads,
            "ff_dim": self.ffn.layers[0].units,
            "rate": self.dropout1.rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)        
   
             
# Функция. Создание или загрузка модели
def create_model_full(create_nn, model_scale, num_transformer_blocks, nn_file, vocab_size):        
    if create_nn:
        #print("Создание модели")    
        embed_dim = 16 * model_scale  # размерность вложения
        num_heads = 1 * model_scale  # количество голов в механизме внимания
        ff_dim = 16 * model_scale  # размерность полносвязного слоя    

        inputs = tf.keras.Input(shape=(None,))
        embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        x = embedding_layer(inputs)
        for _ in range(num_transformer_blocks):
            transformer_block = TransformerBlock_full(embed_dim, num_heads, ff_dim)
            x = transformer_block(x, training=True)
        outputs = layers.Dense(vocab_size, activation='softmax')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        #print("Компиляция модели")
        model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    else:
        #print("Загрузка обученной модели")
        model = tf.keras.models.load_model(f"{nn_file}.keras", custom_objects={'TransformerBlock_full': TransformerBlock_full})
    
    return model
    
# Класс Трансформер
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def build(self, input_shape):
        super(TransformerBlock, self).build(input_shape)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

   
             
# Функция. Создание или загрузка модели
def create_model(create_nn, model_scale, num_transformer_blocks, nn_file, vocab_size):        
    if create_nn:
        #print("Создание модели")    
        embed_dim = 16 * model_scale  # размерность вложения
        num_heads = 1 * model_scale  # количество голов в механизме внимания
        ff_dim = 16 * model_scale * 4 # размерность полносвязного слоя    

        inputs = tf.keras.Input(shape=(None,))
        embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        x = embedding_layer(inputs)
        for _ in range(num_transformer_blocks):
            transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
            x = transformer_block(x, training=True)
        outputs = layers.Dense(vocab_size, activation='softmax')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        #print("Компиляция модели")
        model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    else:
        #print("Загрузка обученной модели")
        model = tf.keras.models.load_model(nn_file)
    
    return model    

    
    
def generate_dataset_ram_shard(tokens, batch_size, sequence_length):
    def shard_generator(index):
        def gen():
            while True:
                yield generate_batch_ram(tokens, batch_size, sequence_length)
        return gen

    dataset = tf.data.Dataset.from_generator(
        lambda: shard_generator(0)(),
        output_signature=(
            tf.TensorSpec(shape=(batch_size, sequence_length), dtype=tf.int64),
            tf.TensorSpec(shape=(batch_size, sequence_length), dtype=tf.int64),
        )
    )
    dataset = dataset.repeat().prefetch(tf.data.experimental.AUTOTUNE)
    return dataset    
    
    
    
# RAM
# Функция. Генерация батчей из массива токенов
def generate_batch_ram(tokens, batch_size, sequence_length):
    indices = np.random.randint(0, len(tokens) - sequence_length - 1, batch_size)
    X = np.zeros((batch_size, sequence_length), dtype=np.int64)
    Y = np.zeros((batch_size, sequence_length), dtype=np.int64)

    for i, idx in enumerate(indices):
        X[i] = tokens[idx:idx + sequence_length]
        Y[i] = tokens[idx + 1:idx + 1 + sequence_length]

    return X, Y

     
     
# Функция. Подгрузка, а также придерживает осн поток до готовности генератора 
def generator(tokens, batch_size, sequence_length):
    while True:
        yield generate_batch_ram(tokens, batch_size, sequence_length)   

   
   
# RAM  
# Функция. Генерация датасета из озу
def generate_dataset_ram(tokens, batch_size, sequence_length):
    dataset = tf.data.Dataset.from_generator(
        lambda: generator(tokens, batch_size, sequence_length),
        output_signature=(
            tf.TensorSpec(shape=(batch_size, sequence_length), dtype=tf.int64),
            tf.TensorSpec(shape=(batch_size, sequence_length), dtype=tf.int64),
        )
    )
    dataset = dataset.repeat().prefetch(tf.data.experimental.AUTOTUNE)
    return dataset



def get_chunk(arr, num_chunks, chunk_index):
    chunk_size = len(arr) // num_chunks
    start = chunk_size * chunk_index
    end = start + chunk_size if chunk_index < num_chunks - 1 else len(arr)
    return arr[start:end]



# Функция. Создание датасета из массива токенов
def create_dataset_from_tokens(tokens, batch_size, sequence_length, num_chunks, chunk_index):
    chunk = get_chunk(tokens, num_chunks, chunk_index)
    dataset = tf.data.Dataset.from_tensor_slices(chunk)
    dataset = dataset.shuffle(buffer_size=len(chunk) - sequence_length - 1)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(lambda x: (x[:sequence_length], x[1:1 + sequence_length]))
    dataset = dataset.repeat().prefetch(tf.data.experimental.AUTOTUNE)
    return dataset



# Функция. Софтмакс
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)



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
        predictions = model.predict(token_array, verbose=0)
        next_token_logits = predictions[0, -1, :]
        
        next_token = tail_free_sampling(next_token_logits, tail_free_threshold=tail_free_threshold)
        tokens.append(next_token)        
        
        #print(f"\r{_}", end='', flush=True)
        
        queue5.pop(0)
        queue5.append(next_token)
        if next_token == tokenizer.token_to_id("[END]"):
            break
        #if ')' in tokenizer.decode([next_token]):
        if '\n\n' in tokenizer.decode(queue5):       
            break
    
    return tokenizer.decode(tokens)

  
              
# Функция. Софтмакс с масштабированием разницы
# логитов
def softmax2(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)



# Функция. Генерация текста аргмаксом
def generate_text_argmax(model, tokenizer, input_text, max_length=50, temperature=1.0):
    tokens = tokenizer.encode(input_text).ids

    for i in range(max_length):
        token_array = np.array(tokens)[None, :]  # Add batch dimension
        logits = model.predict(token_array)  # Замените на реальный метод предсказания вашей модели

        # Предполагается, что logits имеет форму [batch_size, sequence_length, vocab_size]
        last_token_logits = logits[0, -1, :]  # Получение логитов для последнего токена
        probs = softmax2(last_token_logits / temperature)

        if True:#i < 5:
            # Случайный выбор для первых двух токенов
            next_token = np.random.choice(len(probs), p=probs)
        else:
            # Выбор токена с максимальной вероятностью для остальных
            next_token = np.argmax(probs)

        tokens.append(next_token)

        # Остановить генерацию, если достигнут токен конца последовательности
        if next_token == tokenizer.token_to_id("[END]"):
            break

    return tokenizer.decode(tokens)
    
       
             
# Функция. Логарифмический софтмакс
def  log_softmax(logits):
        return  logits  -  np.log(np.sum(np.exp(logits),  axis=-1,  keepdims=True))



#  Функция. Сэмплирование Top_p Top_k
def  sample_pk(predictions,  temperature=1.0,  top_k=0,  top_p=0.0):
        predictions  =  np.asarray(predictions).astype('float64')

        #  Применение  temperature  scaling
        predictions  =  predictions  /  temperature

        #  Применение  log_softmax
        logprobs = log_softmax(predictions)


        #  Применение  top-k  sampling
        if  top_k  >  0:
                top_k_indices  =  np.argsort(logprobs)[-top_k:]
                top_k_logprobs  =  logprobs[top_k_indices]
                top_k_probs  =  np.exp(top_k_logprobs)
                top_k_probs  /=  np.sum(top_k_probs)
                sorted_indices  =  top_k_indices
                sorted_probs  =  top_k_probs
        else:
                sorted_indices  =  np.argsort(logprobs)[::-1]
                sorted_probs  =  np.exp(logprobs[sorted_indices])

        #  Применение  top-p  sampling
        if  top_p  >  0.0:
                cumulative_probs  =  np.cumsum(sorted_probs)
                cutoff  =  np.argmax(cumulative_probs  >  top_p)
                filtered_indices  =  sorted_indices[:cutoff  +  1]
                filtered_probs  =  sorted_probs[:cutoff  +  1]
        else:
                filtered_indices  =  sorted_indices
                filtered_probs  =  sorted_probs

        #  Выбор  следующего  токена
        #filtered_probs  /=  np.sum(filtered_probs)    #  Нормализация
        #chosen_index  =  np.random.choice(filtered_indices,  p=filtered_probs)
        #return  chosen_index

        filtered_preds  =  np.zeros(len(predictions))
        filtered_preds[filtered_indices]  =  filtered_probs
        filtered_preds  /=  np.sum(filtered_preds)
        probas  =  np.random.multinomial(1,  filtered_preds,  1)
        return  np.argmax(probas)



#  Функция.  Генерация  текста через Toppk
def  generate_text_pk(model,  tokenizer,  input_text,  max_length=100,  temperature=1.0,  top_k=0,  top_p=0.0):
        tokens  =  tokenizer.encode(input_text).ids
        for  _  in  range(max_length):
                token_array  =  np.array(tokens)[None,  :]    #  Add  batch  dimension
                predictions  =  model.predict(token_array,  verbose=0)
                next_token_logits  =  predictions[0,  -1,  :]

                next_token  =  sample_pk(next_token_logits,  temperature=temperature,  top_k=top_k,  top_p=top_p)
                tokens.append(next_token)

                #  Отображаем  ответ  с  добавлением  новых  токенов  в  той  же  строке
                #print(f"\rText:  {tokenizer.decode(tokens)}",  end='',  flush=True)

                if  next_token  ==  tokenizer.token_to_id("[END]"):
                        break

        return  tokenizer.decode(tokens)
        
        
# Класс Adam, наследован от оригинала Adam      
class ScaledAdam(tf.keras.optimizers.Adam):
    def __init__(self, accumulation_steps, *args, **kwargs):
        super(ScaledAdam, self).__init__(*args, **kwargs)
        self.accumulation_steps = accumulation_steps

    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
        # Scale the learning rate by the number of accumulation steps
        scaled_lr = self.learning_rate * self.accumulation_steps
        self.learning_rate.assign(scaled_lr)
        
        # Call the base class method
        super(ScaledAdam, self).apply_gradients(grads_and_vars, name, experimental_aggregate_gradients)
        
        # Reset the learning rate to its original value
        self.learning_rate.assign(scaled_lr / self.accumulation_steps)