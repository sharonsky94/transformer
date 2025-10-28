# ИМПОРТ ДЛЯ MODEL.FIT
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tokenizers import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
import time
import gc
from tensorflow.keras.callbacks import LearningRateScheduler
import shutil
import  os
from collections import deque
import numpy as np
from functools import partial



# ПАРАМС ДЛЯ MODEL.FIT

tf.config.optimizer.set_jit(True)

my_tokenizer_path = 'bpe_tokenizer_15k268files.json'
tokenizer = Tokenizer.from_file(my_tokenizer_path)
vocab_size = tokenizer.get_vocab_size()

def load_tokens(token_file_path):
    tokens = np.load(token_file_path)
    return tokens
#tokens = load_tokens('tokens_15kподр.npy')
tokens = load_tokens('tokens_15k268.npy')
print(f"tokens count = {len(tokens)}")

model_scale = 4 # масштаб блока трансформера
num_transformer_blocks = 6 # количество блоков трансформера
# create_nn = True
# nn_file = 'output/sl16_b250_as1/model_checkpoint_3.35.keras'
use_lr_scheduler = False
initial_lr = 0.001 #1e-7 #0.001

sequence_length = 128 # размер сэмпла
# размер батча батча для тпу крат128
accumulation_steps = 10  # количество шагов для накопления градиентов
batch_size = int(100000/sequence_length/accumulation_steps) #512 #8*50 #int(150000/sequence_length)
effective_batch_size = batch_size * accumulation_steps
state_file = 'training_state.npy'

steps_per_epoch = 200 #1000 #15 #for lr_scheduler #500 #token_count // (batch_size * sequence_length)
epochs = len(tokens) // (steps_per_epoch * batch_size * accumulation_steps) #5
save_freq='epoch'

if os.path.exists(state_file):
    create_nn = False
    state = np.load(state_file, allow_pickle=True).item()
    start_step = state.get('step', 0)
    saved_step_index = state.get('feistel_index', 0)
    start_epoch = int(saved_step_index / (steps_per_epoch * batch_size * accumulation_steps))  #state.get('epoch', 0)
    nn_file = state.get('nn_file', 0)
    print(f"🔁 Восстановление обучения: эпоха {start_epoch}, шаг {start_step}, индекс {saved_step_index}, модель {nn_file}")
else:
    create_nn = True
    start_epoch = 0
    start_step = 0
    saved_step_index = 0



# ОБУЧЕНИЕ СЕТИ GRAD_ACCUM МОД DEEPSEEK

# RAM
# Функция для генерации батчей из массива токенов
'''def generate_batch(tokens, batch_size, sequence_length):
    indices = np.random.randint(0, len(tokens) - sequence_length - 1, batch_size)
    X = np.zeros((batch_size, sequence_length), dtype=np.int64)
    Y = np.zeros((batch_size, sequence_length), dtype=np.int64)

    for i, idx in enumerate(indices):
        X[i] = tokens[idx:idx + sequence_length]
        Y[i] = tokens[idx + 1:idx + 1 + sequence_length]

    return X, Y'''
    
'''def feistel_shuffle_index(i, n, rounds=3, key=0xA5A5A5A5):
    """Детерминированный псевдослучайный порядок, позволяющий итерироваться без повторов."""
    l = i & 0xFFFF
    r = i >> 16
    for _ in range(rounds):
        l, r = r, l ^ ((hash((r, key)) & 0xFFFFFFFF) % n)
    return ((r << 16) | l) % n'''

def lcg_shuffle(i, n, a=50000009, c=0, m=None):
    if m is None:
        m = 1
        while m < n:
            m <<= 1
    x = i
    while True:
        x = (a * x + c) % m
        if x < n:
            return x

'''def data_generator():
    while True:
        yield generate_batch(tokens, batch_size, sequence_length)'''

def generator(start_index):
    step = start_index
    while True:
        #idx = feistel_shuffle_index(step, len(tokens) - sequence_length - 1, key=0xA5A5A5A5)
        idx = lcg_shuffle(step, len(tokens) - sequence_length - 1)
        x = tokens[idx : idx + sequence_length]
        y = tokens[idx + 1 : idx + 1 + sequence_length]
        yield x, y
        step += 1

# Проверка и инициализация TPU
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    if 'strategy' not in globals():
        tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print('Running on TPU')
    print("All devices: ", tf.config.list_logical_devices('TPU'))
except ValueError:
    strategy = tf.distribute.get_strategy()
    print('Running on default strategy (CPU/GPU)')
    print("All devices CPU: ", tf.config.list_logical_devices('CPU'))
    print("All devices GPU: ", tf.config.list_logical_devices('GPU'))

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, trainable=True, dtype='float32', rate=0.1):
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
            # Здесь можно добавить любую логику для инициализации весов
            super(TransformerBlock, self).build(input_shape)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
tf.keras.utils.get_custom_objects().update({'TransformerBlock': TransformerBlock})

print(f"Создать модель - {create_nn}")
if create_nn:
    with strategy.scope():
        # здесь был класс TransformerBlock
        embed_dim = 16 * model_scale  # размерность вложения
        num_heads = 1 * model_scale  # количество голов в механизме внимания
        ff_dim = 16 * model_scale * 4  # размерность полносвязного слоя

        inputs = tf.keras.Input(shape=(None,))
        embedding_layer = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim
        )
        x = embedding_layer(inputs)
        for _ in range(num_transformer_blocks):
            transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
            x = transformer_block(x, training=True)
        outputs = layers.Dense(vocab_size, activation='softmax')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy')
else:
    with strategy.scope():
        print("Загрузка обученной модели")
        model = tf.keras.models.load_model(nn_file)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy')

checkpoint_dir = f'output/sl{sequence_length}_b{batch_size}_as{accumulation_steps}'
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_filepath = os.path.join(checkpoint_dir, 'model_checkpoint_{accuracy:.2f}.keras')

checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='loss',
    verbose=1,
    save_best_only=True,
    save_freq=save_freq,
    #save_format='tf'
)
print(f'save_freq = {int(50000/sequence_length)}')

print("Обучение модели")

start_time = time.time()

'''dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(batch_size, sequence_length), dtype=tf.int64),
        tf.TensorSpec(shape=(batch_size, sequence_length), dtype=tf.int64),
    )
)
#dataset = dataset.batch(128, drop_remainder=True) # требует фикс контекста, т.е. с ним код не работает
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)'''
dataset = tf.data.Dataset.from_generator(
    partial(generator, start_index=saved_step_index),
    output_signature=(
        tf.TensorSpec(shape=(sequence_length,), dtype=tf.int64),
        tf.TensorSpec(shape=(sequence_length,), dtype=tf.int64),
    )
).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

@tf.function
def apply_accumulated_gradients(accumulated_gradients):
    def apply_fn(gradients):
        non_none_gradients = [(acc_g, var) for acc_g, var in zip(gradients, model.trainable_variables) if acc_g is not None]
        non_none_gradients = [(tf.reduce_sum(acc_g, axis=None), var) for acc_g, var in non_none_gradients]
        model.optimizer.apply_gradients(non_none_gradients)
        return [tf.zeros_like(g) if g is not None else None for g in gradients]

    accumulated_gradients = strategy.run(apply_fn, args=(accumulated_gradients,))
    return accumulated_gradients

# Исправленный GradientAccumulator для TPU
class GradientAccumulator:
    def __init__(self):
        self._gradients = []
        self._accumulation_steps = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)

    @property
    def gradients(self):
        return [g.read_value() for g in self._gradients]

    def initialize(self, model, accumulation_steps):
        self._accumulation_steps.assign(accumulation_steps)
        if not self._gradients:
            # Создание переменных вне tf.function
            with strategy.scope():
                self._gradients = [
                    tf.Variable(
                        initial_value=tf.zeros_like(var),
                        dtype=var.dtype,
                        trainable=False,
                        synchronization=tf.VariableSynchronization.ON_READ
                    ) for var in model.trainable_variables
                ]

    def reset(self):
        for g in self._gradients:
            g.assign(tf.zeros_like(g))

accumulator = GradientAccumulator()

# Упрощенный train_step
@tf.function
def train_step(iterator, accumulator):
    def step_fn(inputs, targets):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = model.compiled_loss(targets, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        #gradients = [tf.convert_to_tensor(g) if isinstance(g, tf.IndexedSlices) else g for g in gradients]
        gradients = [g / tf.cast(accumulator._accumulation_steps, g.dtype) for g in gradients]

        for i in range(len(gradients)):
            if gradients[i] is not None: # NEW
                accumulator._gradients[i].assign_add(gradients[i])

        return loss

    inputs, targets = next(iterator)
    per_replica_loss = strategy.run(step_fn, args=(inputs, targets))
    return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)

# Упрощенный apply_gradients
@tf.function
def apply_gradients(accumulator):
    def apply_fn():
        model.optimizer.apply_gradients(zip(accumulator.gradients, model.trainable_variables))
        for g in accumulator._gradients:
            g.assign(tf.zeros_like(g))

    strategy.run(apply_fn)

print(f"model_scale = {model_scale}, num_transformer_blocks = {num_transformer_blocks}")
print(f"sequence = {sequence_length}, batch = {batch_size}, accum_steps = {accumulation_steps}")
step_times = deque(maxlen=100)  # храним последние 100 шагов
start_time = time.time()
iterator = iter(dataset)

# Исправленный цикл обучения
#for epoch in range(epochs):
for epoch in range(start_epoch, epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    #iterator = iter(dataset)

    # Инициализация должна быть вне цикла шагов
    accumulator.initialize(model, accumulation_steps)

    #for step in range(steps_per_epoch):
    step_range = range(start_step, steps_per_epoch) if epoch == start_epoch else range(steps_per_epoch)

    for step in step_range:
        step_start = time.time()
        
        # Накопление градиентов
        for _ in range(accumulation_steps):
            loss = train_step(iterator, accumulator)

        # Обновление весов
        apply_gradients(accumulator)

        # Расчёт ETA, считаем trimmed mean (удаляем верхние и нижние 10%)
        step_times.append(time.time() - step_start)
        t = np.sort(step_times)
        t = t[len(t)//10 : -len(t)//10 or None]
        avg = t.mean() if len(t) else np.array(step_times).mean()
        s = epoch * steps_per_epoch + step + 1
        r = epochs * steps_per_epoch - s
        eta_m, eta_s = divmod(int(avg * r), 60)
        
        # Логирование
        print(f"\rStep {step+1}/{steps_per_epoch} Loss: {loss:.4f} | ETA: {eta_m:02d}:{eta_s:02d}", end='',flush=True)
    # Сохраняем модель каждую эпоху
    model.save(checkpoint_filepath.format(accuracy=loss.numpy()))
    
    # Сохраняем состояние после эпохи
    state = {
        'epoch': epoch + 1,
        'step': 0,
        'feistel_index': (epoch + 1) * steps_per_epoch * batch_size * accumulation_steps,
        'nn_file': checkpoint_filepath.format(accuracy=loss.numpy())
    }
    np.save(state_file, state)

if "iterator" in globals():
    del iterator
    gc.collect()
elapsed_time = time.time() - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
print(f"\nTotal elapsed time: {minutes}:{seconds}")