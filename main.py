import tensorflow as tf
from tensorflow import keras

TF_ENABLE_ONEDNN_OPTS=0

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, embedding_dim):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        
        # Создание весов attention heads
        self.w_q = keras.layers.Dense(embedding_dim, use_bias=False)
        self.w_k = keras.layers.Dense(embedding_dim, use_bias=False) 
        self.w_v = keras.layers.Dense(embedding_dim, use_bias=False)
        
        # Маска для padding tokens
        self.masking = tf.keras.layers.Masking(mask_value=0.0)
        
    def call(self, inputs, mask=None):
        query = self.w_q(inputs)  
        key = self.w_k(inputs)
        value = self.w_v(inputs)
        
        # Разделение на головы_attention 
        batch_size = tf.shape(query)[0]
        seq_len = tf.shape(query)[1]
        
        query = tf.reshape(query, (batch_size, seq_len, self.num_heads, self.embedding_dim // self.num_heads))
        key = tf.reshape(key, (batch_size, seq_len, self.num_heads, self.embedding_dim // self.num_heads)) 
        value = tf.reshape(value, (batch_size, seq_len, self.num_heads, self.embedding_dim // self.num_heads))

        query = tf.transpose(query, [0, 2, 1, 3])
        key = tf.transpose(key, [0, 2, 1, 3])  
        value = tf.transpose(value, [0, 2, 1, 3])

        # Вычисление attention scores
        scale_factor = tf.math.sqrt(tf.cast(self.embedding_dim // self.num_heads, tf.float32))
        attention_scores = tf.matmul(query, key, transpose_b=True) / scale_factor
        
        if mask is not None:
            # Используем маску для исключения padding tokens
            mask = tf.expand_dims(mask, axis=1)
            mask = tf.expand_dims(mask, axis=1)
            attention_scores += (mask * -1e9)
            
        attention_weights = tf.nn.softmax(attention_scores, axis=-1) 
        attention_output = tf.matmul(attention_weights, value)

        # Объединение голов
        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, seq_len, self.embedding_dim))
        
        return attention_output

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, ff_dim):
        super(FeedForward, self).__init__()
        self.dense1 = keras.layers.Dense(ff_dim, activation='relu')
        self.dense2 = keras.layers.Dense(embedding_dim)
        
        # Нормализация Batch
        self.layer_norm = keras.layers.LayerNormalization()
        
    def call(self, inputs): 
        x = self.dense1(inputs) 
        x = self.dense2(x)
        return self.layer_norm(inputs + x)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        
        # Attention layer with multiple heads
        self.multihead_attention = MultiHeadSelfAttention(num_heads=num_heads, embedding_dim=embedding_dim)
        
        # Feed forward network
        self.feedforward = FeedForward(embedding_dim=embedding_dim, ff_dim=ff_dim)
        
        # Batch normalization for attention layer
        self.attention_norm = keras.layers.LayerNormalization()
        
        # Dropout layers for regularization
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)

    def call(self, inputs, mask=None):
        attention_output = self.multihead_attention(inputs, mask=mask)  
        attention_output = self.dropout1(attention_output)
        
        # Layer normalization after attention
        residual_connection = self.attention_norm(inputs + attention_output) 
        
        # Feedforward network 
        ff_output = self.feedforward(residual_connection)
        ff_output = self.dropout2(ff_output)
        
        # Final layer norm  
        final_output = self.attention_norm(residual_connection + ff_output)
        
        return final_output

class TransformerModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_heads, ff_dim, 
                 dropout_rate=0.1, max_seq_len=512):
        super(TransformerModel, self).__init__()
        
        # Embedding layer
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding - необхoдно для уточнения синтаксиса
        self.positional_encoding = tf.keras.layers.Embedding(max_seq_len, embedding_dim)
        
        # Transformer blocks (multiple layers)
        self.transformer_blocks = []
        for _ in range(6):  # Предположим, что 6 слоев для сложности
            self.transformer_blocks.append(
                TransformerBlock(embedding_dim=embedding_dim, 
                                num_heads=num_heads, 
                                ff_dim=ff_dim,
                                dropout_rate=dropout_rate)
            )
        
        # Output layers
        self.dense1 = keras.layers.Dense(embedding_dim, activation='relu')
        self.dropout = keras.layers.Dropout(dropout_rate)  
        self.output_layer = keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, inputs, mask=None):
        x = inputs 
        
        # Embedding
        embedding_output = self.embedding(x)
        
        # Positional encoding (объединение с embedding)
        seq_len = tf.shape(embedding_output)[1]
        positions = tf.range(start=0, limit=seq_len, dtype=tf.int32)
        positional_encodings = self.positional_encoding(positions)
        
        # Применяем позиционное кодирование к эмбеддингу
        embedding_output = embedding_output + positional_encodings
        
        # Обработка через Transformer blocks
        x = embedding_output 
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask=mask)
            
        # Среднее значение для последующей обработке (для classification tasks)  
        x = tf.reduce_mean(x, axis=1)
        
        # Dense layers
        x = self.dense1(x)
        x = self.dropout(x)
        output = self.output_layer(x)
        
        return output

# Пример использования:
# model = TransformerModel(vocab_size=30000, embedding_dim=512, 
#                        num_heads=8, ff_dim=1024, dropout_rate=0.1)

# Обучение модели
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
# model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
