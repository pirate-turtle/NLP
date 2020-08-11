import tensorflow as tf

INPUT_SIZE = (20, 1)

input = tf.placeholder(tf.float32, shape=INPUT_SIZE)
hidden = tf.keras.layers.Dense(units=10, activation=tf.nn.sigmoid)(input)
output = tf.keras.layers.Dense(units=2, activation=tf.nn.sigmoid)(hidden)


# DropOut
INPUT_SIZE = (20, 1)

input = tf.placeholder(tf.float32, shape=INPUT_SIZE)
dropout = tf.keras.layers.Dropout(rate=0.5)(input)
hidden = tf.keras.layers.Dense(units=10, activation=tf.nn.sigmoid)(input)
output = tf.keras.layers.Dense(units=2, activation=tf.nn.sigmoid)(hidden)


# Conv1D 연습
# 주요 파라미터: filter, kernel_size
INPUT_SIZE = (1, 28, 28)

input = tf.placeholder(tf.float32, shape=INPUT_SIZE)
conv = tf.keras.layers.Conv1D(
    filters=10,
    kernel_size=3,
    padding='same',
    activation=tf.nn.relu
) (input)


# DropOut 적용 & Conv1D
input = tf.placeholder(tf.float32, shape=INPUT_SIZE)
dropout = tf.keras.layers.Dropout(rate=0.5)(input)
conv = tf.keras.layers.Conv1D(
    filters=10,
    kernel_size=3,
    padding='same',
    activation=tf.nn.relu
) (dropout)


# max pooling
# output으로 내보낼때는 Flatten
input = tf.placeholder(tf.float32, shape=INPUT_SIZE)
dropout = tf.keras.layers.Dropout(rate=0.5)(input)
conv = tf.keras.layers.Conv1D(
    filters=10,
    kernel_size=3,
    padding='same',
    activation=tf.nn.relu
) (dropout)
max_pool = tf.keras.layers.MaxPool1D(pool_size=3, padding='same')(conv)
flatten = tf.keras.layers.Flatten()(max_pool)