import os
import tensorflow as tf
import tensorflow_hub as hub

# Set paths and constants
DATA_DIR = 'path_to_juiceboxpix'
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 10

# Data preprocessing
def parse_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]

def process_path(file_path):
    img = parse_image(file_path)
    label = get_label(file_path)
    return img, label

list_ds = tf.data.Dataset.list_files(str(DATA_DIR + '/*/*'))
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

# Data split
DATASET_SIZE = len(list_ds)
train_size = int(0.8 * DATASET_SIZE)
val_size = int(0.1 * DATASET_SIZE)

train_ds = labeled_ds.take(train_size)
val_ds = labeled_ds.skip(train_size).take(val_size)
test_ds = labeled_ds.skip(train_size + val_size)

# Model definition
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

model = tf.keras.Sequential([
    hub.KerasLayer(model_url, input_shape=IMG_SIZE + (3,)),
    tf.keras.layers.Dense(len(os.listdir(DATA_DIR)), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(train_ds.batch(BATCH_SIZE),
          validation_data=val_ds.batch(BATCH_SIZE),
          epochs=EPOCHS)

# Evaluation
loss, accuracy = model.evaluate(test_ds.batch(BATCH_SIZE))
print(f"Test set accuracy: {accuracy * 100:.2f}%")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('juicebox_detector.tflite', 'wb') as f:
    f.write(tflite_model)
1