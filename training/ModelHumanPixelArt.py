import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
import json
import datetime
import matplotlib
from collections import Counter
import sys
import time

matplotlib.use('Agg') # Wymusza renderowanie obrazków w tle

# --- KONFIGURACJA ---
BATCH_SIZE = 64  # Zmniejszony dla stabilności przy głębszym modelu
IMAGE_SIZE = 64
NOISE_DIM = 256
BUFFER_SIZE = 2000
DATA_DIR = Path("dataset_images")
CSV_PATH = "dataset.csv"

# ==============================================================================
# 1. PRZYGOTOWANIE DANYCH
# ==============================================================================
print("Wczytywanie i filtrowanie pliku CSV...")
try:
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Sukces: Załadowano plik {CSV_PATH}. Znaleziono {len(df)} obrazów w bazie.")
except FileNotFoundError:
    print(f"❌ KRYTYCZNY BŁĄD: Nie znaleziono pliku '{CSV_PATH}'!")
    print("Upewnij się, że plik dataset.csv znajduje się dokładnie w tym samym folderze co skrypt.")
    sys.exit(1) # Zatrzymujemy skrypt, nie ma sensu iść dalej
except Exception as e:
    print(f"❌ KRYTYCZNY BŁĄD: Problem z odczytem CSV: {e}")
    sys.exit(1)

# KROK 1: Zbieramy wszystkie tagi, żeby policzyć ich wystąpienia
all_raw_tags = []
for index, row in df.iterrows():
    tags = [str(val) for val in row[1:] if str(val) != 'none' and pd.notna(val)]
    all_raw_tags.extend(tags)

# Liczymy, ile razy wystąpił każdy tag
tag_counts = Counter(all_raw_tags)
total_images = len(df)

# KROK 2: Tworzymy "Białą listę" tagów
# Zostawiamy tylko te, które występują więcej niż np. 15 razy, ale rzadziej niż na 95% obrazków
MIN_OCCURRENCES = 15
MAX_OCCURRENCES = total_images * 0.95 

valid_tags = {tag for tag, count in tag_counts.items() if MIN_OCCURRENCES <= count <= MAX_OCCURRENCES}

print(f"Początkowa liczba unikalnych tagów: {len(tag_counts)}")
print(f"Liczba tagów po odfiltrowaniu skrajności: {len(valid_tags)}")

tags_list = []
image_paths = []

# KROK 3: Ładowanie właściwe z filtrowaniem
for index, row in df.iterrows():
    img_path = DATA_DIR / row['filename']
    
    # Bierzemy tylko tagi, które są na naszej białej liście
    tags = [str(val) for val in row[1:] if str(val) != 'none' and pd.notna(val) and str(val) in valid_tags]
    
    if img_path.exists():
        tags_list.append(tags)
        image_paths.append(str(img_path))

mlb = MultiLabelBinarizer()
labels_encoded = mlb.fit_transform(tags_list)
NUM_CLASSES = len(mlb.classes_)

with open('tags_dictionary.json', 'w') as f:
    json.dump(list(mlb.classes_), f)

print(f"Gotowe! Model będzie trenowany na {NUM_CLASSES} wyselekcjonowanych klasach.")


def load_and_preprocess_data(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=4) 
    
    image = tf.cast(image, tf.float32)
    
    alpha = image[:, :, 3:] / 255.0 
    rgb = image[:, :, :3]
    
    bg = tf.zeros_like(rgb) 
    
    image = rgb * alpha + bg * (1.0 - alpha)
    image = (image / 127.5) - 1.0 
    label = tf.cast(label, tf.float32)
    
    return image, label

dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels_encoded))
dataset = dataset.map(load_and_preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ==============================================================================
# 2. ARCHITEKTURA MODELU (Functional API - cGAN)
# ==============================================================================

def res_block(x, filters):
    shortcut = x
    x = keras.layers.Conv2D(filters, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    x = keras.layers.Conv2D(filters, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.add([shortcut, x])
    return x

# --- NOWA, LEPSZA METODA TRENOWANIA DLA GANÓW (WGAN-GP) LUB ZMIENIONA ARCHITEKTURA ---

def make_generator_model(num_classes):
    # Dwa ODDZIELNE wejścia
    noise_input = keras.Input(shape=(NOISE_DIM,))
    label_input = keras.Input(shape=(num_classes,))
    
    # Przekształcamy tagi w potężny wektor wpływu (512)
    label_embedding = keras.layers.Dense(512)(label_input) 
    label_embedding = keras.layers.LeakyReLU(0.2)(label_embedding)
    
    # Szum też musi mieć silną reprezentację (zmieniamy z 256 na 512, by miały równe szanse)
    x_noise = keras.layers.Dense(8 * 8 * 512, use_bias=False)(noise_input)
    x_noise = keras.layers.Reshape((8, 8, 512))(x_noise)

    # --- ZMIANA KRYTYCZNA: Zamiast łączyć na początku, "nakładamy" tagi na szum ---
    # Tagi stają się filtrem (mnożnikiem) dla szumu
    label_filter = keras.layers.Dense(512, activation='sigmoid')(label_embedding)
    label_filter = keras.layers.Reshape((1, 1, 512))(label_filter) 
    
    x = keras.layers.Multiply()([x_noise, label_filter])
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(0.2)(x)

    # 8x8 -> 16x16
    x = keras.layers.UpSampling2D(interpolation='nearest')(x)
    x = keras.layers.Conv2D(256, 3, padding='same')(x)
    x = res_block(x, 256)
    x = keras.layers.LeakyReLU(0.2)(x)

    # 16x16 -> 32x32
    x = keras.layers.UpSampling2D(interpolation='nearest')(x)
    x = keras.layers.Conv2D(128, 3, padding='same')(x)
    x = res_block(x, 128)
    x = keras.layers.LeakyReLU(0.2)(x)

    # 32x32 -> 64x64
    x = keras.layers.UpSampling2D(interpolation='nearest')(x)
    x = keras.layers.Conv2D(64, 3, padding='same')(x)
    
    # Ostatnia warstwa - wygenerowanie 3 kolorów RGB z mocną aktywacją tanh
    x = keras.layers.Conv2D(3, 3, padding='same', activation='tanh')(x)

    return keras.Model([noise_input, label_input], x, name="Generator")

def make_discriminator_model(num_classes):
    image_input = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    label_input = keras.Input(shape=(num_classes,))

    # Dodanie szumu na wejściu (ułatwia naukę struktury)
    x = keras.layers.GaussianNoise(0.1)(image_input)

    x = keras.layers.Conv2D(64, 3, strides=2, padding='same')(x) 
    x = keras.layers.LeakyReLU(0.2)(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Conv2D(128, 3, strides=2, padding='same')(x) 
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Conv2D(256, 3, strides=2, padding='same')(x) 
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Flatten()(x)
    
    # Warunkowanie
    label_feat = keras.layers.Dense(512)(label_input)
    label_feat = keras.layers.LeakyReLU(0.2)(label_feat)
    
    x = keras.layers.Concatenate()([x, label_feat])
    x = keras.layers.Dense(256)(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    out = keras.layers.Dense(1, activation='sigmoid')(x)

    return keras.Model([image_input, label_input], out, name="Discriminator")

generator = make_generator_model(NUM_CLASSES)
discriminator = make_discriminator_model(NUM_CLASSES)

# TTUR (Two-Time Scale Update Rule) - Dyskryminator uczy się nieco szybciej
generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False) 

# ==============================================================================
# 3. TRENOWANIE
# ==============================================================================

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    # One-sided label smoothing (0.9 zamiast 1.0)
    real_loss = cross_entropy(tf.ones_like(real_output) * 0.9, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

preview_labels = labels_encoded[:16] 
preview_noise = tf.random.normal([16, NOISE_DIM])

def save_images(model, epoch, folder="generated_images"):
    os.makedirs(folder, exist_ok=True)
    predictions = model([preview_noise, tf.convert_to_tensor(preview_labels, dtype=tf.float32)], training=False)
    
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axs.flat):
        ax.imshow((predictions[i] * 0.5 + 0.5).numpy()) 
        ax.axis('off')
    plt.savefig(f"{folder}/image_at_epoch_{epoch:03d}.png")
    plt.close()

@tf.function 
def train_step(images, labels):
    current_batch_size = tf.shape(images)[0]
    noise = tf.random.normal([current_batch_size, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([noise, labels], training=True)

        real_output = discriminator([images, labels], training=True)
        fake_output = discriminator([generated_images, labels], training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss, tf.reduce_mean(real_output), tf.reduce_mean(fake_output)

# Reszta funkcji (train, summary_writer) pozostaje bez zmian strukturalnych, 
# ale będzie korzystać z nowej architektury.

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

def train(dataset, epochs, checkpoint_path="generator_checkpoint"):
    os.makedirs(checkpoint_path, exist_ok=True)
    # Zmieniamy format dla obu plików na .weights.h5, to standard Keras 3
    gen_checkpoint_file = os.path.join(checkpoint_path, "generator.weights.h5")
    disc_checkpoint_file = os.path.join(checkpoint_path, "discriminator.weights.h5")
    
    # 1. Poprawione wczytywanie
    if os.path.exists(gen_checkpoint_file) and os.path.exists(disc_checkpoint_file):
        try:
            # Wczytujemy TYLKO wagi, bo struktura jest już zbudowana w Pythonie
            generator.load_weights(gen_checkpoint_file)
            discriminator.load_weights(disc_checkpoint_file)
            print("✅ Sukces: Wczytano zapisane wagi z poprzedniego treningu.")
        except ValueError as e:
            print(f"❌ UWAGA: Błąd wczytywania wag ({e}). Zaczynamy od zera.")
    else:
        print("Brak zapisanych wag. Zaczynamy trening od zera.")
    
    print("Rozpoczynam trenowanie...")
    preview_labels_tensor = tf.convert_to_tensor(preview_labels, dtype=tf.float32)
    
    for epoch in range(epochs):
        for image_batch, label_batch in dataset:
            gen_loss, disc_loss, d_real, d_fake = train_step(image_batch, label_batch)
        
        print(f"Epoka {epoch + 1} | Gen: {gen_loss:.4f} | Disc: {disc_loss:.4f} | D_Real: {d_real:.4f} | D_Fake: {d_fake:.4f}")
        
        if (epoch + 1) % 1 == 0:
            save_images(generator, epoch + 1)
            
            # 1. Zapisujemy same wagi w dotychczasowym folderze (żeby nie zepsuć wczytywania przy restarcie)
            generator.save_weights(gen_checkpoint_file)
            discriminator.save_weights(disc_checkpoint_file)
            
            # --- NOWOŚĆ: ZAPIS CAŁEGO MODELU DO TWOJEGO FOLDERU ---
            # Dodajemy 'r' przed ścieżką, aby Windows poprawnie odczytał ukośniki
            export_dir = r"C:\Users\rucki\Desktop\Portfolio Python\Pixel-Art-AI-Generator\Model"
            
            # Upewniamy się, że ten folder na pulpicie istnieje (jeśli nie, Python go stworzy)
            os.makedirs(export_dir, exist_ok=True)
            
            full_model_path = os.path.join(export_dir, "generator_full.keras")
            generator.save(full_model_path)
            
            print(f"💾 Zapisano wagi (lokalnie) oraz PEŁNY MODEL ({full_model_path}) po epoce {epoch + 1}.")
            
            print("❄️ Przerwa na chłodzenie GPU... (30 sekund)")
            time.sleep(30)

train(train_dataset, 500) # Zwiększono liczbę epok dla głębszego modelu