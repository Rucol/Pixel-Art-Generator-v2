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

# Use non-interactive backend for background rendering
matplotlib.use('Agg')

# --- CONFIGURATION ---
BATCH_SIZE = 64  
IMAGE_SIZE = 64
NOISE_DIM = 256
BUFFER_SIZE = 2000
DATA_DIR = Path("dataset_images")
CSV_PATH = "dataset.csv"

# ==============================================================================
# 1. DATA PREPARATION
# ==============================================================================
print("Loading and filtering CSV metadata...")
try:
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Success: Loaded {CSV_PATH}. Found {len(df)} images in database.")
except FileNotFoundError:
    print(f"❌ CRITICAL ERROR: Could not find '{CSV_PATH}'!")
    sys.exit(1)
except Exception as e:
    print(f"❌ CRITICAL ERROR: Problem reading CSV: {e}")
    sys.exit(1)

# Step 1: Collect and count all tags
all_raw_tags = []
for index, row in df.iterrows():
    tags = [str(val) for val in row[1:] if str(val) != 'none' and pd.notna(val)]
    all_raw_tags.extend(tags)

tag_counts = Counter(all_raw_tags)
total_images = len(df)

# Step 2: Tag Filtering (White-listing)
# Keeps tags appearing more than MIN_OCCURRENCES but less than 95% of total images
MIN_OCCURRENCES = 15
MAX_OCCURRENCES = total_images * 0.95 

valid_tags = {tag for tag, count in tag_counts.items() if MIN_OCCURRENCES <= count <= MAX_OCCURRENCES}

print(f"Original unique tags count: {len(tag_counts)}")
print(f"Tags count after outlier filtering: {len(valid_tags)}")

tags_list = []
image_paths = []

# Step 3: Loading and local path verification
for index, row in df.iterrows():
    img_path = DATA_DIR / row['filename']
    tags = [str(val) for val in row[1:] if str(val) != 'none' and pd.notna(val) and str(val) in valid_tags]
    
    if img_path.exists():
        tags_list.append(tags)
        image_paths.append(str(img_path))

mlb = MultiLabelBinarizer()
labels_encoded = mlb.fit_transform(tags_list)
NUM_CLASSES = len(mlb.classes_)

# Exporting tags dictionary for the FastAPI server
with open('tags_dictionary.json', 'w') as f:
    json.dump(list(mlb.classes_), f)

print(f"Ready! Training will proceed with {NUM_CLASSES} selected classes.")

def load_and_preprocess_data(path, label):
    """Loads, decodes and normalizes images to [-1, 1] range."""
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=4) 
    image = tf.cast(image, tf.float32)
    
    # Process alpha channel for background replacement
    alpha = image[:, :, 3:] / 255.0 
    rgb = image[:, :, :3]
    bg = tf.zeros_like(rgb) # Solid black background
    
    image = rgb * alpha + bg * (1.0 - alpha)
    image = (image / 127.5) - 1.0 
    label = tf.cast(label, tf.float32)
    
    return image, label

# Create TensorFlow Dataset pipeline
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels_encoded))
dataset = dataset.map(load_and_preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ==============================================================================
# 2. MODEL ARCHITECTURE (cGAN - Functional API)
# ==============================================================================

def res_block(x, filters):
    """Residual Block for stabilizing deeper layers."""
    shortcut = x
    x = keras.layers.Conv2D(filters, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    x = keras.layers.Conv2D(filters, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.add([shortcut, x])
    return x

def make_generator_model(num_classes):
    """Creates the Generator (Forge) - maps noise and tags to image."""
    noise_input = keras.Input(shape=(NOISE_DIM,))
    label_input = keras.Input(shape=(num_classes,))
    
    # Powerful tag embedding (512-dim influence vector)
    label_embedding = keras.layers.Dense(512)(label_input) 
    label_embedding = keras.layers.LeakyReLU(0.2)(label_embedding)
    
    # Initial noise representation
    x_noise = keras.layers.Dense(8 * 8 * 512, use_bias=False)(noise_input)
    x_noise = keras.layers.Reshape((8, 8, 512))(x_noise)

    # Conditioning: Applying tags as a multiplicative filter over noise
    label_filter = keras.layers.Dense(512, activation='sigmoid')(label_embedding)
    label_filter = keras.layers.Reshape((1, 1, 512))(label_filter) 
    
    x = keras.layers.Multiply()([x_noise, label_filter])
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(0.2)(x)

    # Progressive Upsampling: 8x8 -> 16x16 -> 32x32 -> 64x64
    upsampling_layers = [256, 128]
    for filters in upsampling_layers:
        x = keras.layers.UpSampling2D(interpolation='nearest')(x)
        x = keras.layers.Conv2D(filters, 3, padding='same')(x)
        x = res_block(x, filters)
        x = keras.layers.LeakyReLU(0.2)(x)

    x = keras.layers.UpSampling2D(interpolation='nearest')(x)
    x = keras.layers.Conv2D(64, 3, padding='same')(x)
    
    # Final output layer (RGB) with tanh activation
    x = keras.layers.Conv2D(3, 3, padding='same', activation='tanh')(x)

    return keras.Model([noise_input, label_input], x, name="Generator")

def make_discriminator_model(num_classes):
    """Creates the Discriminator (Judge) - validates image authenticity."""
    image_input = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    label_input = keras.Input(shape=(num_classes,))

    # Add noise to input for training stability
    x = keras.layers.GaussianNoise(0.1)(image_input)

    conv_layers = [64, 128, 256]
    for filters in conv_layers:
        x = keras.layers.Conv2D(filters, 3, strides=2, padding='same')(x) 
        x = keras.layers.BatchNormalization() if filters > 64 else lambda y: y
        x = keras.layers.LeakyReLU(0.2)(x)
        x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Flatten()(x)
    
    # Conditioning logic
    label_feat = keras.layers.Dense(512)(label_input)
    label_feat = keras.layers.LeakyReLU(0.2)(label_feat)
    
    x = keras.layers.Concatenate()([x, label_feat])
    x = keras.layers.Dense(256)(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    out = keras.layers.Dense(1, activation='sigmoid')(x)

    return keras.Model([image_input, label_input], out, name="Discriminator")

generator = make_generator_model(NUM_CLASSES)
discriminator = make_discriminator_model(NUM_CLASSES)

# TTUR (Two-Time Scale Update Rule) - Discriminator learns slightly slower/faster for stability
generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False) 

# ==============================================================================
# 3. TRAINING LOGIC
# ==============================================================================

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    # One-sided label smoothing (0.9 instead of 1.0) for stability
    real_loss = cross_entropy(tf.ones_like(real_output) * 0.9, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

preview_labels = labels_encoded[:16] 
preview_noise = tf.random.normal([16, NOISE_DIM])

def save_images(model, epoch, folder="generated_images"):
    """Generates and saves a grid of sample images during training."""
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

def train(dataset, epochs, checkpoint_path="generator_checkpoint"):
    os.makedirs(checkpoint_path, exist_ok=True)
    gen_checkpoint_file = os.path.join(checkpoint_path, "generator.weights.h5")
    disc_checkpoint_file = os.path.join(checkpoint_path, "discriminator.weights.h5")
    
    # Attempt to load weights from previous sessions
    if os.path.exists(gen_checkpoint_file) and os.path.exists(disc_checkpoint_file):
        try:
            generator.load_weights(gen_checkpoint_file)
            discriminator.load_weights(disc_checkpoint_file)
            print("✅ Success: Loaded existing weights from checkpoint.")
        except ValueError as e:
            print(f"❌ WARNING: Error loading weights ({e}). Starting from scratch.")
    else:
        print("No checkpoints found. Starting fresh training session.")
    
    print("Beginning training process...")
    for epoch in range(epochs):
        for image_batch, label_batch in dataset:
            gen_loss, disc_loss, d_real, d_fake = train_step(image_batch, label_batch)
        
        print(f"Epoch {epoch + 1} | Gen Loss: {gen_loss:.4f} | Disc Loss: {disc_loss:.4f} | D_Real: {d_real:.4f} | D_Fake: {d_fake:.4f}")
        
        # Periodic saves and monitoring
        if (epoch + 1) % 1 == 0:
            save_images(generator, epoch + 1)
            generator.save_weights(gen_checkpoint_file)
            discriminator.save_weights(disc_checkpoint_file)
            
            # Export full model for production use
            export_dir = r"C:\Users\rucki\Desktop\Portfolio Python\Pixel-Art-AI-Generator\Model"
            os.makedirs(export_dir, exist_ok=True)
            full_model_path = os.path.join(export_dir, "generator_full.keras")
            generator.save(full_model_path)
            
            print(f"💾 Saved full model to {full_model_path} after epoch {epoch + 1}.")
            print("❄️ Cooling down GPU... (30s break)")
            time.sleep(30)

# Run the training loop
train(train_dataset, 500)