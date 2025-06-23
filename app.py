import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load models (disable compile to speed up)
encoder = tf.keras.models.load_model("models/encoder.h5", compile=False)
decoder = tf.keras.models.load_model("models/decoder.h5", compile=False)

# Generate digit images from decoder using encoded real MNIST digits
def generate_digit_images(digit, n=5):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)

    # Use real examples of the selected digit
    digit_imgs = x_train[y_train == digit][:n]
    digit_imgs = tf.convert_to_tensor(digit_imgs)
    digit_labels = tf.one_hot([digit] * n, depth=10)

    # Encode → Decode
    z_mean, z_log_var = encoder([digit_imgs, digit_labels])
    z = z_mean
    generated_imgs = decoder([z, digit_labels])
    return generated_imgs.numpy().squeeze()

# Streamlit interface
st.title("Handwritten Digit Generator (CVAE - TensorFlow)")
digit = st.selectbox("Select a digit (0–9)", list(range(10)))
images = generate_digit_images(digit)

# Display images
fig, axs = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    axs[i].imshow(images[i], cmap='gray')
    axs[i].axis('off')
st.pyplot(fig)
