# Chest X-Ray Generation with VAE

A PyTorch implementation of a **Variational Autoencoder (VAE)** designed to generate high-resolution ($256 \times 256$) synthetic chest X-ray images. This project includes custom model architectures, real-time evaluation using **FID** (Fréchet Inception Distance) and **IS** (Inception Score), and scripts for both training and inference.

<img width="1500" height="500" alt="kle_imgs" src="https://github.com/user-attachments/assets/b64394ea-d8e0-423b-b681-1261d1474b2a" />

## 📌 Features
* **Custom VAE Architecture:** Supports both 5-layer and 6-layer deep convolutional encoders/decoders.
* **Metric Integration:** Calculates **FID** and **Inception Score** at the end of every epoch to track generation quality.
* **Grayscale Optimization:** Specialized handling for 1-channel X-ray data, including adaptation layers for Inception-v3 metrics.
* **Latent Space:** Uses a 1024-dimensional latent vector to capture complex anatomical features.

