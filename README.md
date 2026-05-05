# Comparison of VAE and GAN models for Generative AI

This project explores the evolution of generative models, starting from basic **Autoencoders (AE)** and moving towards more complex architectures like **Variational Autoencoders (VAE)** and different **Generative Adversarial Networks (GAN)**.

The project also introduces other types of architectures, such as autoregressive models like **PixelCNN** (on MNIST), flow-based models like **RealNVP** and **GLOW**, and concludes with diffusion models like **DDPM** using a **U-Net** architecture.

The primary objective is to understand how latent space representation allows models to generate new, high-quality data samples.

### Objectives
* **Understand** the limitations of standard AE for data generation.
* **Implement VAE** to smooth the latent space using probabilistic distributions.
* **Deploy GANs** to produce sharp, realistic samples.
* **Use WGAN-GP** to achieve better stability and model convergence.
* **Compare** the performance and output quality across different architectures.

---

## Approach

### 1. AE: Fundamentals
Analysis of how Autoencoders compress data into a bottleneck (latent space) and reconstruct it.
* **Dataset:** MNIST.
* **Goal:** Image reconstruction and visualization of the latent space clusters.

### 2. VAE: From Reconstruction to Generation
Transitioning to Variational Autoencoders by adding a probabilistic layer to the latent space.
* **Dataset:** [Cat faces dataset](https://github.com/fferlito/Cat-faces-dataset).
* **Goal:** Generate new, unique cat images by sampling the latent space.

### 3. GAN: Adversarial Training and WGAN-GP
Implementation of the minimax game between a Generator and a Discriminator.
* **Goal:** Achieve higher image sharpness and overcome the "blurriness" often found in VAEs.
* **Focus:** Using Wasserstein GAN with Gradient Penalty (WGAN-GP) for improved training stability.

### 4. Introduction to PixelCNN
* **Dataset:** MNIST.
* **Goal:** Observe how an autoregressive model produces images pixel by pixel, conditioned on previous ones.

### 5. RealNVP (Flow-based Model)
* **Concept:** Implementing non-volume preserving (NVP) transformations.
* **Goal:** Use invertible mapping to transform a simple distribution into a complex data distribution.

### 6. GLOW
* **Goal:** Explore more efficient and scalable generative flows.

### 7. Diffusion Models (DDPM)
* **Architecture:** U-Net.
* **Goal:** Learn the reverse process of adding noise to data to generate samples from pure Gaussian noise.

## Bibliographie

### Manuals

- Géron, A. *Machine Learning avec Scikit-Learn*. Dunod.
- Charniak, E. *Introduction au Deep Learning*. Dunod.
- Géron, A. *Deep Learning avec TensorFlow*. Dunod.
- Foster, D. *Deep learning génératif*. Dunod.
