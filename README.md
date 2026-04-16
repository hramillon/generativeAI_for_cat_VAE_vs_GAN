# Comparison VAE and GAN models for generative AI

This project explores the evolution of generative models, starting from basic **Autoencoders (AE)** and moving towards more complex architectures like **Variational Autoencoders (VAE)** and **Generative Adversarial Networks (GAN)**. 

The primary objective is to understand how latent space representation allows models to generate new, data samples.

objectives :

* Understand the limitations of standard AE for generation.
* Implement and tune VVAE to smooth the latent space.
* Deploy GAN to produce samples.
* Compare them

## Approach

### 1. AE, Fundamentals
Analysis of how Autoencoders compress data into a bottleneck (latent space) and reconstruct it.
* **Dataset:** MNIST
* **Goal:** Image reconstruction and visualization of the latent space.

### 2. Pre-training with Autoencoders
Testing if an AE can serve as a feature extractor for more complex tasks.
* **Dataset:** CIFAR-10
* **Goal:** Use the encoder part to "pre-learn" features before a classification task.

### 3. VAE: From Reconstruction to Generation
Transitioning to Variational Autoencoders by adding a probabilistic layer to the latent space.
* **Dataset:** [Cat Samples made of different sources :](https://github.com/fferlito/Cat-faces-dataset)
* **Goal:** Generate new cat images.

### 4. GAN: Adversarial Training
* **Goal:** Achieve higher image sharpness and overcome the "blurriness" of VAEs.


## Bibliographie

### Manuals

- Géron, A. *Machine Learning avec Scikit-Learn*. Dunod.
- Charniak, E. *Introduction au Deep Learning*. Dunod.
- Géron, A. *Deep Learning avec TensorFlow*. Dunod.
- Foster, D. *Deep learning génératif*. Dunod.
