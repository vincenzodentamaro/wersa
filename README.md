# WERSA: Wavelet-Enhanced Random Spectral Attention

This repository provides the official implementation of **WERSA**, a novel attention mechanism with linear **O(n)** time complexity, designed to scale Transformer models to very long sequences **without a performance trade-off**.

This guide provides a complete walkthrough for installing the `wersa` package and building a custom Qwen-like causal language model using this attention layer.

---

## 🔬 The Science Behind WERSA

Standard attention mechanisms in Transformers have a **quadratic (O(n²))** time complexity, making them computationally expensive for long sequences. **WERSA** solves this by combining several powerful principles to achieve **linear (O(n))** efficiency while maintaining high performance.

### Core Principles

- **Multi-Resolution Analysis**  
  WERSA uses Haar wavelet transforms to decompose the input into multiple scales. This allows the model to simultaneously capture fine-grained local details (with high-frequency wavelets) and broad global context (with low-frequency wavelets).

- **Adaptive Filtering**  
  An MLP generates input-dependent filters, and learnable `scale_weights` modulate each wavelet level. This enables the model to dynamically prioritize the most informative frequency components, suppressing noise and amplifying important patterns.

- **Linear Complexity via Random Features**  
  To achieve O(n) complexity, WERSA uses random feature projection to approximate the softmax kernel, avoiding the computation of the full quadratic attention matrix. The **WERSA Long-Context Approximation Theorem** provides formal guarantees that this approach approximates standard attention with bounded error.

---

## 📊 WERSA Architecture Flow

The diagram below illustrates the flow of information through the WERSA mechanism, from input projections to the final attention output.

<!-- Optionally, insert architecture diagram here -->

---

## 🚀 Performance Highlights

The scientific principles behind WERSA translate into tangible benefits:

- **ArXiv classification benchmark**  
  WERSA improves accuracy over vanilla attention by **1.2%**, reduces training time by **81%**, and lowers FLOPS by **73.4%**.

- **ArXiv-128k dataset**  
  On this large-scale benchmark (which causes Out-Of-Memory errors for standard and FlashAttention-2), WERSA achieves the best accuracy (**79.1%**) among all viable methods, demonstrating its unique ability to scale to very long sequences.

---

## ⚙️ Step 1: Environment & Installation

Before using the package, ensure you have a Python environment with **PyTorch** and the necessary **Hugging Face** libraries.

### 1. Install Core Dependencies with Pip

Install PyTorch for your specific CUDA version, along with the core Hugging Face libraries:

```bash
# Example for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Hugging Face libraries
pip install transformers datasets accelerate



# By Vincenzo Dentamaro vincenzo.dentamaro@uniba.it 
