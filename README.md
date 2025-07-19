WERSA: Wavelet-Enhanced Random Spectral AttentionThis repository provides the official implementation of WERSA, a novel attention mechanism with linear O(n) time complexity, designed to scale Transformer models to very long sequences without a performance trade-off.This guide provides a complete walkthrough for installing the wersa package and building a custom Qwen-like causal language model using this attention layer.🔬 The Science Behind WERSAStandard attention mechanisms are computationally costly, with a quadratic (O(n2)) complexity that makes processing long sequences impractical. WERSA addresses this by combining several powerful principles to achieve linear (O(n)) efficiency while maintaining high performance.Core PrinciplesMulti-Resolution Analysis: WERSA uses Haar wavelet transforms to decompose the input into multiple scales. This allows the model to simultaneously capture fine-grained local details (with high-frequency wavelets) and broad global context (with low-frequency wavelets).Adaptive Filtering: An MLP generates input-dependent filters, and learnable scale_weights modulate each wavelet level. This allows the model to dynamically prioritize the most informative frequency components, suppressing noise and amplifying important patterns.Linear Complexity via Random Features: To achieve O(n) complexity, WERSA uses random feature projection to approximate the softmax kernel, avoiding the computation of the full quadratic attention matrix. The WERSA Long-Context Approximation Theorem provides formal guarantees that this approach approximates standard attention with bounded error.WERSA Architecture FlowThe diagram below illustrates the flow of information through the WERSA mechanism, from input projections to the final attention output.Performance HighlightsThe scientific principles behind WERSA translate into tangible benefits. On the ArXiv classification benchmark, WERSA not only improves accuracy over vanilla attention by 1.2% but also cuts training time by 81% and FLOPS by 73.4%.On the challenging ArXiv-128k dataset, which causes Out-Of-Memory errors for standard and FlashAttention-2, WERSA achieves the best accuracy (79.1%) among all viable methods, demonstrating its unique ability to scale to very long sequences.⚙️ Step 1: Environment & InstallationBefore using the package, ensure you have a Python environment with PyTorch and the necessary Hugging Face libraries.1. Install Core Dependencies with PipInstall PyTorch for your specific CUDA version, along with the core Hugging Face libraries.# Example for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Hugging Face libraries
pip install transformers datasets accelerate
2. Install the WERSA PackageClone this repository and install the local wersa package in editable mode. This allows any changes you make to the source code to be immediately available.# Navigate to the root directory of this project
pip install -e .
🚀 Step 2: Building a Qwen-like Model with WERSAWith the package installed, you can now build a causal language model using the custom WersaForCausalLM class. This class replaces the standard attention mechanism with WERSA.The following script demonstrates how to configure and pre-train a small, Qwen-style architecture on the WikiText dataset.train_qwen_wersa.py# train_qwen_wersa.py
import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from wersa import WersaConfig, WersaForCausalLM

def main():
    # 1. Configuration and Tokenizer
    print("Setting up model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    config = WersaConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        max_position_embeddings=512,
        wersa_decomp_levels=4,
        wersa_random_features=128,
    )

    model = WersaForCausalLM(config)
    print(f"Model created with {model.num_parameters() / 1e6:.2f}M parameters.")

    # 2. Dataset Preparation
    print("Loading and preparing dataset...")
    raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    raw_dataset = raw_dataset.filter(lambda example: len(example['text']) > 0)
    raw_dataset = raw_dataset.shuffle(seed=42)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=config.max_position_embeddings)

    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"]
    )

    block_size = config.max_position_embeddings
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 3. Training
    print("Setting up Trainer...")
    training_args = TrainingArguments(
        output_dir="./wersa-qwen-style-pretrain",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting pre-training...")
    trainer.train()
    print("Pre-training finished.")

    # 4. Save the final model
    trainer.save_model("./wersa-qwen-style-final")
    tokenizer.save_pretrained("./wersa-qwen-style-final")
    print("Model and tokenizer saved successfully.")

if __name__ == "__main__":
    main()
