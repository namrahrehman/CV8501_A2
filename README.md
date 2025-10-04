# CV8501 Assignment 2 — Skin Lesion Classification as Medical VQA

This repository contains code for **skin lesion classification** using the HAM10000 dataset by reframing it as a **medical Visual Question Answering (VQA)** problem.

## Overview
We compare:
- **ViT-Base (google/vit-base-patch16-224)** – zero-shot and fine-tuned
- **LLaVA-Med v1.5 (Mistral-7B)** – zero-shot and LoRA fine-tuned

All models are evaluated on Accuracy, Macro/Weighted F1, and one-vs-rest AUC.

## Structure
preprocessing/ → Script to create 7-option VQA dataset
vit/ → ViT fine-tuning and zero-shot evaluation scripts
vlm_llava_med/ → Closed-set scoring for LLaVA-Med results
data/ → Place HAM10000 dataset (train/val/test CSV + images)
results/ → Outputs and plots


## Quick Start
```bash
# Create environment
conda env create -f environment.yml
conda activate cv8501-a2

# Generate VQA data
python preprocessing/make_vqa_all7.py --preprocessed_dir data/preprocessed_224_full

# Train ViT baseline
python vit/vit_finetuning.py --data_dir data/preprocessed_224_full

# Zero-shot ViT
python vit/vit_zero_shot_evaluation.py --data_dir data/preprocessed_224_full

For LLaVA-Med fine-tuning and evaluation, see vlm_llava_med/README_LLAVA_MED.md.
