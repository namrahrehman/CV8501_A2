# LLaVA-Med fine-tuning and evaluation

Pin the upstream repo and follow its install instructions.

## Zero-shot
Use the 7-option prompt:
```
<image>
What is the most likely diagnosis?
Options: Actinic keratoses and intraepithelial carcinoma (Bowen's disease); Basal cell carcinoma; Benign keratosis-like lesion; Dermatofibroma; Melanocytic nevus; Melanoma; Vascular lesion
Answer with the option text only.
```

## LoRA fine-tuning (summary)
- r=64, alpha=128, lr=2e-4 (projector 2e-5), bf16, GA=16, seq=1024, cosine, warmup=0.03, epochs=2, ZeRO-3
- image encoder: openai/clip-vit-large-patch14-336
- projector: mlp2x_gelu

## Closed-set scoring
After generation, evaluate with:
```bash
python vlm_llava_med/eval_closedset.py   --preds llava_outputs.jsonl   --option_list vlm_llava_med/prompts/option_list.txt   --out metrics_llava.json
```
