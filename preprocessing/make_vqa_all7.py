#!/usr/bin/env python3
import json, os, random, argparse, pandas as pd, numpy as np
from collections import Counter

def set_seed(s):
    random.seed(s); np.random.seed(s)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessed_dir", required=True, type=str)
    ap.add_argument("--out_dir", default="data/ham10000_vqa_7options", type=str)
    ap.add_argument("--seed", default=42, type=int)
    return ap.parse_args()

def create_7_option_dataset(preprocessed_dir, out_dir, seed=42):
    set_seed(seed)
    print("HAM10000 VQA Dataset - All 7 Options")
    print("="*50)

    PREPROCESSED_DIR = preprocessed_dir
    IMG_DIR = os.path.join(PREPROCESSED_DIR, "images")
    OUT_DIR = out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    DX2NAME = {
        "akiec": "Actinic keratoses and intraepithelial carcinoma (Bowen's disease)",
        "bcc":   "Basal cell carcinoma",
        "bkl":   "Benign keratosis-like lesion",
        "df":    "Dermatofibroma",
        "nv":    "Melanocytic nevus",
        "mel":   "Melanoma",
        "vasc":  "Vascular lesion",
    }
    ALL_CODES = ["akiec", "bcc", "bkl", "df", "nv", "mel", "vasc"]

    print("Loading metadata...")
    all_data = []
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(PREPROCESSED_DIR, f"{split}.csv")
        if os.path.exists(split_path):
            df = pd.read_csv(split_path)
            df['split'] = split
            all_data.append(df)
            print(f"   Loaded {len(df)} samples from {split}")

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"   Total samples: {len(combined_df)}")

    rows = []
    for _, row in combined_df.iterrows():
        img_path = os.path.join(IMG_DIR, f"{row['image_id']}.jpg")
        if os.path.exists(img_path) and row['dx'] in DX2NAME:
            rows.append({
                "lesion_id": f"lesion_{row['image_id']}",
                "image_id": row['image_id'],
                "dx": row['dx'],
                "img": img_path,
                "split": row.get('split', 'unknown'),
                "age": row.get('age', ''),
                "sex": row.get('sex', ''),
                "localization": row.get('localization', '')
            })
    print(f"   Valid samples: {len(rows)}")

    MCQ_TEMPLATES = [
        "What is the most likely diagnosis?\nOptions: {choices}\nAnswer with the option text.",
        "Choose the correct diagnosis for this image:\nOptions: {choices}\nAnswer with the option text.",
        "Select the lesion type:\nOptions: {choices}\nAnswer with the option text.",
        "What is the dermatological diagnosis?\nOptions: {choices}\nAnswer with the option text.",
        "Identify the skin condition:\nOptions: {choices}\nAnswer with the option text.",
        "What type of lesion is shown?\nOptions: {choices}\nAnswer with the option text.",
        "Classify this dermatoscopic image:\nOptions: {choices}\nAnswer with the option text."
    ]

    def make_mcq_7(item):
        correct = DX2NAME[item["dx"]]
        all_names = [DX2NAME[c] for c in ALL_CODES]
        random.shuffle(all_names)
        choices_str = "; ".join(all_names)
        q = random.choice(MCQ_TEMPLATES).format(choices=choices_str)
        return {
            "id": f"ham_mcq7_{item['image_id']}",
            "image": item["img"],
            "conversations": [
                {"from": "human", "value": "<image>\n" + q},
                {"from": "gpt", "value": correct}
            ],
            "metadata": {
                "image_id": item["image_id"],
                "dx": item["dx"],
                "age": item.get("age", ""),
                "sex": item.get("sex", ""),
                "localization": item.get("localization", ""),
                "choices": all_names,
                "num_options": 7
            }
        }

    print("\nCreating stratified train/validation splits...")
    labels = [x["dx"] for x in rows]
    groups = [x["lesion_id"] for x in rows]

    from sklearn.model_selection import StratifiedGroupKFold
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, val_idx = next(skf.split(rows, labels, groups))
    train_items = [rows[i] for i in train_idx]
    val_items = [rows[i] for i in val_idx]

    print(f"   Training samples: {len(train_items)}")
    print(f"   Validation samples: {len(val_items)}")

    print("\nGenerating VQA datasets with all 7 options...")
    train_data = [make_mcq_7(item) for item in train_items]
    val_data = [make_mcq_7(item) for item in val_items]

    print("\nSaving datasets...")
    with open(os.path.join(OUT_DIR, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"   OK: train.json - {len(train_data)} samples")

    with open(os.path.join(OUT_DIR, "val.json"), "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    print(f"   OK: val.json - {len(val_data)} samples")

    train_classes = [item["dx"] for item in train_items]
    val_classes = [item["dx"] for item in val_items]

    dataset_info = {
        "dataset_name": "HAM10000_VQA_7Options",
        "description": "VQA dataset with all 7 diagnostic options for easier learning",
        "format": "Multiple Choice Questions (7 options)",
        "total_samples": len(rows),
        "train_samples": len(train_items),
        "val_samples": len(val_items),
        "classes": DX2NAME,
        "question_format": {
            "type": "Multiple Choice",
            "options_per_question": 7,
            "description": "Each question presents all 7 diagnostic options, with one correct answer"
        },
        "class_distribution": {
            "train": dict(Counter(train_classes)),
            "validation": dict(Counter(val_classes))
        },
        "files": {
            "train": "train.json",
            "validation": "val.json",
            "images": os.path.join(PREPROCESSED_DIR, "images")
        }
    }

    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj

    dataset_info = convert_numpy_types(dataset_info)
    with open(os.path.join(OUT_DIR, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    print(f"\nDataset creation complete!")
    print(f"   Output directory: {OUT_DIR}")
    print(f"   Total samples: {len(rows)}")
    print(f"   Training: {len(train_items)}")
    print(f"   Validation: {len(val_items)}")
    print(f"   Format: All 7 options per question")
    print(f"\nCreated files:")
    for file in os.listdir(OUT_DIR):
        print(f"   {file}")
    return OUT_DIR

if __name__ == "__main__":
    args = parse_args()
    create_7_option_dataset(args.preprocessed_dir, args.out_dir, args.seed)
