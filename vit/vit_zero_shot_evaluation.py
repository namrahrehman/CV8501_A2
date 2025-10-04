#!/usr/bin/env python3
import argparse, os, json, torch, pandas as pd
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
import matplotlib.pyplot as plt, seaborn as sns
from PIL import Image
from tqdm import tqdm
import numpy as np

class HAM10000Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_dir, processor):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.processor = processor
        config_path = os.path.join(os.path.dirname(csv_path), "config.json")
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, f"{row['image_id']}.jpg")
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(image, return_tensors="pt")
        label = row['label']
        return {'pixel_values': inputs['pixel_values'].squeeze(0),'label': torch.tensor(label, dtype=torch.long)}

def evaluate_model(model, dataloader, device):
    model.eval(); all_predictions=[]; all_labels=[]; all_probabilities=[]
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device); labels = batch['label'].to(device)
            outputs = model(pixel_values); probabilities = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(outputs.logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    acc = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
    try:
        auc_ovr = roc_auc_score(all_labels, all_probabilities, multi_class='ovr', average='macro')
        auc_ovo = roc_auc_score(all_labels, all_probabilities, multi_class='ovo', average='macro')
    except ValueError:
        auc_ovr = auc_ovo = 0.0
    return {'accuracy': acc,'f1_macro': f1_macro,'f1_weighted': f1_weighted,'auc_ovr': auc_ovr,'auc_ovo': auc_ovo,
            'predictions': all_predictions,'labels': all_labels,'probabilities': all_probabilities}

def plot_confusion_matrix(labels, predictions, class_names, title="Confusion Matrix"):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.tight_layout()
    plt.savefig('vit_zero-shot_confusion_matrix.png', dpi=300, bbox_inches='tight'); plt.close()

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True, type=str)
    p.add_argument('--batch_size', type=int, default=16)
    return p.parse_args()

def main():
    args = get_args()
    data_dir = args.data_dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=7, ignore_mismatched_sizes=True).to(device)
    with open(os.path.join(data_dir, "config.json"), 'r') as f:
        config = json.load(f)
    diagnosis_mapping = config['diagnosis_mapping']
    class_names = [dx for dx, _ in sorted(diagnosis_mapping.items(), key=lambda x: x[1])]
    test_dataset = HAM10000Dataset(os.path.join(data_dir, "test.csv"), os.path.join(data_dir, "images"), processor)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    results = evaluate_model(model, test_loader, device)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Macro: {results['f1_macro']:.4f}, F1 Weighted: {results['f1_weighted']:.4f}")
    print(f"AUC OvR: {results['auc_ovr']:.4f}, AUC OvO: {results['auc_ovo']:.4f}")
    report = classification_report(results['labels'], results['predictions'], target_names=class_names, digits=4)
    print(report)
    plot_confusion_matrix(results['labels'], results['predictions'], class_names)
    with open('zero_shot_results.json', 'w') as f: json.dump({
        'model': model_name, 'evaluation_type': 'zero_shot', 'accuracy': results['accuracy'],
        'f1_macro': results['f1_macro'], 'f1_weighted': results['f1_weighted'],
        'auc_ovr': results['auc_ovr'], 'auc_ovo': results['auc_ovo'],
        'num_test_samples': len(test_dataset), 'class_names': class_names
    }, f, indent=2)

if __name__ == "__main__":
    main()
