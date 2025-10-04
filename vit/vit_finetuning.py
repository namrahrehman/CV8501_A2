#!/usr/bin/env python3
# (Argparse-enabled version of your fine-tuning script)
import argparse, os, json, torch, numpy as np, pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
import matplotlib.pyplot as plt, seaborn as sns
from PIL import Image
from tqdm import tqdm

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

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train(); total_loss=0; correct=0; total=0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in progress_bar:
        pixel_values = batch['pixel_values'].to(device); labels = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values, labels=labels); loss = outputs.loss
        loss.backward(); optimizer.step(); scheduler.step()
        total_loss += loss.item()
        predictions = torch.argmax(outputs.logits, dim=1)
        correct += (predictions == labels).sum().item(); total += labels.size(0)
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}','Acc': f'{100.*correct/total:.2f}%'})
    return total_loss / len(dataloader), correct / total

def evaluate_model(model, dataloader, device, class_names):
    model.eval(); all_predictions=[]; all_labels=[]; all_probabilities=[]; total_loss=0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device); labels = batch['label'].to(device)
            outputs = model(pixel_values, labels=labels); loss = outputs.loss
            probabilities = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(outputs.logits, dim=1)
            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
    try:
        auc_ovr = roc_auc_score(all_labels, all_probabilities, multi_class='ovr', average='macro')
        auc_ovo = roc_auc_score(all_labels, all_probabilities, multi_class='ovo', average='macro')
    except ValueError:
        auc_ovr = auc_ovo = 0.0
    avg_loss = total_loss / len(dataloader)
    return {'accuracy': accuracy,'f1_macro': f1_macro,'f1_weighted': f1_weighted,
            'auc_ovr': auc_ovr,'auc_ovo': auc_ovo,'loss': avg_loss,
            'predictions': all_predictions,'labels': all_labels,'probabilities': all_probabilities}

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(train_losses, label='Training Loss'); ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True)
    ax2.plot(train_accs, label='Training Accuracy'); ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.legend(); ax2.grid(True)
    plt.tight_layout(); plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.savefig('vit_training_progress.png', dpi=300, bbox_inches='tight'); plt.close()

def plot_confusion_matrix(labels, predictions, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.tight_layout()
    fn = title.lower().replace(" ", "_")
    plt.savefig(f'{fn}.png', dpi=300, bbox_inches='tight')
    # alias expected by LaTeX
    if "fine" in fn:
        plt.savefig('vit_fine-tuned_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def get_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True, type=str)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--warmup_steps', type=int, default=100)
    return p.parse_args()

def main():
    args = get_args()
    data_dir = args.data_dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate = args.lr; num_epochs = args.epochs; batch_size = args.batch_size; warmup_steps = args.warmup_steps
    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=7, ignore_mismatched_sizes=True).to(device)
    with open(os.path.join(data_dir, "config.json"), 'r') as f:
        config = json.load(f)
    diagnosis_mapping = config['diagnosis_mapping']
    class_names = [dx for dx, _ in sorted(diagnosis_mapping.items(), key=lambda x: x[1])]
    train_dataset = HAM10000Dataset(os.path.join(data_dir, "train.csv"), os.path.join(data_dir, "images"), processor)
    val_dataset = HAM10000Dataset(os.path.join(data_dir, "val.csv"), os.path.join(data_dir, "images"), processor)
    test_dataset = HAM10000Dataset(os.path.join(data_dir, "test.csv"), os.path.join(data_dir, "images"), processor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    train_losses=[]; val_losses=[]; train_accs=[]; val_accs=[]; best_val_acc=0; best_model_state=None
    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler, device, epoch+1)
        val_results = evaluate_model(model, val_loader, device, class_names)
        val_loss = val_results['loss']; val_acc = val_results['accuracy']
        train_losses.append(tr_loss); val_losses.append(val_loss); train_accs.append(tr_acc); val_accs.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc; best_model_state = model.state_dict().copy()
    if best_model_state is not None: model.load_state_dict(best_model_state)
    test_results = evaluate_model(model, test_loader, device, class_names)
    report = classification_report(test_results['labels'], test_results['predictions'], target_names=class_names, digits=4)
    print(report)
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(test_results['labels'], test_results['predictions'], class_names, title="ViT Fine-tuned Confusion Matrix")
    results_summary = {
        'model': model_name,'evaluation_type': 'fine_tuned',
        'test_accuracy': test_results['accuracy'],'test_f1_macro': test_results['f1_macro'],
        'test_f1_weighted': test_results['f1_weighted'],'test_auc_ovr': test_results['auc_ovr'],
        'test_auc_ovo': test_results['auc_ovo'],'test_loss': test_results['loss'],
        'best_val_accuracy': best_val_acc,'num_epochs': num_epochs,'learning_rate': learning_rate,
        'batch_size': batch_size,'num_test_samples': len(test_dataset),'class_names': class_names,
        'training_history': {'train_losses': train_losses,'val_losses': val_losses,'train_accs': train_accs,'val_accs': val_accs}
    }
    with open('finetuning_results.json', 'w') as f: json.dump(results_summary, f, indent=2)
    os.makedirs('./vit_ham10000_finetuned', exist_ok=True)
    model.save_pretrained('./vit_ham10000_finetuned'); processor.save_pretrained('./vit_ham10000_finetuned')
    print("Saved artifacts: finetuning_results.json, training_history.png, vit_training_progress.png, vit_fine-tuned_confusion_matrix.png")

if __name__ == "__main__":
    main()
