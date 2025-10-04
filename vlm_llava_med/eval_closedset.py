#!/usr/bin/env python3
import json, argparse, numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def read_options(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', required=True, help='JSONL from LLaVA outputs')
    ap.add_argument('--option_list', required=True, help='txt: each canonical option per line')
    ap.add_argument('--out', default='metrics_llava.json')
    args = ap.parse_args()

    options = read_options(args.option_list)
    opt2idx = {o:i for i,o in enumerate(options)}

    y_true, y_pred, y_prob = [], [], []
    with open(args.preds, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if 'gold' in r:
                gt = r['gold'].strip()
                if gt not in opt2idx: 
                    continue
                y_true.append(opt2idx[gt])
            elif 'label_idx' in r:
                y_true.append(int(r['label_idx']))
            else:
                continue

            if 'logprobs' in r and isinstance(r['logprobs'], dict):
                scores = np.full(len(options), -1e9, dtype=np.float32)
                for k,v in r['logprobs'].items():
                    if k in opt2idx:
                        scores[opt2idx[k]] = float(v)
                s = scores - scores.max()
                probs = np.exp(s) / np.exp(s).sum()
                y_prob.append(probs.tolist())
                y_pred.append(int(np.argmax(probs)))
            else:
                pt = r.get('pred_text', '').strip()
                if pt in opt2idx:
                    idx = opt2idx[pt]
                    y_pred.append(idx)
                    onehot = np.zeros(len(options), dtype=np.float32); onehot[idx]=1
                    y_prob.append(onehot.tolist())
                else:
                    y_pred.append(0)
                    y_prob.append((np.ones(len(options))/len(options)).tolist())

    y_true = np.array(y_true); y_pred = np.array(y_pred); y_prob = np.array(y_prob)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    try:
        auc_ovr = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except ValueError:
        auc_ovr = 0.0

    out = {
        'accuracy': float(acc),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'auc_ovr': float(auc_ovr),
        'num_samples': int(len(y_true))
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
