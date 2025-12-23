
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
from tqdm import tqdm
import os


class SentimentDataset(Dataset):

    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,  
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_data(json_file='train.json'):
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Маппинг меток
    label_map = {
        'negative': 0,
        'neutral': 1,
        'positive': 2
    }
    
    texts = []
    labels = []
    skipped = 0
    
    for item in tqdm(data, desc="Обработка данных"):
        if 'text' not in item or 'sentiment' not in item:
            skipped += 1
            continue
        
        text = str(item['text']).strip()
        sentiment = item['sentiment'].lower().strip()

        if not text or text == 'nan':
            skipped += 1
            continue

        if sentiment not in label_map:
            skipped += 1
            continue
        
        texts.append(text)
        labels.append(label_map[sentiment])

    
    unique, counts = np.unique(labels, return_counts=True)
    label_names = {v: k for k, v in label_map.items()}
    for label_id, count in zip(unique, counts):
        print(f"   {label_names[label_id].upper()}: {count} ({count/len(labels)*100:.1f}%)")
    
    return texts, labels, label_map


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    

    accuracy = accuracy_score(labels, predictions)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_model(
    train_file='train.json',
    model_name='cointegrated/rubert-tiny-sentiment-balanced',
    output_dir='./rubert_finetuned',
    test_size=0.2,
    batch_size=16,
    num_epochs=3,
    learning_rate=2e-5,
    max_length=512
):

    

    texts, labels, label_map = load_data(train_file)


    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3  # negative, neutral, positive
    )
    

    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, max_length)
        training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=2,
        report_to='none',  # Отключаем wandb и прочее
        warmup_steps=500,
        fp16=False  # CPU-only, поэтому False
    )
    

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    trainer.train()
    
    results = trainer.evaluate()
    for key, value in results.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
    
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    label_names = {v: k.upper() for k, v in label_map.items()}
    target_names = [label_names[i] for i in sorted(label_names.keys())]
    print(classification_report(val_labels, pred_labels, target_names=target_names))

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(f'{output_dir}/label_map.json', 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    
    
    return trainer, results


if __name__ == "__main__":

    trainer, results = train_model(
        train_file='train.json',
        output_dir='./rubert_finetuned',
        test_size=0.2,
        batch_size=8, 
        num_epochs=3,
        learning_rate=2e-5,
        max_length=512  
    )
