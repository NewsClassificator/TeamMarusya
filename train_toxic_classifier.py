"""
Скрипт для обучения классификатора токсичных комментариев на rubert-tiny
Датасет: dataset.txt с метками __label__NORMAL, __label__INSULT, __label__THREAT, __label__OBSCENITY
"""

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


class ToxicCommentsDataset(Dataset):
    """Датасет для классификации токсичных комментариев"""
    
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
        
        # Токенизация
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


def load_dataset(file_path='dataset.txt'):
    """
    Загрузка датасета из файла формата:
    __label__CATEGORY текст комментария
    
    Поддерживает мультиметки (например: __label__INSULT,__label__THREAT)
    Для мультиметок берется первая метка
    """
    print(f"📂 Загрузка данных из {file_path}...")
    
    # Маппинг меток на числовые значения
    label_map = {
        'NORMAL': 0,      # Нормальные комментарии
        'INSULT': 1,      # Оскорбления
        'THREAT': 2,      # Угрозы
        'OBSCENITY': 3    # Непристойности
    }
    
    texts = []
    labels = []
    skipped = 0
    multilab = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Чтение файла"), 1):
            line = line.strip()
            if not line:
                skipped += 1
                continue
            
            # Парсинг формата __label__CATEGORY текст
            if not line.startswith('__label__'):
                print(f"⚠️  Строка {line_num} не начинается с __label__, пропускаем")
                skipped += 1
                continue
            
            # Разделяем метки и текст
            parts = line.split(None, 1)  # Разделяем по первому пробелу
            if len(parts) < 2:
                print(f"⚠️  Строка {line_num} не содержит текста, пропускаем")
                skipped += 1
                continue
            
            label_part, text = parts
            
            # Обработка мультиметок (берем первую)
            if ',' in label_part:
                multilab += 1
                label_part = label_part.split(',')[0]
            
            # Извлекаем имя метки
            label_name = label_part.replace('__label__', '').strip()
            
            # Проверяем, что метка известна
            if label_name not in label_map:
                print(f"⚠️  Неизвестная метка '{label_name}' в строке {line_num}, пропускаем")
                skipped += 1
                continue
            
            texts.append(text)
            labels.append(label_map[label_name])
    
    print(f"\n✅ Загружено записей: {len(texts)}")
    print(f"⚠️  Пропущено записей: {skipped}")
    if multilab > 0:
        print(f"🔀 Записей с несколькими метками: {multilab} (использована первая метка)")
    
    # Статистика по классам
    unique, counts = np.unique(labels, return_counts=True)
    label_names_reverse = {v: k for k, v in label_map.items()}
    
    print(f"\n📊 Распределение классов:")
    for label_id, count in zip(unique, counts):
        percentage = count / len(labels) * 100
        print(f"   {label_names_reverse[label_id]:12s}: {count:6d} ({percentage:5.1f}%)")
    
    return texts, labels, label_map


def compute_metrics(eval_pred):
    """Вычисление метрик"""
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


def train_toxic_classifier(
    dataset_file='dataset.txt',
    model_name='cointegrated/rubert-tiny',
    output_dir='./rubert_toxic_classifier',
    test_size=0.2,
    batch_size=8,
    num_epochs=5,
    learning_rate=2e-5,
    max_length=512
):
    """
    Обучение классификатора токсичных комментариев
    
    Args:
        dataset_file: путь к файлу с данными
        model_name: название базовой модели
        output_dir: папка для сохранения модели
        test_size: доля валидационной выборки
        batch_size: размер батча
        num_epochs: количество эпох
        learning_rate: скорость обучения
        max_length: максимальная длина текста
    """
    
    print("=" * 80)
    print("🚀 ОБУЧЕНИЕ КЛАССИФИКАТОРА ТОКСИЧНЫХ КОММЕНТАРИЕВ")
    print("=" * 80)
    print(f"📦 Базовая модель: {model_name}")
    print(f"📁 Выходная директория: {output_dir}")
    print(f"📏 Максимальная длина: {max_length} токенов")
    print(f"🎯 Batch size: {batch_size}")
    print(f"🔄 Эпохи: {num_epochs}")
    print(f"📈 Learning rate: {learning_rate}")
    print("=" * 80)
    
    # 1. Загрузка данных
    texts, labels, label_map = load_dataset(dataset_file)
    
    # 2. Разделение на train/validation
    print(f"\n📊 Разделение данных (test_size={test_size})...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    print(f"   Обучающая выборка: {len(train_texts)}")
    print(f"   Валидационная выборка: {len(val_texts)}")
    
    # 3. Загрузка модели и токенизатора
    print(f"\n🤖 Загрузка модели и токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4  # NORMAL, INSULT, THREAT, OBSCENITY
    )
    
    # 4. Создание датасетов
    print(f"\n📦 Создание датасетов...")
    train_dataset = ToxicCommentsDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = ToxicCommentsDataset(val_texts, val_labels, tokenizer, max_length)
    
    # 5. Параметры обучения
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
        report_to='none',
        warmup_steps=500,
        fp16=False  # CPU-only
    )
    
    # 6. Создание тренера
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # 7. Обучение
    print(f"\n🎓 Начинаем обучение...")
    print("=" * 80)
    trainer.train()
    
    # 8. Оценка
    print(f"\n📊 Оценка модели на валидационной выборке...")
    results = trainer.evaluate()
    print(f"\n📈 Результаты на валидации:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
    
    # 9. Детальный отчет
    print(f"\n📋 Детальный отчет по классам:")
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    
    label_names_reverse = {v: k for k, v in label_map.items()}
    target_names = [label_names_reverse[i] for i in sorted(label_names_reverse.keys())]
    
    print(classification_report(val_labels, pred_labels, target_names=target_names))
    
    # 10. Сохранение модели
    print(f"\n💾 Сохранение модели в {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Сохранение маппинга меток
    import json
    with open(f'{output_dir}/label_map.json', 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"📁 Модель сохранена в: {output_dir}")
    print(f"🎯 Финальная accuracy: {results['eval_accuracy']:.4f}")
    print(f"🎯 Финальный F1-score: {results['eval_f1']:.4f}")
    print("=" * 80)
    
    return trainer, results


if __name__ == "__main__":
    # Запуск обучения
    trainer, results = train_toxic_classifier(
        dataset_file='dataset.txt',
        model_name='cointegrated/rubert-tiny',
        output_dir='./rubert_toxic_classifier',
        test_size=0.2,
        batch_size=8,
        num_epochs=5,
        learning_rate=2e-5,
        max_length=512
    )
