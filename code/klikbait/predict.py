"""
Скрипт для определения кликбейтов с использованием обученной модели
"""
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch


class ClickbaitDetector:
    """Детектор кликбейтных заголовков"""
    
    def __init__(self, model_path="my_awesome_model"):
        """
        Инициализация детектора
        
        Args:
            model_path: путь к сохраненной модели
        """
        import os
        
        # Если указана директория с checkpoint'ами, используем последний
        if os.path.isdir(model_path) and not os.path.exists(os.path.join(model_path, "config.json")):
            checkpoints = [d for d in os.listdir(model_path) if d.startswith("checkpoint-")]
            if checkpoints:
                # Берем checkpoint с максимальным номером
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
                model_path = os.path.join(model_path, latest_checkpoint)
                print(f"Используется checkpoint: {latest_checkpoint}")
        
        print(f"Загрузка модели из {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1  # CPU
        )
        print("Модель загружена!")
    
    def predict(self, text):
        """
        Предсказание для одного заголовка
        
        Args:
            text: заголовок новости
            
        Returns:
            dict: {"label": "кликбейт"/"не кликбейт", "score": 0.95}
        """
        result = self.classifier(text)[0]
        return result
    
    def predict_batch(self, texts):
        """
        Предсказание для списка заголовков
        
        Args:
            texts: список заголовков
            
        Returns:
            list: список предсказаний
        """
        return self.classifier(texts)
    
    def is_clickbait(self, text, threshold=0.5):
        """
        Проверка является ли заголовок кликбейтом
        
        Args:
            text: заголовок новости
            threshold: порог вероятности
            
        Returns:
            bool: True если кликбейт
        """
        result = self.predict(text)
        return result['label'] == 'кликбейт' and result['score'] >= threshold


def main():
    parser = argparse.ArgumentParser(description="Определение кликбейтных заголовков")
    parser.add_argument("text", nargs="*", help="Текст заголовка для анализа")
    parser.add_argument("-f", "--file", help="Файл с заголовками (по одному на строку)")
    parser.add_argument("-m", "--model", default="my_awesome_model", help="Путь к модели")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Порог для классификации")
    
    args = parser.parse_args()
    
    # Создаем детектор
    detector = ClickbaitDetector(args.model)
    
    # Определяем источник текстов
    texts = []
    if args.file:
        # Из файла
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    elif args.text:
        # Из аргументов командной строки
        texts = [' '.join(args.text)]
    else:
        # Интерактивный режим
        print("\n=== Интерактивный режим ===")
        print("Введите заголовок (или 'exit' для выхода):\n")
        while True:
            try:
                text = input("> ").strip()
                if text.lower() in ['exit', 'quit', 'q']:
                    break
                if not text:
                    continue
                    
                result = detector.predict(text)
                is_clickbait = result['label'] == 'кликбейт'
                confidence = result['score'] * 100
                
                print(f"\n📊 Результат: {result['label'].upper()}")
                print(f"   Уверенность: {confidence:.1f}%")
                
                if is_clickbait:
                    print("   🚨 Это кликбейт!\n")
                else:
                    print("   ✅ Нормальный заголовок\n")
                    
            except KeyboardInterrupt:
                print("\n\nВыход...")
                break
        return
    
    # Обработка списка текстов
    print(f"\nОбработка {len(texts)} заголовков...\n")
    results = detector.predict_batch(texts)
    
    for text, result in zip(texts, results):
        is_clickbait = result['label'] == 'кликбейт'
        confidence = result['score'] * 100
        
        icon = "🚨" if is_clickbait else "✅"
        print(f"{icon} [{result['label'].upper():12}] ({confidence:5.1f}%) {text}")
    
    # Статистика
    clickbait_count = sum(1 for r in results if r['label'] == 'кликбейт')
    print(f"\n📈 Статистика: {clickbait_count}/{len(texts)} кликбейтов ({clickbait_count/len(texts)*100:.1f}%)")


if __name__ == "__main__":
    main()
