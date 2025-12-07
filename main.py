import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import os
from typing import List, Dict


class RuBERTSentimentAnalyzer:
    """Анализатор эмоционального окраса текстов на основе RuBERT модели"""
    
    def __init__(self, 
                 model_name: str = "cointegrated/rubert-tiny-sentiment-balanced",
                 max_length: int = 512,
                 batch_size: int = 32,
                 temperature: float = 1.0,
                 confidence_threshold: float = 0.5,
                 device: str = "auto"):
        """
        Инициализация анализатора с настраиваемыми параметрами
        
        Args:
            model_name: Название предобученной модели
            max_length: Максимальная длина токенизированной последовательности
            batch_size: Размер батча для обработки нескольких текстов
            temperature: Температура для софтмакса (влияет на уверенность)
            confidence_threshold: Порог уверенности для классификации
            device: Устройство для вычислений ('cpu', 'cuda', 'auto')
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        
        # Определение устройства
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Загружаю модель {model_name}...")
        print(f"Используется устройство: {self.device}")
        
        # Загрузка модели и токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Перемещение модели на нужное устройство
        self.model = self.model.to(self.device)
        
        # Переключаемся в режим инференса
        self.model.eval()
        
        # Маппинг лейблов
        self.label_mapping = {
            0: "NEGATIVE",  # Негативный
            1: "NEUTRAL",   # Нейтральный  
            2: "POSITIVE"   # Позитивный
        }
        
        print("Модель успешно загружена!")
        print(f"Параметры: max_length={max_length}, temperature={temperature}, threshold={confidence_threshold}")
    
    def update_parameters(self, **kwargs):
        """
        Обновление параметров модели во время работы
        
        Доступные параметры:
        - temperature: float
        - confidence_threshold: float
        - max_length: int
        - batch_size: int
        """
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)
                print(f"Параметр {param} обновлен до {value}")
            else:
                print(f"Неизвестный параметр: {param}")
                
    def get_parameters(self) -> Dict:
        """Получение текущих параметров модели"""
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "temperature": self.temperature,
            "confidence_threshold": self.confidence_threshold,
            "device": self.device
        }
        
    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Разбивает текст на фрагменты по количеству токенов
        
        Args:
            text: Исходный текст
            chunk_size: Размер фрагмента в токенах (оставляем запас от 512)
            
        Returns:
            Список текстовых фрагментов
        """
        # Токенизируем весь текст
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Если текст помещается в один фрагмент, возвращаем его как есть
        if len(tokens) <= self.max_length - 2:  # -2 для [CLS] и [SEP]
            return [text]
        
        # Разбиваем на фрагменты
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
        
        return chunks
    
    def predict_sentiment(self, text: str) -> Dict[str, any]:
        """
        Анализ эмоционального окраса одного текста
        
        Args:
            text: Текст для анализа
            
        Returns:
            Словарь с результатами анализа
        """
        # Токенизация
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=self.max_length
        )
        
        # Перемещение на нужное устройство
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Предсказание
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Применяем температуру для контроля уверенности
            scaled_logits = outputs.logits / self.temperature
            predictions = torch.nn.functional.softmax(scaled_logits, dim=-1)
            
        # Получаем вероятности для каждого класса
        probs = predictions.cpu().numpy()[0]
        predicted_class = np.argmax(probs)
        confidence = float(probs[predicted_class])
        
        # Проверяем порог уверенности
        if confidence < self.confidence_threshold:
            predicted_label = "UNCERTAIN"
        else:
            predicted_label = self.label_mapping[predicted_class]
        
        return {
            "text": text,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "probabilities": {
                "NEGATIVE": float(probs[0]),
                "NEUTRAL": float(probs[1]), 
                "POSITIVE": float(probs[2])
            },
            "is_uncertain": confidence < self.confidence_threshold
        }
    
    def predict_sentiment_with_chunking(self, text: str) -> Dict[str, any]:
        """
        Анализ эмоционального окраса с поддержкой чанкования для длинных текстов
        
        Args:
            text: Текст для анализа
            
        Returns:
            Словарь с результатами анализа
        """
        # Разбиваем текст на фрагменты
        chunks = self.chunk_text(text)
        
        # Показываем информацию о токенах
        total_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
        if len(chunks) > 1:
            print(f"📏 Текст содержит {total_tokens} токенов, разбит на {len(chunks)} фрагментов")
        
        # Если только один фрагмент, используем обычный метод
        if len(chunks) == 1:
            result = self.predict_sentiment(text)
            result["chunks_used"] = 1
            result["chunking_details"] = None
            return result
        
        # Анализируем каждый фрагмент
        chunk_results = []
        for i, chunk in enumerate(chunks):
            chunk_result = self.predict_sentiment(chunk)
            chunk_results.append(chunk_result)
            print(f"  Фрагмент {i+1}/{len(chunks)}: {chunk_result['predicted_label']} (уверенность: {chunk_result['confidence']:.3f})")
        
        # Агрегируем результаты
        final_result = self.aggregate_chunk_results(chunk_results, text)
        final_result["chunks_used"] = len(chunks)
        final_result["chunking_details"] = {
            "chunk_results": [
                {
                    "chunk_index": i,
                    "label": r["predicted_label"], 
                    "confidence": r["confidence"]
                } 
                for i, r in enumerate(chunk_results)
            ]
        }
        
        return final_result
    
    def aggregate_chunk_results(self, chunk_results: List[Dict], original_text: str) -> Dict[str, any]:
        """
        Агрегирует результаты анализа фрагментов
        
        Args:
            chunk_results: Список результатов анализа фрагментов
            original_text: Исходный текст
            
        Returns:
            Агрегированный результат
        """
        from collections import Counter
        
        # Собираем все предсказания (исключая UNCERTAIN)
        valid_predictions = []
        valid_confidences = []
        
        for result in chunk_results:
            if result["predicted_label"] != "UNCERTAIN":
                valid_predictions.append(result["predicted_label"])
                valid_confidences.append(result["confidence"])
        
        # Если все фрагменты неопределенные
        if not valid_predictions:
            avg_confidence = np.mean([r["confidence"] for r in chunk_results])
            return {
                "text": original_text,
                "predicted_label": "UNCERTAIN",
                "confidence": float(avg_confidence),
                "probabilities": {
                    "NEGATIVE": 0.33,
                    "NEUTRAL": 0.34, 
                    "POSITIVE": 0.33
                },
                "is_uncertain": True
            }
        
        # Подсчитываем голоса
        vote_counts = Counter(valid_predictions)
        most_common_label = vote_counts.most_common(1)[0][0]
        
        # Вычисляем среднюю уверенность для самого частого класса
        same_label_confidences = [
            conf for pred, conf in zip(valid_predictions, valid_confidences)
            if pred == most_common_label
        ]
        avg_confidence = float(np.mean(same_label_confidences))
        
        # Вычисляем агрегированные вероятности
        # Усредняем вероятности всех валидных фрагментов
        agg_probs = {"NEGATIVE": 0.0, "NEUTRAL": 0.0, "POSITIVE": 0.0}
        for result in chunk_results:
            if result["predicted_label"] != "UNCERTAIN":
                for label in agg_probs:
                    agg_probs[label] += result["probabilities"][label]
        
        # Нормализуем
        total_valid_chunks = len(valid_predictions)
        for label in agg_probs:
            agg_probs[label] /= total_valid_chunks
        
        # Проверяем порог уверенности
        if avg_confidence < self.confidence_threshold:
            final_label = "UNCERTAIN"
        else:
            final_label = most_common_label
        
        return {
            "text": original_text,
            "predicted_label": final_label,
            "confidence": avg_confidence,
            "probabilities": agg_probs,
            "is_uncertain": avg_confidence < self.confidence_threshold,
            "voting_details": {
                "votes": dict(vote_counts),
                "total_chunks": len(chunk_results),
                "valid_chunks": total_valid_chunks,
                "winner": most_common_label,
                "winner_votes": vote_counts[most_common_label]
            }
        }
    
    def validate_on_test_data(self, test_csv_path: str = "/home/deck/Desktop/models2/test_simple.csv"):
        """
        Валидация модели на размеченных тестовых данных
        
        Args:
            test_csv_path: Путь к CSV файлу с размеченными данными
            
        Returns:
            Словарь с результатами валидации
        """
        try:
            print(f"📋 Загружаю тестовые данные из {test_csv_path}...")
            df = pd.read_csv(test_csv_path)
            
            # Проверяем наличие нужных колонок
            if 'Текст новости' not in df.columns or 'Эмоциональный окрас' not in df.columns:
                raise ValueError("В CSV файле должны быть колонки 'Текст новости' и 'Эмоциональный окрас'")
            
            print(f"✅ Загружено {len(df)} размеченных новостей")
            
            # Анализируем каждую новость
            predictions = []
            true_labels = []
            detailed_results = []
            
            # Используем enumerate для надежного подсчета
            for i, (idx, row) in enumerate(df.iterrows()):
                text = row['Текст новости']
                true_label = row['Эмоциональный окрас']
                
                # Преобразуем текст в строку и проверяем на пустоту
                text = str(text).strip()
                true_label = str(true_label).strip()
                
                if not text or text == 'nan':
                    print(f"⚠️ Пропускаю новость {i+1}/{len(df)} - пустой текст")
                    continue
                
                print(f"Валидирую новость {i+1}/{len(df)}...")
                
                result = self.predict_sentiment_with_chunking(text)
                predicted_label = result['predicted_label']
                confidence = result['confidence']
                
                predictions.append(predicted_label)
                true_labels.append(true_label)
                
                # Сохраняем детальный результат
                is_correct = predicted_label == true_label
                detailed_results.append({
                    'text_preview': text[:100] + "..." if len(text) > 100 else text,
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'confidence': confidence,
                    'is_correct': is_correct
                })
                
                # Показываем результат
                status = "✅" if is_correct else "❌"
                print(f"  {status} Ожидалось: {true_label}, Получено: {predicted_label} (уверенность: {confidence:.3f})")
            
            # Вычисляем метрики
            from collections import Counter
            
            correct_predictions = sum(1 for i in range(len(predictions)) if predictions[i] == true_labels[i])
            total_predictions = len(predictions)
            accuracy = correct_predictions / total_predictions
            
            # Подсчет по классам
            true_counter = Counter(true_labels)
            pred_counter = Counter(predictions)
            
            # Confusion matrix (простая версия)
            confusion_data = {}
            for true_label in set(true_labels):
                confusion_data[true_label] = {}
                for pred_label in set(predictions):
                    confusion_data[true_label][pred_label] = sum(
                        1 for i in range(len(predictions)) 
                        if true_labels[i] == true_label and predictions[i] == pred_label
                    )
            
            validation_results = {
                'accuracy': accuracy,
                'correct_predictions': correct_predictions,
                'total_predictions': total_predictions,
                'true_distribution': dict(true_counter),
                'predicted_distribution': dict(pred_counter),
                'confusion_matrix': confusion_data,
                'detailed_results': detailed_results
            }
            
            self.print_validation_results(validation_results)
            return validation_results
            
        except FileNotFoundError:
            print(f"❌ Файл {test_csv_path} не найден!")
            return None
        except Exception as e:
            print(f"❌ Ошибка при валидации: {e}")
            return None
    
    def print_validation_results(self, results: Dict):
        """Красивый вывод результатов валидации"""
        print("\n" + "="*80)
        print("📊 РЕЗУЛЬТАТЫ ВАЛИДАЦИИ МОДЕЛИ")
        print("="*80)
        
        accuracy = results['accuracy']
        correct = results['correct_predictions']
        total = results['total_predictions']
        
        print(f"\n🎯 ОБЩАЯ ТОЧНОСТЬ: {accuracy:.3f} ({correct}/{total})")
        print(f"📈 Процент правильных предсказаний: {accuracy*100:.1f}%")
        
        print(f"\n📊 РАСПРЕДЕЛЕНИЕ ИСТИННЫХ МЕТОК:")
        for label, count in results['true_distribution'].items():
            percentage = (count / total) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
        
        print(f"\n🔮 РАСПРЕДЕЛЕНИЕ ПРЕДСКАЗАННЫХ МЕТОК:")
        for label, count in results['predicted_distribution'].items():
            percentage = (count / total) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
        
        print(f"\n📋 МАТРИЦА ОШИБОК:")
        header = "Истинное\\Предсказанное"
        print(f"{header:<25}", end="")
        all_labels = sorted(set(list(results['true_distribution'].keys()) + list(results['predicted_distribution'].keys())))
        for label in all_labels:
            print(f"{label:>10}", end="")
        print()
        
        for true_label in all_labels:
            print(f"{true_label:<25}", end="")
            for pred_label in all_labels:
                count = results['confusion_matrix'].get(true_label, {}).get(pred_label, 0)
                print(f"{count:>10}", end="")
            print()
        
        # Показываем ошибки
        errors = [r for r in results['detailed_results'] if not r['is_correct']]
        if errors:
            print(f"\n❌ ОШИБКИ КЛАССИФИКАЦИИ ({len(errors)} из {total}):")
            for i, error in enumerate(errors[:5], 1):  # Показываем первые 5 ошибок
                print(f"\n{i}. {error['text_preview']}")
                print(f"   Ожидалось: {error['true_label']}")
                print(f"   Получено: {error['predicted_label']} (уверенность: {error['confidence']:.3f})")
            
            if len(errors) > 5:
                print(f"\n... и еще {len(errors) - 5} ошибок")
        else:
            print(f"\n🎉 ВСЕ ПРЕДСКАЗАНИЯ ПРАВИЛЬНЫЕ!")

    def analyze_news_batch(self, news_texts: List[str]) -> List[Dict[str, any]]:
        """
        Анализ списка новостных текстов с поддержкой чанкования
        
        Args:
            news_texts: Список текстов новостей
            
        Returns:
            Список результатов анализа
        """
        results = []
        for i, text in enumerate(news_texts):
            print(f"Анализирую новость {i+1}/{len(news_texts)}...")
            # Используем чанкование для всех текстов
            result = self.predict_sentiment_with_chunking(text)
            results.append(result)
        
        return results
    
    def print_results(self, results: List[Dict[str, any]]):
        """Красивый вывод результатов анализа"""
        print("\n" + "="*80)
        print("РЕЗУЛЬТАТЫ АНАЛИЗА ЭМОЦИОНАЛЬНОГО ОКРАСА НОВОСТЕЙ")
        print("="*80)
        
        for i, result in enumerate(results, 1):
            print(f"\n📰 НОВОСТЬ {i}:")
            print(f"Текст: {result['text'][:250]}{'...' if len(result['text']) > 250 else ''}")
            print(f"🎯 Эмоциональный окрас: {result['predicted_label']}")
            print(f"🔮 Уверенность: {result['confidence']:.3f}")
            
            # Показываем информацию о чанковании
            if result.get('chunks_used', 1) > 1:
                print(f"📏 Использовано фрагментов: {result['chunks_used']}")
                if 'voting_details' in result:
                    votes = result['voting_details']['votes']
                    winner = result['voting_details']['winner']
                    print(f"📊 Голосование по фрагментам: {votes}")
                    print(f"🏆 Победивший класс: {winner} ({result['voting_details']['winner_votes']} голосов)")
            
            print(f"📊 Вероятности:")
            for label, prob in result['probabilities'].items():
                print(f"   {label}: {prob:.3f}")
            
            if result.get('is_uncertain', False):
                print("⚠️  Предупреждение: Низкая уверенность модели!")
        
        # Добавляем общую статистику по чанкованию
        chunked_count = sum(1 for r in results if r.get('chunks_used', 1) > 1)
        if chunked_count > 0:
            total_chunks = sum(r.get('chunks_used', 1) for r in results)
            avg_chunks = total_chunks / len(results)
            print(f"\n📈 СТАТИСТИКА ЧАНКОВАНИЯ:")
            print(f"Текстов с чанкованием: {chunked_count}/{len(results)}")
            print(f"Среднее количество фрагментов на текст: {avg_chunks:.1f}")


def load_news_from_csv(file_path: str, sample_size: int = 5) -> List[str]:
    """
    Загрузка новостей из CSV файла
    
    Args:
        file_path: Путь к CSV файлу
        sample_size: Количество новостей для выборки
        
    Returns:
        Список текстов новостей
    """
    try:
        print(f"📁 Загружаю новости из {file_path}...")
        df = pd.read_csv(file_path)
        
        # Ищем колонку с текстом (возможные названия)
        text_columns = ['text', 'title', 'content', 'news', 'article', 'body']
        text_column = None
        
        for col in text_columns:
            if col in df.columns:
                text_column = col
                break
        
        if text_column is None:
            # Если не нашли стандартные названия, берем первую текстовую колонку
            for col in df.columns:
                if df[col].dtype == 'object':
                    text_column = col
                    break
        
        if text_column is None:
            raise ValueError("Не удалось найти колонку с текстом в CSV файле")
        
        print(f"📄 Найдена колонка с текстом: '{text_column}'")
        print(f"📊 Всего новостей в файле: {len(df)}")
        
        # Убираем пустые значения и берем выборку
        texts = df[text_column].dropna().astype(str).tolist()
        
        if len(texts) < sample_size:
            sample_size = len(texts)
            print(f"⚠️ В файле меньше текстов чем запрошено, берем {sample_size}")
        
        # Берем случайную выборку
        import random
        sample_texts = random.sample(texts, sample_size)
        
        print(f"✅ Выбрано {len(sample_texts)} новостей для анализа")
        return sample_texts
        
    except FileNotFoundError:
        print(f"❌ Файл {file_path} не найден!")
        return []
    except Exception as e:
        print(f"❌ Ошибка при чтении файла: {e}")
        return []


def main():
    """Основная функция"""
    print("🚀 Анализатор эмоционального окраса новостей")
    print("=" * 50)
    
    # Выбор модели
    print("\n🤖 Выберите модель для использования:")
    print("  1. Базовая модель (cointegrated/rubert-tiny-sentiment-balanced)")
    print("  2. Дообученная модель (./rubert_finetuned)")
    
    model_choice = input("\nВведите номер модели (1/2) [по умолчанию 1]: ").strip()
    
    if model_choice == '2':
        if os.path.exists('./rubert_finetuned'):
            model_name = './rubert_finetuned'
            print("✅ Используется дообученная модель")
        else:
            print("⚠️  Дообученная модель не найдена, используется базовая")
            model_name = 'cointegrated/rubert-tiny-sentiment-balanced'
    else:
        model_name = 'cointegrated/rubert-tiny-sentiment-balanced'
        print("✅ Используется базовая модель")
    
    # Создаем анализатор с выбранной моделью
    print(f"\n⏳ Загрузка модели...")
    analyzer = RuBERTSentimentAnalyzer(
        model_name=model_name,
        temperature=1.0,
        confidence_threshold=0.5,
        max_length=512
    )
    print(f"✅ Модель загружена: {model_name}")
    
    # Интерактивный режим
    print(f"\n💬 Интерактивный режим:")
    print("Доступные команды:")
    print("  - Введите текст для анализа")
    print("  - 'params' - показать текущие параметры")
    print("  - 'model' - переключить модель")
    print("  - 'set <параметр> <значение>' - изменить параметр")
    print("  - 'validate' - запустить валидацию на test_simple.csv")
    print("  - 'exit' - выход")
    print("\nПримеры настройки:")
    print("  set temperature 0.8")
    print("  set confidence_threshold 0.7")
    print("  set max_length 128")
    
    while True:
        user_input = input("\n>>> ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'выход']:
            break
        elif user_input.lower() == 'params':
            params = analyzer.get_parameters()
            print("Текущие параметры:")
            for key, value in params.items():
                print(f"  {key}: {value}")
            print(f"  model: {analyzer.model_name}")
        elif user_input.lower() == 'model':
            print("\n🤖 Переключение модели:")
            print("  1. Базовая модель (cointegrated/rubert-tiny-sentiment-balanced)")
            print("  2. Дообученная модель (./rubert_finetuned)")
            
            new_model_choice = input("\nВведите номер модели (1/2): ").strip()
            
            if new_model_choice == '2':
                if os.path.exists('./rubert_finetuned'):
                    new_model_name = './rubert_finetuned'
                    print("✅ Переключение на дообученную модель...")
                else:
                    print("❌ Дообученная модель не найдена!")
                    continue
            elif new_model_choice == '1':
                new_model_name = 'cointegrated/rubert-tiny-sentiment-balanced'
                print("✅ Переключение на базовую модель...")
            else:
                print("❌ Неверный выбор")
                continue
            
            # Пересоздаем анализатор с новой моделью
            print("⏳ Загрузка модели...")
            old_params = analyzer.get_parameters()
            analyzer = RuBERTSentimentAnalyzer(
                model_name=new_model_name,
                temperature=old_params['temperature'],
                confidence_threshold=old_params['confidence_threshold'],
                max_length=old_params['max_length']
            )
            print(f"✅ Модель загружена: {new_model_name}")
        elif user_input.lower() == 'validate':
            print("\n🔍 Запуск валидации на размеченных данных...")
            validation_results = analyzer.validate_on_test_data()
            if validation_results:
                print(f"\n📊 Валидация завершена! Точность: {validation_results['accuracy']*100:.1f}%")
        elif user_input.lower().startswith('set '):
            parts = user_input.split()
            if len(parts) == 3:
                param_name = parts[1]
                try:
                    param_value = float(parts[2]) if '.' in parts[2] else int(parts[2])
                    analyzer.update_parameters(**{param_name: param_value})
                except ValueError:
                    print("Ошибка: значение должно быть числом")
            else:
                print("Использование: set <параметр> <значение>")
        elif user_input:
            # Проверяем, является ли ввод путём к файлу
            if user_input.endswith('.txt') and os.path.exists(user_input):
                print(f"\n📄 Чтение текста из файла: {user_input}")
                with open(user_input, 'r', encoding='utf-8') as f:
                    text = f.read()
                print(f"📏 Длина текста: {len(text)} символов")
            else:
                text = user_input
            
            result = analyzer.predict_sentiment_with_chunking(text)
            print(f"🎯 Результат: {result['predicted_label']} (уверенность: {result['confidence']:.3f})")
            
            # Показываем информацию о чанковании
            if result.get('chunks_used', 1) > 1:
                print(f"📏 Использовано фрагментов: {result['chunks_used']}")
                if 'voting_details' in result:
                    print(f"📊 Голосование: {result['voting_details']['votes']}")
            
            if result['is_uncertain']:
                print("⚠️  Низкая уверенность - результат помечен как UNCERTAIN")
    
    print("\n👋 Анализ завершен!")


if __name__ == "__main__":
    main()
