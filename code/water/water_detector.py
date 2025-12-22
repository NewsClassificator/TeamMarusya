"""
Анализатор "воды" в текстах на основе логистической регрессии
Использует модель logreg_water_model.pkl для определения водянистости текста
"""

import re
import joblib
import pandas as pd
import pymorphy3
from collections import Counter
from typing import Dict, Tuple, List
import nltk
from nltk.corpus import stopwords


class WaterDetector:
    """Детектор 'воды' в текстах на основе машинного обучения"""
    
    def __init__(self, model_path: str = "logreg_water_model.pkl"):
        """
        Инициализация детектора
        
        Args:
            model_path: Путь к файлу с обученной моделью
        """
        print(f"🔧 Инициализация WaterDetector...")
        
        # Загрузка модели
        try:
            self.model = joblib.load(model_path)
            print(f"✅ Модель загружена из {model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ Файл модели {model_path} не найден!")
        
        # Инициализация морфологического анализатора
        self.morph = pymorphy3.MorphAnalyzer()
        print("✅ Морфологический анализатор pymorphy3 инициализирован")
        
        # Загрузка стоп-слов
        try:
            self.ru_stopwords = set(stopwords.words("russian"))
        except LookupError:
            print("⏳ Загружаю стоп-слова...")
            nltk.download('stopwords', quiet=True)
            self.ru_stopwords = set(stopwords.words("russian"))
        print(f"✅ Загружено {len(self.ru_stopwords)} русских стоп-слов")
        
        # Названия признаков (должны соответствовать обучению модели)
        self.feature_names = [
            "readability_index", 
            "stopword_ratio", 
            "adj_ratio", 
            "adv_ratio", 
            "repetition_ratio"
        ]
        
        print("🎉 WaterDetector готов к работе!\n")
    
    def count_syllables(self, word: str) -> int:
        """
        Подсчет количества слогов в слове (упрощенно: 1 гласная = 1 слог)
        
        Args:
            word: Слово для анализа
            
        Returns:
            Количество слогов
        """
        vowels = 'аеёиоуыэюяАЕЁИОУЫЭЮЯ'
        count = 0
        for char in word:
            if char in vowels:
                count += 1
        return count
    
    def analyze_text_simple(self, text: str) -> Tuple[int, int, int]:
        """
        Базовый анализ текста
        
        Args:
            text: Текст для анализа
            
        Returns:
            Кортеж (количество предложений, количество слов, количество слогов)
        """
        # Разбиваем на предложения
        raw_sentences = re.split(r'[.!?…]+', text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        
        # Находим все русские слова
        words = re.findall(r'\b[а-яА-ЯёЁ]+\b', text)
        
        # Подсчитываем слоги
        syllables = 0
        for word in words:
            normal_word = self.morph.parse(word)[0].normal_form
            syllables += self.count_syllables(normal_word)
        
        return len(sentences), len(words), syllables
    
    def readability_index(self, text: str) -> Tuple[float, str]:
        """
        Вычисление индекса читаемости текста (формула Флеша для русского языка)
        
        Args:
            text: Текст для анализа
            
        Returns:
            Кортеж (числовой индекс, текстовая категория)
        """
        sentences, words, syllables = self.analyze_text_simple(text)
        
        if sentences == 0 or words == 0:
            return 0.0, "ТЕКСТ СЛИШКОМ КОРОТКИЙ"
        
        # Формула индекса читаемости Флеша (адаптированная для русского)
        index = 206.835 - 1.3 * (words / sentences) - 60.1 * (syllables / words)
        
        # Определяем уровень читаемости
        if index > 90:
            level = "ОЧЕНЬ ВЫСОКИЙ"
        elif index > 80:
            level = "ВЫСОКИЙ"
        elif index > 70:
            level = "ВЫШЕ СРЕДНЕГО"
        elif index > 60:
            level = "СРЕДНИЙ"
        elif index > 50:
            level = "НИЖЕ СРЕДНЕГО"
        elif index > 30:
            level = "НИЗКИЙ"
        else:
            level = "ОЧЕНЬ НИЗКИЙ"
        
        return round(index, 2), level
    
    def stopword_ratio(self, text: str) -> float:
        """
        Вычисление доли стоп-слов в тексте
        
        Args:
            text: Текст для анализа
            
        Returns:
            Доля стоп-слов (от 0 до 1)
        """
        words = re.findall(r'\b[а-яА-ЯёЁ]+\b', text.lower())
        
        if len(words) == 0:
            return 0.0
        
        stopword_count = sum(1 for word in words if word in self.ru_stopwords)
        ratio = stopword_count / len(words)
        
        return ratio
    
    def pos_ratios(self, text: str) -> Tuple[float, float]:
        """
        Вычисление долей прилагательных и наречий в тексте
        
        Args:
            text: Текст для анализа
            
        Returns:
            Кортеж (доля прилагательных, доля наречий)
        """
        words = re.findall(r'\b[а-яА-ЯёЁ]+\b', text)
        pos = Counter()
        
        # Определяем часть речи для каждого слова
        for w in words:
            p = self.morph.parse(w)[0].tag.POS
            pos[p] += 1
        
        total = sum(pos.values())
        if total == 0:
            return 0.0, 0.0
        
        # ADJF - полное прилагательное, ADJS - краткое прилагательное
        adj = pos.get("ADJF", 0) + pos.get("ADJS", 0)
        # ADVB - наречие
        adv = pos.get("ADVB", 0)
        
        return adj / total, adv / total
    
    def repetition_ratio(self, text: str) -> float:
        """
        Вычисление доли самого часто повторяющегося слова
        
        Args:
            text: Текст для анализа
            
        Returns:
            Доля самого частого слова
        """
        words = re.findall(r'\b[а-яА-ЯёЁ]+\b', text.lower())
        if not words:
            return 0.0
        
        counts = Counter(words)
        max_count = max(counts.values())
        
        return max_count / len(words)
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Извлечение всех признаков из текста
        
        Args:
            text: Текст для анализа
            
        Returns:
            Словарь с признаками
        """
        # Индекс читаемости
        readability, _ = self.readability_index(text)
        
        # Доля стоп-слов
        stopword_r = self.stopword_ratio(text)
        
        # Доли прилагательных и наречий
        adj_r, adv_r = self.pos_ratios(text)
        
        # Доля повторений
        rep_r = self.repetition_ratio(text)
        
        features = {
            "readability_index": readability,
            "stopword_ratio": stopword_r,
            "adj_ratio": adj_r,
            "adv_ratio": adv_r,
            "repetition_ratio": rep_r
        }
        
        return features
    
    def predict(self, text: str, return_proba: bool = False) -> Dict:
        """
        Предсказание наличия 'воды' в тексте
        
        Args:
            text: Текст для анализа
            return_proba: Возвращать ли вероятности классов
            
        Returns:
            Словарь с результатами анализа
        """
        # Извлекаем признаки
        features = self.extract_features(text)
        
        # Преобразуем в формат для модели
        X = pd.DataFrame([features])[self.feature_names]
        
        # Делаем предсказание
        prediction = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        
        # Формируем результат
        result = {
            "text": text[:200] + "..." if len(text) > 200 else text,
            "is_water": bool(prediction),
            "water_label": "ВОДА" if prediction == 1 else "НЕ ВОДА",
            "confidence": float(proba[prediction]),
            "features": features
        }
        
        if return_proba:
            result["probabilities"] = {
                "not_water": float(proba[0]),
                "water": float(proba[1])
            }
        
        return result
    
    def interpret_features(self, features: Dict[str, float]) -> Dict[str, str]:
        """
        Интерпретация признаков текста
        
        Args:
            features: Словарь с признаками
            
        Returns:
            Словарь с интерпретациями
        """
        interpretations = {}
        
        # Интерпретация индекса читаемости
        ri = features["readability_index"]
        if ri > 80:
            interpretations["readability"] = "очень легко читается"
        elif ri > 60:
            interpretations["readability"] = "нормально читается"
        elif ri > 40:
            interpretations["readability"] = "тяжеловато читается"
        else:
            interpretations["readability"] = "сложно читается"
        
        # Интерпретация стоп-слов
        sw = features["stopword_ratio"]
        if sw < 0.25:
            interpretations["stopwords"] = "плотный текст"
        elif sw < 0.35:
            interpretations["stopwords"] = "нормально"
        else:
            interpretations["stopwords"] = "подозрение на воду (много стоп-слов)"
        
        # Интерпретация прилагательных
        adj = features["adj_ratio"]
        if adj < 0.12:
            interpretations["adjectives"] = "фактология"
        elif adj < 0.18:
            interpretations["adjectives"] = "нейтрально"
        else:
            interpretations["adjectives"] = "возможная вода (много описаний)"
        
        # Интерпретация наречий
        adv = features["adv_ratio"]
        if adv < 0.03:
            interpretations["adverbs"] = "сухой текст"
        elif adv < 0.07:
            interpretations["adverbs"] = "нормально"
        else:
            interpretations["adverbs"] = "эмоциональная вода (много наречий)"
        
        # Интерпретация повторений
        rep = features["repetition_ratio"]
        if rep < 0.05:
            interpretations["repetitions"] = "хорошо (мало повторов)"
        elif rep < 0.1:
            interpretations["repetitions"] = "терпимо"
        else:
            interpretations["repetitions"] = "вода (много повторов)"
        
        return interpretations
    
    def analyze_text(self, text: str, detailed: bool = True) -> Dict:
        """
        Полный анализ текста с детальной информацией
        
        Args:
            text: Текст для анализа
            detailed: Включать ли детальную интерпретацию
            
        Returns:
            Словарь с полным анализом
        """
        result = self.predict(text, return_proba=True)
        
        if detailed:
            result["interpretations"] = self.interpret_features(result["features"])
        
        return result
    
    def print_analysis(self, result: Dict):
        """
        Красивый вывод результатов анализа
        
        Args:
            result: Результат анализа
        """
        print("\n" + "="*80)
        print("📝 АНАЛИЗ ТЕКСТА НА 'ВОДУ'")
        print("="*80)
        
        print(f"\n📄 Текст: {result['text']}")
        
        print(f"\n🎯 РЕЗУЛЬТАТ: {result['water_label']}")
        print(f"   Уверенность: {result['confidence']*100:.1f}%")
        
        if "probabilities" in result:
            probs = result["probabilities"]
            print(f"\n📊 Вероятности:")
            print(f"   Не вода: {probs['not_water']*100:.1f}%")
            print(f"   Вода: {probs['water']*100:.1f}%")
        
        print(f"\n📈 ПРИЗНАКИ:")
        features = result["features"]
        print(f"   Индекс читаемости: {features['readability_index']:.2f}")
        print(f"   Доля стоп-слов: {features['stopword_ratio']:.3f}")
        print(f"   Доля прилагательных: {features['adj_ratio']:.3f}")
        print(f"   Доля наречий: {features['adv_ratio']:.3f}")
        print(f"   Доля повторений: {features['repetition_ratio']:.3f}")
        
        if "interpretations" in result:
            print(f"\n💡 ИНТЕРПРЕТАЦИЯ:")
            interp = result["interpretations"]
            print(f"   Читаемость: {interp['readability']}")
            print(f"   Стоп-слова: {interp['stopwords']}")
            print(f"   Прилагательные: {interp['adjectives']}")
            print(f"   Наречия: {interp['adverbs']}")
            print(f"   Повторения: {interp['repetitions']}")
        
        print("="*80 + "\n")
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """
        Анализ списка текстов
        
        Args:
            texts: Список текстов для анализа
            
        Returns:
            Список результатов анализа
        """
        results = []
        for i, text in enumerate(texts, 1):
            print(f"Анализирую текст {i}/{len(texts)}...")
            result = self.analyze_text(text, detailed=True)
            results.append(result)
        
        return results
    
    def analyze_csv(self, csv_path: str, text_column: str = "text", 
                    output_path: str = None) -> pd.DataFrame:
        """
        Анализ текстов из CSV файла
        
        Args:
            csv_path: Путь к CSV файлу
            text_column: Название колонки с текстом
            output_path: Путь для сохранения результатов (опционально)
            
        Returns:
            DataFrame с результатами анализа
        """
        print(f"📁 Загружаю данные из {csv_path}...")
        df = pd.read_csv(csv_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Колонка '{text_column}' не найдена в CSV файле!")
        
        print(f"📊 Найдено {len(df)} текстов для анализа")
        
        # Анализируем каждый текст
        results = []
        for idx, text in enumerate(df[text_column], 1):
            if idx % 10 == 0:
                print(f"   Обработано {idx}/{len(df)}...")
            
            result = self.predict(text, return_proba=True)
            results.append(result)
        
        # Добавляем результаты в DataFrame
        df["is_water"] = [r["is_water"] for r in results]
        df["water_label"] = [r["water_label"] for r in results]
        df["confidence"] = [r["confidence"] for r in results]
        df["water_probability"] = [r["probabilities"]["water"] for r in results]
        
        # Добавляем признаки
        for feature in self.feature_names:
            df[feature] = [r["features"][feature] for r in results]
        
        print(f"✅ Анализ завершен!")
        print(f"\n📊 Статистика:")
        print(f"   Всего текстов: {len(df)}")
        print(f"   Вода: {df['is_water'].sum()} ({df['is_water'].sum()/len(df)*100:.1f}%)")
        print(f"   Не вода: {(~df['is_water']).sum()} ({(~df['is_water']).sum()/len(df)*100:.1f}%)")
        
        if output_path:
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"💾 Результаты сохранены в {output_path}")
        
        return df


def main():
    """Основная функция для интерактивного использования"""
    print("🚀 Детектор 'воды' в текстах")
    print("=" * 50)
    
    # Создаем детектор
    detector = WaterDetector()
    
    print("\n💬 Интерактивный режим:")
    print("Доступные команды:")
    print("  - Введите текст для анализа")
    print("  - 'file <путь>' - анализ текста из файла")
    print("  - 'csv <путь>' - анализ CSV файла")
    print("  - 'example' - анализ примеров")
    print("  - 'exit' - выход")
    
    while True:
        user_input = input("\n>>> ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'выход']:
            break
        
        elif user_input.lower().startswith('file '):
            file_path = user_input[5:].strip()
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                result = detector.analyze_text(text, detailed=True)
                detector.print_analysis(result)
            except FileNotFoundError:
                print(f"❌ Файл {file_path} не найден!")
            except Exception as e:
                print(f"❌ Ошибка: {e}")
        
        elif user_input.lower().startswith('csv '):
            csv_path = user_input[4:].strip()
            try:
                df = detector.analyze_csv(csv_path, output_path=csv_path.replace('.csv', '_analyzed.csv'))
                print(f"\n📋 Первые строки результатов:")
                print(df[['water_label', 'confidence', 'water_probability']].head())
            except Exception as e:
                print(f"❌ Ошибка: {e}")
        
        elif user_input.lower() == 'example':
            examples = [
                "Исследование показало увеличение производительности на 25% при внедрении новой системы.",
                "Это невероятно удивительное и потрясающе замечательное решение, которое действительно очень сильно помогает достигать поставленных целей и реализовывать планы."
            ]
            
            for i, example in enumerate(examples, 1):
                print(f"\n{'='*80}")
                print(f"ПРИМЕР {i}:")
                result = detector.analyze_text(example, detailed=True)
                detector.print_analysis(result)
        
        elif user_input:
            result = detector.analyze_text(user_input, detailed=True)
            detector.print_analysis(result)
    
    print("\n👋 Анализ завершен!")


if __name__ == "__main__":
    main()
