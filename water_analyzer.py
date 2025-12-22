from typing import Dict, Tuple, List
import re
import warnings
import joblib
import pandas as pd
import pymorphy3
from collections import Counter
import nltk
from nltk.corpus import stopwords

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class WaterAnalyzer:
    
    def __init__(self, model_path: str = "logreg_water_model.pkl"):
        self.model = joblib.load(model_path)
        self.morph = pymorphy3.MorphAnalyzer()
        
        try:
            self.ru_stopwords = set(stopwords.words("russian"))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            self.ru_stopwords = set(stopwords.words("russian"))
        
        self.feature_names = [
            "readability_index", 
            "stopword_ratio", 
            "adj_ratio", 
            "adv_ratio", 
            "repetition_ratio"
        ]
    
    def count_syllables(self, word: str) -> int:
        vowels = 'аеёиоуыэюяАЕЁИОУЫЭЮЯ'
        return sum(1 for char in word if char in vowels)
    
    def analyze_text_simple(self, text: str) -> Tuple[int, int, int]:
        raw_sentences = re.split(r'[.!?…]+', text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        words = re.findall(r'\b[а-яА-ЯёЁ]+\b', text)
        
        syllables = 0
        for word in words:
            normal_word = self.morph.parse(word)[0].normal_form
            syllables += self.count_syllables(normal_word)
        
        return len(sentences), len(words), syllables
    
    def readability_index(self, text: str) -> float:
        sentences, words, syllables = self.analyze_text_simple(text)
        
        if sentences == 0 or words == 0:
            return 0.0
        
        index = 206.835 - 1.3 * (words / sentences) - 60.1 * (syllables / words)
        return round(index, 2)
    
    def stopword_ratio(self, text: str) -> float:
        words = re.findall(r'\b[а-яА-ЯёЁ]+\b', text.lower())
        
        if len(words) == 0:
            return 0.0
        
        stopword_count = sum(1 for word in words if word in self.ru_stopwords)
        return stopword_count / len(words)
    
    def pos_ratios(self, text: str) -> Tuple[float, float]:
        words = re.findall(r'\b[а-яА-ЯёЁ]+\b', text)
        pos = Counter()
        
        for w in words:
            p = self.morph.parse(w)[0].tag.POS
            pos[p] += 1
        
        total = sum(pos.values())
        if total == 0:
            return 0.0, 0.0
        
        adj = pos.get("ADJF", 0) + pos.get("ADJS", 0)
        adv = pos.get("ADVB", 0)
        
        return adj / total, adv / total
    
    def repetition_ratio(self, text: str) -> float:
        words = re.findall(r'\b[а-яА-ЯёЁ]+\b', text.lower())
        if not words:
            return 0.0
        
        counts = Counter(words)
        return max(counts.values()) / len(words)
    
    def extract_features(self, text: str) -> Dict[str, float]:
        readability = self.readability_index(text)
        stopword_r = self.stopword_ratio(text)
        adj_r, adv_r = self.pos_ratios(text)
        rep_r = self.repetition_ratio(text)
        
        return {
            "readability_index": readability,
            "stopword_ratio": stopword_r,
            "adj_ratio": adj_r,
            "adv_ratio": adv_r,
            "repetition_ratio": rep_r
        }
    
    def predict(self, text: str, return_proba: bool = False) -> Dict:
        features = self.extract_features(text)
        X = pd.DataFrame([features])[self.feature_names].values
        
        prediction = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        
        result = {
            "text": text,
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
        interpretations = {}
        
        ri = features["readability_index"]
        if ri > 80:
            interpretations["readability"] = "очень легко читается"
        elif ri > 60:
            interpretations["readability"] = "нормально читается"
        elif ri > 40:
            interpretations["readability"] = "тяжеловато читается"
        else:
            interpretations["readability"] = "сложно читается"
        
        sw = features["stopword_ratio"]
        if sw < 0.25:
            interpretations["stopwords"] = "плотный текст"
        elif sw < 0.35:
            interpretations["stopwords"] = "нормально"
        else:
            interpretations["stopwords"] = "подозрение на воду"
        
        adj = features["adj_ratio"]
        if adj < 0.12:
            interpretations["adjectives"] = "фактология"
        elif adj < 0.18:
            interpretations["adjectives"] = "нейтрально"
        else:
            interpretations["adjectives"] = "возможная вода"
        
        adv = features["adv_ratio"]
        if adv < 0.03:
            interpretations["adverbs"] = "сухой текст"
        elif adv < 0.07:
            interpretations["adverbs"] = "нормально"
        else:
            interpretations["adverbs"] = "эмоциональная вода"
        
        rep = features["repetition_ratio"]
        if rep < 0.05:
            interpretations["repetitions"] = "хорошо"
        elif rep < 0.1:
            interpretations["repetitions"] = "терпимо"
        else:
            interpretations["repetitions"] = "вода"
        
        return interpretations
    
    def analyze(self, text: str, detailed: bool = True) -> Dict:
        result = self.predict(text, return_proba=True)
        
        if detailed:
            result["interpretations"] = self.interpret_features(result["features"])
        
        return result
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        return [self.analyze(text, detailed=True) for text in texts]
    
    def analyze_csv(self, csv_path: str, text_column: str = "text", 
                    output_path: str = None) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found")
        
        results = []
        for text in df[text_column]:
            result = self.predict(text, return_proba=True)
            results.append(result)
        
        df["is_water"] = [r["is_water"] for r in results]
        df["water_label"] = [r["water_label"] for r in results]
        df["confidence"] = [r["confidence"] for r in results]
        df["water_probability"] = [r["probabilities"]["water"] for r in results]
        
        for feature in self.feature_names:
            df[feature] = [r["features"][feature] for r in results]
        
        if output_path:
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        return df
