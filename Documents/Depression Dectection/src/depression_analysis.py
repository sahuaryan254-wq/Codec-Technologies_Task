from collections import Counter

EMOTION_RISK_WEIGHTS = {
    'Angry': 0.2,
    'Disgust': 0.15,
    'Fear': 0.25,
    'Happy': 0.0,
    'Sad': 0.8,
    'Surprise': 0.1,
    'Neutral': 0.3,
}

class DepressionAnalyzer:
    def __init__(self) -> None:
        pass

    def calculate_risk(self, emotion_history: list[str]) -> float:
        if not emotion_history:
            return 0.0
        total = sum(EMOTION_RISK_WEIGHTS.get(emotion, 0.0) for emotion in emotion_history)
        risk_score = min(100.0, (total / len(emotion_history)) * 100.0)
        return round(risk_score, 2)

    def detect_persistent_sadness(self, emotion_history: list[str], window_size: int = 10) -> bool:
        if len(emotion_history) < window_size:
            return False
        window = emotion_history[-window_size:]
        sad_count = Counter(window)['Sad']
        return sad_count / window_size >= 0.6
