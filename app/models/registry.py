from app.core.config import settings
from app.models.stt_whisper import WhisperSTT
from app.models.image_classifier import ImageClassifier
from app.models.object_detector import ObjectDetector
from app.models.text_embedder import TextEmbedder

class ModelRegistry:
    """
    모델 싱글톤/캐시 레지스트리.
    - 로딩은 startup에서 1회
    - 각 endpoint는 여기서 꺼내만 씀
    """
    stt = None
    classifier = None
    detector = None
    embedder = None

    @classmethod
    def load_all(cls):
        device = settings.DEVICE  # 보통 android는 'cpu'
        cls.stt = WhisperSTT(settings.STT_MODEL_PATH, device=device)
        cls.classifier = ImageClassifier(settings.CLS_MODEL_PATH, device=device)
        cls.detector = ObjectDetector(settings.DET_MODEL_PATH, device=device)
        cls.embedder = TextEmbedder(settings.EMB_MODEL_PATH, device=device)
