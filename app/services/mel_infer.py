# app/services/mel_infer.py

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

import librosa

from PIL import Image
from torchvision import transforms


@dataclass
class MelInferConfig:
    device: str = "cpu"

    # 오디오 설정
    sample_rate: int = 16000          # PCM이 어떤 샘플레이트인지 서버가 알아야 함
    segment_sec: float = 5.0          # 5초 단위로 자르거나 패딩

    # Mel Spectrogram 설정
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    fmin: int = 20
    fmax: int = 8000

    # 이미지 모델 입력 설정 (MobileNetV2 기본 입력)
    img_size: int = 224


class MelInfer:
    """
    mel_spectrogram_model.pt (torch.save(model, ...)) 로 저장된
    MobileNetV2 기반 분류 모델을 로드해서,
    PCM(int16) -> mel spectrogram 이미지 -> 모델 추론까지 수행한다.
    """

    def __init__(self, model_path: str, cfg: MelInferConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # 주의:
        # torch.save(model)로 저장한 전체 모델은 로드 시점에 torchvision/torch 버전 호환성이 중요할 수 있음
        # 가능하면 state_dict로 저장/로드가 더 안전하지만, 질문 조건에 따라 전체 모델 로드를 예시로 든다.
        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.eval()

        # MobileNetV2 (ImageNet pretrained) 기반이면 보통 아래 정규화를 사용한다.
        # 학습 때 다른 정규화를 썼다면 반드시 학습과 동일하게 맞춰야 한다.
        self.preprocess = transforms.Compose([
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.ToTensor(),  # [0,1], shape: (C,H,W)
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # 2-class 예시 (0: normal, 1: deepvoice/phishing)
        self.classes = {0: "normal", 1: "deepvoice"}

    def _pcm_i16_to_float32(self, audio_i16: np.ndarray) -> np.ndarray:
        """
        int16 PCM을 float32 파형(-1~1)으로 변환
        """
        if audio_i16.dtype != np.int16:
            audio_i16 = audio_i16.astype(np.int16)

        audio_f = audio_i16.astype(np.float32) / 32768.0

        # 혹시 NaN/Inf 방지
        audio_f = np.nan_to_num(audio_f, nan=0.0, posinf=0.0, neginf=0.0)
        return audio_f

    def _fix_length_5sec(self, wav: np.ndarray) -> np.ndarray:
        """
        5초 길이로 자르거나(앞부분), 부족하면 뒤를 0으로 패딩한다.
        """
        target_len = int(self.cfg.sample_rate * self.cfg.segment_sec)

        if wav.shape[0] > target_len:
            return wav[:target_len]
        if wav.shape[0] < target_len:
            pad = target_len - wav.shape[0]
            return np.pad(wav, (0, pad), mode="constant", constant_values=0.0)
        return wav

    def _wav_to_mel_image(self, wav: np.ndarray) -> Image.Image:
        """
        wav(float32, 1D) -> mel spectrogram(log scale) -> PIL Image(3채널)
        """
        # Mel Spectrogram (power)
        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=self.cfg.sample_rate,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            n_mels=self.cfg.n_mels,
            fmin=self.cfg.fmin,
            fmax=self.cfg.fmax,
            power=2.0,
        )

        # log scale(dB)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # 0~255로 정규화해서 이미지화
        # 학습 때 정규화 방식이 다르면 반드시 동일하게 맞춰야 한다.
        mel_min = mel_db.min()
        mel_max = mel_db.max()
        if mel_max - mel_min < 1e-6:
            mel_norm = np.zeros_like(mel_db, dtype=np.uint8)
        else:
            mel_norm = (mel_db - mel_min) / (mel_max - mel_min)
            mel_norm = (mel_norm * 255.0).astype(np.uint8)

        # (n_mels, time) -> 이미지로 만들 때 시각적으로 보통 y축이 위가 고주파가 되므로 뒤집기도 함
        # 학습 때 뒤집었는지 여부가 중요하므로, 여기서는 "뒤집지 않는" 기본만 둔다.
        gray = Image.fromarray(mel_norm, mode="L")

        # MobileNet 입력 맞추기: 3채널로 복제
        rgb = Image.merge("RGB", (gray, gray, gray))
        return rgb

    def predict_from_pcm_i16(self, audio_i16: np.ndarray) -> Dict[str, Any]:
        """
        PCM int16 numpy array -> phishing_score 등 결과 반환
        """
        if audio_i16.size == 0:
            raise ValueError("Empty PCM array")

        wav = self._pcm_i16_to_float32(audio_i16)
        wav = self._fix_length_5sec(wav)

        img = self._wav_to_mel_image(wav)
        x = self.preprocess(img).unsqueeze(0).to(self.device)  # (1,3,224,224)

        with torch.no_grad():
            logits = self.model(x)  # (1,2) 기대
            probs = F.softmax(logits, dim=1).squeeze(0)  # (2,)

        # class 1 확률을 "보이스피싱/딥보이스 점수"로 사용한다는 가정
        phishing_score = float(probs[1].item())
        pred_idx = int(torch.argmax(probs).item())

        return {
            "phishing_score": phishing_score,
            "pred_class": self.classes.get(pred_idx, str(pred_idx)),
            "probs": {
                "normal": float(probs[0].item()),
                "deepvoice": float(probs[1].item()),
            },
        }
