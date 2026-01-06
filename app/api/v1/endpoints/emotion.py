import torch
import torchaudio

LABELS = ["기쁨", "당황", "분노", "불안", "슬픔"]  # config.json id2label 순서 :contentReference[oaicite:1]{index=1}

def load_emotion_model(torchscript_path: str) -> torch.jit.ScriptModule:
    # 동적 양자화(trace) 모델이면 보통 CPU에서 돌리는 걸 권장
    model = torch.jit.load(torchscript_path, map_location="cpu")
    model.eval()
    return model

def preprocess_audio(
    file_path: str,
    target_sr: int = 16000,
    target_sec: float = 5.0
) -> torch.Tensor:
    """
    return: torch.FloatTensor shape [1, 80000]
    """
    waveform, sr = torchaudio.load(file_path)  # [C, T]

    # mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # resample
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)

    # trim/pad to 5 sec
    max_len = int(target_sr * target_sec)  # 80000
    cur_len = waveform.size(1)

    if cur_len > max_len:
        waveform = waveform[:, :max_len]
    elif cur_len < max_len:
        pad_len = max_len - cur_len
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))

    # TorchScript 입력 형태: [1, 80000] float32
    x = waveform.squeeze(0).unsqueeze(0).to(torch.float32)
    return x

@torch.no_grad()
def infer_emotion_probs(model: torch.jit.ScriptModule, audio_path: str) -> dict:
    x = preprocess_audio(audio_path)              # [1, 80000]
    logits = model(x)                             # [1, num_labels]
    probs = torch.softmax(logits, dim=-1)[0]      # [num_labels]

    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
