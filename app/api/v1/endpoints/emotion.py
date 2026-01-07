import torch
import torchaudio
import os, zipfile
import soundfile as sf

LABELS = ["기쁨", "당황", "분노", "불안", "슬픔"]  # => 협박 + 불안등의 라벨링을 추가해야함 중요...

def load_emotion_model(torchscript_path: str, device: str = "cpu") -> torch.jit.ScriptModule:
    model = torch.jit.load(torchscript_path, map_location=device)
    model.eval()
    return model
    
def preprocess_audio(file_path: str, target_sr: int = 16000, target_sec: float = 5.0) -> torch.Tensor:
    # soundfile: (T, C) 형태로 읽힘
    audio, sr = sf.read(file_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio).T  # [C, T]
    
    # mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # resample (torchaudio는 load만 torchcodec 필요, resample은 그대로 써도 됨)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    # trim/pad to 5 sec
    max_len = int(target_sr * target_sec)  # 80000
    cur_len = waveform.size(1)

    if cur_len > max_len:
        waveform = waveform[:, :max_len]
    elif cur_len < max_len:
        waveform = torch.nn.functional.pad(waveform, (0, max_len - cur_len))

    return waveform.squeeze(0).unsqueeze(0).to(torch.float32)  # [1, 80000]
@torch.no_grad()
def infer_emotion_probs(model: torch.jit.ScriptModule, audio_path: str) -> dict:
    x = preprocess_audio(audio_path)              # [1, 80000]
    logits = model(x)                             # [1, num_labels]
    probs = torch.softmax(logits, dim=-1)[0]      # [num_labels]

    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
