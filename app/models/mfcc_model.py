# app/models/mfcc_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MFCCBestModel(nn.Module):
    """
    state_dict 키가 cnn.*, lstm.*, fc.* 인 모델을 로딩하기 위한 구조.

    - CNN: Conv2d + BN (+ ReLU/Pool은 파라미터 없어서 state_dict에 안 보임)
    - LSTM: input_size=832, hidden_size=128, num_layers=1, bidirectional=False (체크포인트 기준)
    - FC: 1차원 logit 출력

    주의:
    - CNN의 정확한 stride/pool 구조는 state_dict만으로 복원 불가.
    - 따라서 CNN 출력 후 adaptive pooling을 사용해서 LSTM input_size=832를 보장하는 방식.
    """

    def __init__(self, conv1_w: torch.Tensor, conv2_w: torch.Tensor, fc_in: int, fc_out: int):
        super().__init__()

        # conv weight shape: (out_ch, in_ch, kh, kw)
        o1, i1, kh1, kw1 = conv1_w.shape
        o2, i2, kh2, kw2 = conv2_w.shape

        self.cnn = nn.Sequential(
            nn.Conv2d(i1, o1, kernel_size=(kh1, kw1), stride=1, padding=(kh1 // 2, kw1 // 2), bias=True),  # cnn.0
            nn.BatchNorm2d(o1),  # cnn.1
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(i2, o2, kernel_size=(kh2, kw2), stride=1, padding=(kh2 // 2, kw2 // 2), bias=True),  # cnn.4
            nn.BatchNorm2d(o2),  # cnn.5
            nn.ReLU(),
        )

        self.lstm_input_size = 832  # checkpoint 확정
        self.hidden_size = 128

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.fc = nn.Linear(fc_in, fc_out)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        x: (B, T, 40)  # MFCC 시퀀스
        lengths: (B,)  # 유효 길이(패딩 제외)
        """
        # (B, T, 40) -> (B, 1, 40, T)  (Conv2d 입력)
        x = x.transpose(1, 2).unsqueeze(1)

        feat = self.cnn(x)  # (B, C, F, T')

        B, C, Freq, Time = feat.shape

        # LSTM input_size=832가 되어야 하므로 C*F == 832가 되도록 F를 강제
        if self.lstm_input_size % C != 0:
            raise RuntimeError(f"Cannot make LSTM input 832 from channels={C}.")
        target_F = self.lstm_input_size // C

        feat = F.adaptive_avg_pool2d(feat, (target_F, Time))  # (B, C, target_F, Time)

        # (B, C, F, T) -> (B, T, C*F)
        feat = feat.permute(0, 3, 1, 2).contiguous().view(B, Time, C * target_F)

        # Time 축이 바뀌었을 수 있어 lengths를 클램프
        lengths = torch.clamp(lengths, max=Time)

        packed = nn.utils.rnn.pack_padded_sequence(
            feat, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)

        h = h_n[-1]               # (B, hidden=128)
        logits = self.fc(h)       # (B, 1) or (B,) depending on fc_out
        return logits.squeeze(-1) # (B,)
