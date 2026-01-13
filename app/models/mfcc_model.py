# app/models/mfcc_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Res2NetBlock(nn.Module):
    def __init__(self, in_p: int, planes: int, stride: int = 1, scale: int = 4, downsample: nn.Module | None = None):
        super().__init__()
        width = planes // scale

        self.conv1 = nn.Conv2d(in_p, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
                for _ in range(scale - 1)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(scale - 1)])

        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.se = SEBlock(planes)
        self.ds = downsample
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        spx = torch.split(out, out.size(1) // self.scale, dim=1)

        out_list = []
        s = None

        for i in range(self.scale - 1):
            if i == 0:
                s = spx[i]
            else:
                curr_s = spx[i]
                # stride로 인해 spatial이 달라질 수 있으니 안전하게 맞춤
                if s.shape[-2:] != curr_s.shape[-2:]:
                    s = F.interpolate(s, size=curr_s.shape[-2:], mode="nearest")
                s = s + curr_s

            s = self.convs[i](s)
            s = F.relu(self.bns[i](s), inplace=True)
            out_list.append(s)

        last_sp = spx[-1]
        if self.convs[0].stride[0] > 1:
            last_sp = F.avg_pool2d(last_sp, kernel_size=3, stride=self.convs[0].stride, padding=1)

        target_size = out_list[0].shape[-2:]
        for i in range(len(out_list)):
            if out_list[i].shape[-2:] != target_size:
                out_list[i] = F.interpolate(out_list[i], size=target_size, mode="nearest")
        if last_sp.shape[-2:] != target_size:
            last_sp = F.interpolate(last_sp, size=target_size, mode="nearest")

        out = torch.cat(out_list + [last_sp], dim=1)
        out = self.se(self.bn3(self.conv3(out)))

        if self.ds is not None:
            residual = self.ds(x)
        if out.shape[-2:] != residual.shape[-2:]:
            residual = F.interpolate(residual, size=out.shape[-2:], mode="nearest")

        return F.relu(out + residual, inplace=True)


class Res2Net50SE(nn.Module):
    """
    체크포인트(best_res2net50_se.pth) 구조와 동일하게:
      - 입력: (B, 40, T) or (B, T, 40) or (B, 1, 40, T)
      - 내부 conv1: in_channels=1
      - fc: Linear(512, 2)

    기본 forward 반환:
      - (B,) 단일 logit  (logit_spoof - logit_bonafide)
        => sigmoid(logit) == softmax(logits2)[:, 1]
    """

    def __init__(self, layers: list[int] = [3, 4, 6, 3]):
        super().__init__()
        self.in_p = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.fc = nn.Linear(512, 2)

    def _make_layer(self, p: int, blocks: int, stride: int = 1) -> nn.Sequential:
        ds = None
        if stride != 1 or self.in_p != p:
            ds = nn.Sequential(
                nn.Conv2d(self.in_p, p, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(p),
            )

        layers = [Res2NetBlock(self.in_p, p, stride=stride, downsample=ds)]
        self.in_p = p
        for _ in range(1, blocks):
            layers.append(Res2NetBlock(p, p))
        return nn.Sequential(*layers)

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        허용 입력:
          - (B, 40, T)
          - (B, T, 40)
          - (B, 1, 40, T)
        반환:
          - (B, 1, 40, T)
        """
        if x.dim() == 4:
            # (B, 1, 40, T) 형태 기대
            if x.size(1) != 1:
                raise ValueError(f"Expected channel dim=1 for 4D input, got shape={tuple(x.shape)}")
            return x

        if x.dim() != 3:
            raise ValueError(f"Expected 3D or 4D tensor, got shape={tuple(x.shape)}")

        # 3D: (B, 40, T) or (B, T, 40)
        if x.size(1) == 40:
            x = x.unsqueeze(1)  # (B, 1, 40, T)
            return x
        if x.size(2) == 40:
            x = x.transpose(1, 2).contiguous().unsqueeze(1)  # (B, 1, 40, T)
            return x

        raise ValueError(f"Cannot infer MFCC layout from shape={tuple(x.shape)} (need 40 on dim1 or dim2)")

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None, return_two_class_logits: bool = False) -> torch.Tensor:
        x = self._normalize_input(x)

        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x)), inplace=True), kernel_size=3, stride=2, padding=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B, 512)
        logits2 = self.fc(x)  # (B, 2)

        if return_two_class_logits:
            return logits2

        # 기존 "binary logit 1개" 파이프라인 호환용
        # sigmoid(logit_diff) == softmax(logits2)[:,1]
        logit_diff = logits2[:, 1] - logits2[:, 0]
        return logit_diff


# 프로젝트에서 기존에 MFCCBestModel 클래스를 import해서 쓰고 있을 가능성이 높아서 alias 제공
MFCCBestModel = Res2Net50SE