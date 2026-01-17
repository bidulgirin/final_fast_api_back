# 통화 실시간 탐지 문서

## 1) 흐름 요약 (의도 포함)

이 실시간 탐지는 "오디오 신호의 빠른 위험 점수"와 "STT 기반 텍스트 위험 점수"를 결합해, 지연을 최소화하면서도 경고 민감도를 확보하려는 구조입니다. 의도는 크게 두 가지입니다.

- **저지연**: 오디오(MFCC/MEL)만으로도 빠르게 1차 판단을 내리되,
- **정밀도 보강**: STT가 들어오면 텍스트 기반 위험 신호를 추가해 경고 품질을 높임.

결과적으로 **빠르게 반응하되, 텍스트 신호가 있으면 더 보수적으로 판단**하는 설계입니다.

## 2) 주요 코드 설명 (의도 포함)

### 2-1. 앱 시작 시 모델 1회 로드
File: `app/api/v1/endpoints/real_time_check.py`

```python
async def startup_load_models():
    ...
    mfcc_infer = MFCCInfer(...)
    mel_infer = MelBestInfer(...)
    text_infer = TextInfer(TextInferConfig(...))
    stt_infer = STTInfer(STTInferConfig(...))
```

Why: 모델 로드는 매우 무겁기 때문에, 요청마다 로드하면 지연이 커집니다. 앱 시작 시 1회 로드해 재사용하려는 의도입니다.

### 2-2. 오디오 모델 점수 계산 및 1차 융합
File: `app/api/v1/endpoints/real_time_check.py`

```python
mfcc_result = mfcc_infer.predict_from_pcm_i16(audio_i16)
mel_result = mel_infer.predict_from_pcm_i16(audio_i16)
audio_fused = fuse_scores(mfcc_score, mel_score, w_mfcc=0.5, w_mel=0.5)
```

Why: MFCC/MEL 두 모델을 평균 가중으로 합쳐 편향을 줄이고, 빠른 리스크 점수를 얻기 위한 구성입니다.

### 2-3. STT 수행 (threadpool) + 텍스트 위험 추론
File: `app/api/v1/endpoints/real_time_check.py`

```python
stt_text = await asyncio.wait_for(
    run_in_threadpool(stt_infer.transcribe_from_pcm_i16, audio_i16, 16000),
    timeout=3.0,
)
...
await stt_store.add_text(call_id, stt_text.strip())
buffered = await stt_store.get_last_texts(call_id, n=text_infer.cfg.buffer_size)
text_payload = text_infer.predict(buffered)
```

Why: STT는 무겁기 때문에 이벤트 루프를 막지 않도록 threadpool에서 실행합니다. 또한 실시간 특성상 “최근 N개 발화”로 문맥을 만들기 위해 버퍼를 사용합니다.

### 2-4. 최종 점수 융합 및 경고 판단
File: `app/api/v1/endpoints/real_time_check.py`

```python
final_fused = audio_fused if text_payload is None else fuse_three(audio_fused, text_risk, w_audio=0.8, w_text=0.2)
...
if final_fused >= 0.85:
    should_alert = True
```

Why: 오디오는 항상 들어오므로 기본값이며, 텍스트가 있을 때만 텍스트 신호로 위험을 상향하도록 설계되어 있습니다. 즉, 지연 없이 경고를 낼 수 있으면서도 텍스트가 오면 더 보수적으로 판단합니다.

### 2-5. STT 버퍼 저장소
File: `app/services/stt_store.py`

```python
class STTBufferStore:
    ...
    async def add_text(self, call_id: str, text: str) -> None:
        ...
        if len(st.texts) > self.max_keep:
            st.texts = st.texts[-self.max_keep :]
```

Why: call_id 단위로 최근 텍스트만 유지해 메모리를 제한하고, 최신 문맥만 사용하려는 의도입니다.

### 2-6. 통화 점수 누적 및 최종 결정
File: `app/services/vp_store.py`

```python
final_score = 0.7 * mean_score + 0.3 * max_score
flag = final_score >= 0.5
```

Why: 평균 점수로 안정성을 확보하면서, 순간적으로 높은 위험 구간도 반영하기 위해 max를 일부 섞습니다.

## 3) 코드 주석 추가

아래 파일들에 실시간 탐지 의도를 명확히 하기 위한 주석을 추가했습니다.
- `app/api/v1/endpoints/real_time_check.py`
- `app/services/stt_infer.py`
- `app/services/stt_store.py`
- `app/services/vp_store.py`

## 4) 개선사항 / 보완점

- **타임아웃/실패 대응 강화**: STT 타임아웃 시 빈 텍스트로 계속 진행하는데, STT 실패 비율을 로그/메트릭으로 추적해야 원인 분석이 가능합니다.
- **텍스트/오디오 가중치 튜닝**: `w_audio=0.8, w_text=0.2`는 고정값입니다. 실제 통화 품질, 텍스트 신뢰도에 따라 동적으로 조정하는 정책이 필요합니다.
- **연속 발화에 대한 적응**: 최근 N개만 보는 구조이므로 통화 길이가 길어질수록 초반 맥락이 사라집니다. 중요한 키워드가 이전에 등장했을 경우 누적 히스토리 반영 방식을 고려할 수 있습니다.
- **모델 로드 안정성**: `startup_load_models()`에서 모델 로드가 실패하면 실시간 API는 503을 반환합니다. 재시도 전략(백오프) 또는 헬스체크용 엔드포인트가 있으면 운영에 유리합니다.
- **STT/오디오 결과 간 타임스탬프 정합성**: 현재는 chunk 단위로 오디오/텍스트를 단순 결합합니다. 실제 실시간 통화에서는 시간 동기화가 경고 품질에 영향을 줄 수 있습니다.

## 5) 전체 요약

실시간 통화 탐지는 오디오 기반 점수(MFCC/MEL)로 빠르게 위험 신호를 만들고, STT 텍스트가 들어오면 위험도를 보강하는 구조입니다. STT는 threadpool로 처리해 지연을 줄이고, 최근 N개 텍스트만 버퍼링해 실시간성/메모리 한계를 맞춥니다. 최종 점수는 오디오에 더 큰 비중을 두되 텍스트가 있으면 보수적으로 상향하는 방식이며, 운영 관점에서는 STT 실패 추적, 가중치 동적 조정, 재시도 정책이 개선 포인트입니다.
