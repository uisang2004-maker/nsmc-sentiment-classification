# NSMC 영화 리뷰 감성 분류 (KcBERT 파인튜닝)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uisang2004-maker/nsmc-sentiment-classification/blob/main/nsmc_kcbert_classification.ipynb)

네이버 영화 리뷰 데이터셋(NSMC) 15만 건으로 한국어 사전학습 모델 **KcBERT**를 파인튜닝해 리뷰의 긍정/부정을 분류한 프로젝트입니다.
고수준 학습 도구(PyTorch Lightning, Hugging Face `Trainer`) 없이 **PyTorch 학습 루프를 직접 구현**했습니다.

## 저장소 구성

| 파일 | 내용 |
|---|---|
| [`nsmc_kcbert_classification.ipynb`](nsmc_kcbert_classification.ipynb) | 전체 파이프라인과 실행 결과 |
| [`from_scratch/nsmc_from_scratch.ipynb`](from_scratch/nsmc_from_scratch.ipynb) | 참고 코드 없이 처음부터 다시 구현한 버전 ([설명](#처음부터-다시-구현하기-from_scratch)) |
| [`notes/do_it_nlp_summary.md`](notes/do_it_nlp_summary.md) | 교재 1~4장 개념 정리 (토큰화, 셀프 어텐션, GPT와 BERT, 문서 분류 모델 구조) |

## 결과

| 데이터 | 샘플 수 | Loss | Accuracy |
|---|---:|---:|---:|
| Validation | 15,000 | 0.2359 | **90.16%** |
| Test | 49,997 | 0.2444 | **89.92%** |

- 1 epoch (4,219 step), Google Colab T4 GPU
- Validation과 Test 정확도 차이가 0.24%p로 작아, 검증 데이터가 실제 성능을 잘 대표함을 확인

## 구현 내용

| 단계 | 내용 |
|---|---|
| 데이터 | NSMC 원본(TSV)을 pandas로 직접 로드. 따옴표 파싱 오류를 막기 위해 `QUOTE_NONE` 적용, 빈 리뷰 5건 제거 |
| 데이터 분리 | train의 10%를 validation으로 분리하고, test는 학습 종료 후 한 번만 평가 |
| EDA | 레이블 균형 확인(부정 75,170 / 긍정 74,825), 리뷰 길이 분포(평균 35자, 최대 158자)로 `max_length=128` 결정 |
| 토큰화 | KcBERT WordPiece 토크나이저, 최대 128토큰으로 자르고 패딩 |
| 데이터셋 | `torch.utils.data.Dataset`을 상속한 `NSMCDataset`과 `DataLoader`로 배치 구성 |
| 모델 | `beomi/kcbert-base` (파라미터 108.9M) + 분류 헤드 |
| 학습 | AdamW, 선형 warmup 스케줄러(전체의 10%), 혼합 정밀도(AMP) + `GradScaler` |
| 평가 | `model.eval()` + `torch.no_grad()`로 loss와 accuracy 계산 |

### 학습 루프

```python
optimizer.zero_grad()                          # 1. 이전 기울기 초기화
with torch.autocast(device_type="cuda", dtype=torch.float16):
    logits = model(**batch).logits             # 2. 순전파
    loss = F.cross_entropy(logits, labels)     # 3. 손실 계산
scaler.scale(loss).backward()                  # 4. 역전파
scaler.step(optimizer)                         # 5. 가중치 업데이트
scaler.update()
scheduler.step()
```

## 하이퍼파라미터

| 항목 | 값 |
|---|---|
| Pretrained model | `beomi/kcbert-base` |
| Max length | 128 |
| Batch size | 32 |
| Learning rate | 5e-5 |
| Warmup ratio | 0.1 |
| Epochs | 1 |
| Seed | 42 |

## 실행 방법

1. 위의 **Open in Colab** 배지를 누릅니다.
2. Colab 메뉴에서 **런타임 → 런타임 유형 변경 → T4 GPU**를 선택합니다.
3. 셀을 위에서부터 차례로 실행합니다. 데이터와 모델은 자동으로 내려받습니다.

로컬에서 실행하려면 CUDA를 지원하는 GPU 환경에서 다음을 설치합니다.

```bash
pip install -r requirements.txt
```

## 처음부터 다시 구현하기 (`from_scratch/`)

[`from_scratch/nsmc_from_scratch.ipynb`](from_scratch/nsmc_from_scratch.ipynb)는 위 노트북을 한 줄씩 이해한 뒤, **참고 코드를 보지 않고 빈 노트북에서 전체 파이프라인을 다시 작성한 버전**입니다. 데이터 로드부터 Dataset, 학습 루프, 평가 함수까지 직접 작성했습니다.

| 데이터 | Loss | Accuracy |
|---|---:|---:|
| Validation | 0.2317 | **90.36%** |
| Test | 0.2424 | **90.04%** |

같은 설정과 시드로 학습한 첫 노트북(Test 89.92%)과 0.1%p 정도 차이가 나는데, GPU 연산(특히 혼합 정밀도)이 실행마다 완전히 같은 결과를 보장하지 않아 생기는 범위의 차이입니다.

### 다시 구현하며 찾은 버그

에러 없이 결과만 틀리게 만드는 버그들이라, 원리를 이해하지 않으면 발견하기 어려운 것들이었습니다.

| 버그 | 증상 | 원인과 해결 |
|---|---|---|
| 검증 데이터 분리 순서 | 검증 데이터가 학습 데이터에도 남아 **검증 정확도가 실제보다 높게 나옴** (에러 없음) | 검증 샘플을 뽑자마자 `reset_index`를 해서, `train_df.drop(val_df.index)`가 뽑힌 줄이 아닌 맨 앞 줄들을 지움. 원래 인덱스로 `drop`한 **뒤에** 인덱스를 정리하도록 순서 수정 |
| `test_df`에 학습 데이터 대입 | 테스트 평가가 **이미 학습한 데이터**로 이뤄짐 (에러 없음) | `test_df = train_df.dropna(...)` 오타. `test_df.dropna(...)`로 수정하고 개수(49,997)로 확인 |
| 검증 분리가 `if` 블록 안에 들어감 | 기본 설정에서 `val_df`가 만들어지지 않아 `NameError` | 선택적인 학습 데이터 축소(`train_size`)와 항상 실행돼야 하는 검증 분리를 별도 단계로 분리 |

### 배운 점

- 학습 루프의 각 단계(기울기 초기화 → 순전파 → 손실 계산 → 역전파 → 업데이트)가 무엇을 하는지, 손실 계산과 역전파가 다른 단계라는 점
- 데이터 분리에서는 **인덱스와 실행 순서**가 결과의 신뢰성을 좌우하며, 데이터 개수와 학습/검증 간 중복 여부를 직접 확인해야 한다는 점
- 코드를 읽고 이해한 것과 빈 화면에서 직접 짤 수 있는 것은 다르며, 직접 짜면서 막히는 곳이 아직 이해하지 못한 부분이라는 점

## 다음 단계

- 배치별 동적 패딩(`DataCollatorWithPadding`)으로 학습 속도 개선
- 중복 리뷰 제거 후 성능 비교
- epoch 수와 학습률 조정, `klue/roberta-base` 등 다른 사전학습 모델과 비교
- 오분류 사례 분석 (반어, 혼합 감성 리뷰)

## 참고

- 데이터: [NSMC (Naver Sentiment Movie Corpus)](https://github.com/e9t/nsmc)
- 모델: [KcBERT](https://huggingface.co/beomi/kcbert-base)
- 『Do it! BERT와 GPT로 배우는 자연어 처리』 4장 문서 분류 실습을 참고했습니다. 원 교재의 `ratsnlp` 패키지는 옛 버전 의존성 때문에 최신 Python에서 설치되지 않아, 최신 `transformers`와 순수 PyTorch로 다시 구현했습니다.

## 실행 환경

Python 3 (Google Colab) · torch 2.11.0 · transformers 5.22.0
