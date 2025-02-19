---
title: "[InstructGPT] Training language models to follow instructions with human feedback"
last_modified_at: 2025-02-18
categories:
  - paper_review
tags:
  - LLM
  - InstructGPT
  - RLHF
  - OpenAI
excerpt: "InstructGPT paper review"
use_math: true
classes: wide
---

> NeurIPS 2022 Accept. [[Paper](https://arxiv.org/abs/2203.02155)]   
> Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe  
> 4 Mar 2022

## Summary

## Related work

## Methods and experimental details

### Dataset

OpenAI API를 통해 제출된 text prompts를 이용한다. 다만 GPT-3에 제출된 prompt들은 지시문 형태에 적절하지 않은 경우가 많아 초기에는 labeler가 prompt를 작성하도록 하였다. Prompt는 user ID 당 200개로 제한하고, train, validation, test 데이터 간에는 같은 user ID가 섞이지 않도록 한다. 개인 식별이 가능한 데이터는 필터링한다. 데이터는 3가지 성격의 prompting을 포함한다.
- Plain: 라벨러들이 아무 task에 대해 자유롭게 prompt 생성. 모델은 다양한 질문 유형을 학습.
- Few-shot: 특정 지시에 대해 해당하는 질문/답변 쌍을 생성. 모델은 질문/답변 쌍의 패턴을 학습.
- User-based: API를 이용하려는 사용자가 사용하려는 기능을 토대로 라벨러가 prompt 생성. 모델이 더욱 실용적인 답변을 생성하도록 학습.

3가지 모델에 대해 학습한다. SFT 모델에 대해서는 라벨러들이 작성한 input prompt/output의 supervised learning이 이루어진다. RM 모델에서는 모델의 답변을 라벨러들이 점수를 매기고 이를 예측한다. output/reward의 supervised learning이 이루어진다. PPO 모델에서는 주어진 질문(s)에 대해 답변을 하고(a) 이를 RM 모델을 통해 평가(r=f(a))하고 이를 바탕으로 강화학습이 이루어진다. 각 모델에 대해 13K, 33K, 31K의 데이터셋으로 학습한다.

### Models

<center>
<img src='{{"assets/images/InstructGPT/igpt1.png" | relative_url}}' style="max-width: 100%; width: auto;">
<figcaption style="text-align: center;">Learning process</figcaption>
</center>

#### SFT
GPT-3를 fine-tune한 모델이다. 1 epoch만으로도 오버피팅이 발생함에도, epoch가 더 진행됨에 따라 RM 스코어, 인간 평가 점수가 모두 높아지는 특성이 있다.
#### RM
SFT 학습이 끝난 후 마지막 unembedding layer를 제거하고 scalar prediction layer를 붙여 reward를 예측하도록 한다. 175B는 불안정하여 6B를 사용한다. 학습에는 $K=4$에서 $K=9$ 사이의 응답 개수에 대해 비교한 데이터를 사용한다. 더 나은 답변을 고르는 방식으로는 총 $\binom{K}{2}$의 페어가 나온다. 문제는 각 쌍을 단일 데이터로 사용하는 경우 high-correlation 때문에 과적합이 발생하기 쉽다. 따라서 단일 배치로 데이터를 구성하여 처리하니 계산량이 감소하고 과적합을 막는다. Loss function은 선호되는 쪽과 선호되지 않는 쪽의 차이를 최대화하는 방향으로 이루어져있다.

$$
\text{loss} (\theta) = -\frac{1}{\left(\frac{K}{2}\right)} \mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[ \log \left( \sigma \left( r_{\theta} (x, y_w) - r_{\theta} (x, y_l) \right) \right) \right]
$$

#### RL
PPO의 objective function을 사용한다.

$$
\text{objective} (\phi) = \mathbb{E}_{(x,y) \sim D_{\phi}^{\text{RL}}} \left[ r_{\theta}(x,y) - \beta \log \left( \frac{\pi_{\phi}^{\text{RL}}(y \mid x)}{\pi^{\text{SFT}}(y \mid x)} \right) \right] + \gamma \mathbb{E}_{x \sim D_{\text{pretrain}}} \left[ \log \left( \pi_{\phi}^{\text{RL}}(x) \right) \right]
$$


## Results

<center>
<img src='{{"assets/images/InstructGPT/igpt2.png" | relative_url}}' style="max-width: 100%; width: auto;">
<figcaption style="text-align: center;">Result</figcaption>
</center>

Labeler 그룹을 5개로 나누어 5-fold cross validation을 진행하였다. InstrctGPT의 경우 학습 데이터를 만든 training labeler 뿐만 아니라 이에 포함되지 않은 held-out labeler(검증 라벨러)의 선호도 또한 잘 만족한다. 즉 과적합되지 않고 선호도를 일반화시킨다는 것이다.

<center>
<img src='{{"assets/images/InstructGPT/igpt3.png" | relative_url}}' style="max-width: 100%; width: auto;">
<figcaption style="text-align: center;">Public NLP Dataset result</figcaption>
</center>

Public NLP 데이터셋을 바탕으로 GPT-3를 fine tuning 시킨 모델의 경우(FLAN, T0), InstructGPT보다 낮은 성능을 보여준다. 이는 instruction에 적절한 prompt가 선호도에 중요한 역할을 함을 보여준다.

<center>
<img src='{{"assets/images/InstructGPT/igpt4.png" | relative_url}}' style="max-width: 100%; width: auto;">
<figcaption style="text-align: center;">TruthfulQA Dataset result</figcaption>
</center>

InstructGPT는 사실과 일치하는 응답을 요구하는 TruthfulQA 데이터셋에서도 GPT-3보다 좋은 성능을 보여준다.

<center>
<img src='{{"assets/images/InstructGPT/igpt5.png" | relative_url}}' style="max-width: 100%; width: auto;">
<figcaption style="text-align: center;">Metadata result</figcaption>
</center>

InstructGPT는 Instruction+QA 방식의 prompt를 사용하여 모델이 모르는 정보에 대해서는 답변하지 않도록 학습한다. 이에 환각이 줄어드는 현상을 보인다.

<center>
<img src='{{"assets/images/InstructGPT/igpt6.png" | relative_url}}' style="max-width: 100%; width: auto;">
<figcaption style="text-align: center;">RealToxicityPrompts result</figcaption>
</center>

RealToxicityPrompts 데이터셋을 사용하였고, perspective API를 통해 toxicity를 평가한다. Safe하고 respectful한 output을 내라고 지시받았을 때(respectful prompt), toxicity가 감소하는 모습을 보인다.

<center>
<img src='{{"assets/images/InstructGPT/igpt7.png" | relative_url}}' style="max-width: 100%; width: auto;">
<figcaption style="text-align: center;">Winogender and CrowS-Pairs result</figcaption>
</center>

Winogender, CrowS-Pairs 데이터셋을 사용하여 성별, 인종, 직업에 대해 편향된 응답을 생성하는지 확인하는 task를 진행한다. 답변의 무작위성이 높야아(엔트로피가 높아야) 편향되지 않은 것이다. 프롬프트가 없을 때는 GPT-3와 동일한 수준의 편향을 보인다. 특징적인 부분은 "respectful prompt"를 지시받았을 때에도 오히려 편향성이 증가한다. 모델이 더 확신을 가지고 응답하게 되면 편향된 답변을 줄이기 어렵다. 결과적으로, 편향성은 줄이지 못했다.

InstructGPT의 경우 public NLP 데이터셋을 사용할 때 성능 저하가 발생하는데 이를 alignment tax라 명명한다. 이를 해결하기 위해 PPO 학습 중 GPT-3의 사전 훈련 데이터를 혼합하여 업데이트하는 pretraing mix를 추가적으로 하면 성능이 좀 나아진다 (PPO-ptx).

## Discussion

### Implications for alignment research
Alignment research(인간의 목표나 가치, 윤리 기준에 맞게 학습시키는 것)의 측면에서 이 연구가 주는 교훈은 다음과 같다.
- Pretraining에 소모되는 비용에 비해 model alignment에 들어가는 비용이 상대적으로 적다. GPT3 pretraining가 3,640 petaflops/s-days, SFT가 4.9 petaflops/s-days, PPO-ptx가 60 petaflops/s-days인 것을 비교하면 효율적인 학습이 가능하다.
- '지시문을 따르는 것'이라는 환경을 잘 일반화한다. 즉, 지도 학습에 표시되지 않은 지시문에 대해서도 좋은 성능을 보여준다.
- Fine tuning으로 인한 성능 저하 문제를 해결한다. 인간의 지시를 따르도록 fine tuning 되었지만 만약 성능이 나빠진다면, 그것을 fine tuning에 대한 성능 저하, 즉, alignment tax라고 한다. 결국에 인간의 의도를 따르도록 하는 방향으로 앞으로의 연구가 진행되어야 한다는 입장에서 RLHF는 낮은 tax alignment로 좋은 해결책이 된다.
- 기존 alignmnet 연구들은 이론적이거나 작은 도메인에서만 이루어졌거나 (실제 사용자 데이터가 아닌) public NLP dataset을 사용하였다. InstructGPT의 경우, 실제 사용자들이 사용하는 제품에서 데이터를 얻어 활용함으로써 real world에 적합하다.
  
### Who are we aligning to?
모델의 alignment가 결국에는 편향되기 마련이다. 이번 연구에서도 Training labeler의 선호도, OpenAI researcher의 선호도, API를 사용하는 유저의 선호도 각각의 영향을 받는다. 모든 사람의 선호를 반영하는 것은 불가능할 뿐더러 기업, 개발자, 유저, 사회의 이해관계가 엮여 있기 때문에 이는 Alignment가 가진 자체적인 한계라 볼 수 있다. 최선은 alignment를 기술적인 문제로 보는 시각 뿐만 아니라 윤리적이고 사회적인 문제이기도 함을 인지하고 AI가 편향되지 않도록 연구 진행이 필요함을 강조한다.

### Limitations
결국에 labeler도 자신의 가치관에 따른 prompt/응답 pair를 만들어내기 때문에 그러한 방향으로 편향이 생길 수 있다. 논문에서는 40명의 contractors를 고용했다고 밝히는데 대부분 영어로 지시문을 만드는 primarily English-speaking 이 사람들이 GPT 이용자 전체를 대변할 수는 없기 때문이다.

또한 이용자의 instruction을 따른다는 점에서, 악의를 가진 누군가는 real world에 해로운 방향의 instruction을 제시할 수 있다.