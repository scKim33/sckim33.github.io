---
title: "[Decision Transformer] Reinforcement Learning via Sequence Modeling"
last_modified_at: 2025-02-18
categories:
  - paper_review
tags:
  - Reinforcement Learning
  - Transformer
  - Meta
  - Google
excerpt: "Decision Transformer paper review"
use_math: true
classes: wide
---

> NeurIPS 2021 Poster. [[Paper](https://arxiv.org/abs/2106.01345)][[Site](https://sites.google.com/berkeley.edu/decision-transformer)]   
> Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe  
> 2 Jun 2021

## Summary
강화학습을 conditional sequence modeling으로 보아 문제를 정의한다. Decision Transformer 구조는 바람직한 reward를 얻도록 하는 action을 예측하도록 한다. 그 결과 dynamic programming 없이도 정책 학습이 가능하다. 이 논문에서는 Offline RL을 다룬다.

## Introduction

기존의 RL은 TD에서 bootstrapping이 불안정을 유발할 수 있고, feature reward를 discounting하게 되면 long-term에서의 성능이 저하될 수 있다. 또한 bellman backup이 희박한 보상 환경에서는 전파 속도가 떨어지고 노이즈와 같은 distractor signal에 취약하다.

트랜스포머 모델은 고차원 분포, 언어 모델의 zero-shot generalization이나 OOD image generation 같은 분야에서 효과적이다. 정책 학습을 DP 대신 sequence modeling으로 대체하게 되면 상태, 행동, 보상을 autoregressive 모델로 학습한다. 트랜스포머에서는 미래 보상을 고려한 상태-행동 sequence를 생성하기 때문에 sparse reward에서 RL의 단점을 보완할 수 있다. 또한 트랜스포머가 long-term dependencies를 잘 학습하고 일반화 및 전이 학습이 용이하기 때문에 그 점도 장점이 될 수 있다.


## Preliminaries

### Offline reinforcement learning
Offline RL에서는 환경과의 상호작용을 하지 않고, 고정되고 제한된 데이터셋을 사용하게 된다. 이러한 점이 agent로 하여금 exploring을 원활하게 수행하기 어렵도록 만들기 때문에 난이도 있는 세팅이라고 볼 수 있다.

### Transformers
GPT 아키텍쳐 기반으로, autoregressive generation을 위해 causal self-attention mask를 사용한다. 그리고 n개 token에 대한 summation/softmax 연산을 직전 token에 대한 연산으로 바꾼다.

## Method

<center>
<img src='{{"assets/images/Decision Transformer/dt1.png" | relative_url}}' style="max-width: 100%; width: auto;">
<figcaption style="text-align: center;">Model structure</figcaption>
</center>

### Trajectory representation
Trajectory에 들어가는 요소는 reward, state, action이 있다. 여기서 reward는 앞으로 받을 보상의 총합 $\hat{R}\_t = \sum\_{t' = t}^{T} r\_{t'}$을 사용한다.

$$
\tau = \left( \hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \dots, \hat{R}_T, s_T, a_T \right)
$$

목표로 하는 보상을 직접 설정할 수 있기 때문에($\hat{R}\_0$를 통해 설정) 테스트 환경에서 행동을 수행하여 얻게된 리워드를 빼주면 다음 $\hat{R}$을 구할 수 있다.

### Architecture
$K$ timestep에 대해 $3K$ 길이의 token이 나온다. 이걸 embedding으로 변환시켜 주는데 각각 linear layer나 convolutional layer를 통과시키고 layernorm 취해준다. 그리고 GPT model을 통과시켜 action을 예측한다.

### Training
Offline 학습을 한다. 데이터셋으로부터 길이 $K$의 sequence를 뽑아 미니배치를 구성한다. Cross-entropy loss나 MSE를 이용하여 input token이 $s\_t$일 때, $a\_t$를 예측하도록 한다. 이 논문에서는 state, reward까지는 예측하지 않았는데, 그것도 좋은 연구방향이 될 거라고 한다.


## Evaluations on Offline RL Benchmarks

<center>
<img src='{{"assets/images/Decision Transformer/dt2.png" | relative_url}}' style="max-width: 100%; width: auto;">
<figcaption style="text-align: center;">Results</figcaption>
</center>

## Discussion

### Does Decision Transformer perform behavior cloning on a subset of the data?

<center>
<img src='{{"assets/images/Decision Transformer/dt3.png" | relative_url}}' style="max-width: 100%; width: auto;">
<figcaption style="text-align: center;">D4RL results</figcaption>
</center>

에피소드 기준으로 높은 리워드를 받은 순으로 정렬하여 학습 데이터로 사용하는 것을 percentile behavior cloning(%BC)이라 한다. 데이터가 많은 경우와 적은 경우에 대해 상반된 결과를 보여준다. 데이터가 많이 존재하는 D4RL 환경에서는 %BC의 경우 대체로 Decision Transformer와 비슷한 성능을 보여준다.

<center>
<img src='{{"assets/images/Decision Transformer/dt4.png" | relative_url}}' style="max-width: 100%; width: auto;">
<figcaption style="text-align: center;">Atari results</figcaption>
</center>

하지만 데이터가 적은 Atari 환경에서는 %BC가 더 나쁜 성능을 보여준다. Decision Transformer는 데이터가 부족한 상황에서도 강력한 일반화 성능을 보여주며, 단순히 BC를 하는 것이 아님을 알 수 있다.


### How well does Decision Transformer model the distribution of returns?

<center>
<img src='{{"assets/images/Decision Transformer/dt5.png" | relative_url}}' style="max-width: 100%; width: auto;">
<figcaption style="text-align: center;">Conditioned target return results</figcaption>
</center>

Decision Transformer가 보상 분포를 잘 모델링하는지 확인하고자 한다. 만약 목표 보상을 설정했을 때 목표 보상에 비례하게끔 실제로 performance가 나온다면, decision transformer가 어떤 행동을 해야 얼마만큼의 보상을 획득하는지에 대한 이해가 잘 되어있음을 알 수 있다. 이를 실제로 비교해본 결과 목표 보상에 대해 performance가 어느정도 비례함을 확인했다. 데이터셋에 있는 최대 보상보다 더 높은 보상을 받을 수도 있어 extrapolation도 잘 이루어지는 것을 볼 수 있다.

### What is the benefit of using a longer context length?

<center>
<img src='{{"assets/images/Decision Transformer/dt6.png" | relative_url}}' style="max-width: 100%; width: auto;">
<figcaption style="text-align: center;">Context length results</figcaption>
</center>

더 긴 context를 제공하는 것이 더 나은 결과를 주는 것인지 확인하기 위해 $K=30$ 또는 $K=50$과 $K=1$을 비교한다. $K=1$의 경우 이전 정보를 전혀 알지 못하는 상태인데, 기존 강화학습에서는 frame을 stacking하여 입력값에 통으로 넣어 이를 해결한다. $K>1$가 Atari 환경에서 더 나은 성능을 보여주는 것을 확인하였다. 그 이유에 대해 긴 context로 학습이 transformer에게 어떤 policy가 특정 action을 생성했는지에 대해 식별하는 능력을 제공해주었다고 설명한다.


### Does Decision Transformer perform effective long-term credit assignment?

<center>
<img src='{{"assets/images/Decision Transformer/dt7.png" | relative_url}}' style="max-width: 100%; width: auto;">
<figcaption style="text-align: center;">Key-to-Door env results</figcaption>
</center> 


Long-term credit assignment(특정 행동이 미래 보상에 얼마나 기여했는가)를 평가하기 위해 3단계로 이루어진 환경에서 테스트한다. 1단계에서 열쇠를 집고, 3단계에서 문에 도달해야 점수를 얻는데, 2단계가 중간에 끼어 있기 때문에 long-term으로 credit assignment가 이루어지지 않으면 좋은 결과를 내기 어렵다. 데이터셋은 random walk를 사용했으며 %BC와 Decision Transformer가 좋은 결과를 보여준다.

### Can transformers be accurate critics in sparse reward settings?

<center>
<img src='{{"assets/images/Decision Transformer/dt8.png" | relative_url}}' style="max-width: 100%; width: auto;">
<figcaption style="text-align: center;">Key-to-Door env results</figcaption>
</center> 

Decision tranformer가 action 대신 reward를 출력하게 함으로써 reward의 distribution($p(\hat{R\_1})$)을 예측하도록 한다. 실제로 에피소드의 진행에 따라 case별로 보상을 예측한다. 더불어 attention weight이 특정 상태(열쇠를 집고, 문에 도달하는)에 높은 값으로 부여되어 있어 중요한 상태를 잘 파악하고 있다.

### Does Decision Transformer perform well in sparse reward settings?

<center>
<img src='{{"assets/images/Decision Transformer/dt9.png" | relative_url}}' style="max-width: 100%; width: auto;">
<figcaption style="text-align: center;">Delayed reward results</figcaption>
</center> 


TD 학습의 잘 알려진 단점은 보상을 잘 퍼뜨려 놓아야 한다는 점인데 이는 비현실적이다. D4RL 세팅에서 에피소드의 마지막에 보상을 몰아넣었을 때 (sparse reward setting), decision transformer는 TD 기법인 CQL보다 우수한 성능을 보인다. (Agonistic은 특정한 가정에 종속되지 않는 방법을 의미하는데, BC의 경우 리워드 상관없이 지도학습이니까 그렇게 쓴 것.)

### Why does Decision Transformer avoid the need for value pessimism or behavior regularization?

TD 기법의 경우 데이터셋만으로 가치함수를 근사하기 때문에 이에 따른 가치함수의 과대평가가 이루어질 수 있다. 이를 막기 위해 Q 값을 보수적으로 학습하거나(CQL) 정책을 정규화하는 방식을 취한다(BCQ, BRAC). 이러한 점에서 decision transformer는 자유로운 편이다.

### How can Decision Transformer benefit online RL regimes?

온라인 학습 또한 transformer의 과거 데이터 기반 행동 예측을 통해 적은 샘플로도 학습이 가능할 수 있다. Decision transformer는 '기억'하는 능력이 뛰어나므로 다양한 행동 패턴을 학습하고 이를 적용함으로써 기존 RL에서와 같이 exploration을 수행할 수 있다.

## Related Work

## Conclusion

언어/시퀀스 모델에 사용되는 구조가 RL에서도 적용될 수 있음을 보였다. Decision transformer는 지도 학습을 사용하였지만 더 큰 데이터셋에서는 자기 지도 학습을 통해 GPT같은 RL foundation model을 구성할 수도 있다. 보상 또한 스칼라 값 대신 분포를 예측할 수 있다. 행동 예측 뿐 아니라 상태를 예측할 수 있고 그렇게 되면 model-based RL과의 연관성도 생각할 수 있다.

해결해야 할 부분도 존재한다. 아직 MDP 세팅에서 transformer가 가질 수 있는 error 타입에 대해서 연구가 필요하다. 또한 학습 데이터에 의존하는 만큼 잘못된 학습 데이터가 모델의 출력에 미치는 영향에 대해서도 알아볼 필요가 있다.