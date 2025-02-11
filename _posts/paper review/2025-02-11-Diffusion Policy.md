---
title: "[Diffusion Policy] Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
last_modified_at: 2025-02-11
categories:
  - paper_review
tags:
  - Diffusion
  - Imitation learning
excerpt: "Diffusion Policy paper review"
use_math: true
classes: wide
---

> Robotics: Science and Systems 2023. [[Paper](https://arxiv.org/abs/2303.04137)] [[Site](https://diffusion-policy.cs.columbia.edu//)]  
> Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song  
> 7 Mar 2023

## Summary

Diffusion policy는 다양한 해법이 존재하는 action distribution을 효과적으로 다루고, 고차원 action spaces에서도 적절히 동작하며, 학습 시에도 안정적으로 동작한다. 이를 위해 receding horizon control, visual conditioning, time-series diffusion transformer를 사용한다.

## Introduction

<center>
<img src='{{"assets/images/Diffusion Policy/diff1.png" | relative_url}}' style="max-width: 100%; width: auto;">
<figcaption style="text-align: center;"></figcaption>
</center>

DDPM의 방식을 따라 visual observation condition이 주어지면 action-score gradient를 추론하게 된다. 이 방식이 주는 장점은 3가지가 있다.
- Expressing multimodal action distributions: Langevin dynamics sampling을 통해 임의의 정규화 가능한 확률 분포를 표현할 수 있다. 즉, multimodal action distribution을 표현할 수 있다.
- High-dimensional output space: 이미지 생성 모델에서도 많이 사용되는 diffusion이니만큼 고차원 output도 잘 표현한다.
- Stable training: Energy-based policy는 negative sample을 요구한다는 점에서 학습 안정성이 떨어지는 문제가 있는데, diffusion은 energy function의 gradient를 학습함으로써 안정적이다.

아울러 논문의 주요 기여는 다음과 같다.
- Closed-loop action sequences: receding-horizon control을 통해 closed-loop에서 지속적으로 re-plan이 가능하도록 한다.
- Visual conditioning: visual observations가 joint가 아닌 conditioning의 역할을 한다. K번의 denoising 과정에서 매번 이미지가 입/출력으로 주어질 필요가 없어지게 되어 계산량이 감소하고 real-time으로 동작하게 한다.
- Time-series diffusion transformer: 기존 CNN based model에서 관찰되는 over-smoothing effects를 최소화하고 좋은 성능을 내도록 한다.

## Diffusion policy formulation

Diffusion policy에서는 에너지 함수 $E$를 최소화하기 위해 $E$의 gradient를 diffusion model을 이용하여 구한다. DDPM의 방식을 채용하여 노이즈가 많이 낀 $x^k$로부터 denoising 과정을 거쳐 최종적으로 $x^0$ (action trajectory)을 계산하게 된다.

$$
\begin{aligned}
\mathbf{x^{k-1}} &= \mathbf{x^k} - \gamma \nabla E(\mathbf{x^k}) \\
&= \alpha (\mathbf{x}^k - \gamma \varepsilon_{\theta} (\mathbf{x}^k, k) + \mathcal{N}(0, \sigma^2 I))
\end{aligned}
$$

Cost function은 다음과 같이 noise 자체를 잘 예측하도록 학습된다.

$$
\mathcal{L} = \text{MSE}(\varepsilon^k, \varepsilon_{\theta} (\mathbf{x}^0 + \varepsilon^k, k))
$$

학습이 끝나게 되면 위에 있는 공식을 통해 action trajectory를 계산하게 된다. DDPM과의 차이점은 output이 이미지가 아닌 robot action이라는 점, denoising 과정에서 observation $O_t$를 condition으로 넣어준다는 점이 있다. Observation의 경우 이전 $T_o$ step만큼의 observation history를 넣어주고 $T_p$ step만큼의 action을 예측한다. 이 때, $T_a$ step은 replanning 없이 action을 수행한다. 이를 종합하면

$$
\Lambda_i^{k-1} = \alpha (\Lambda_i^k - \gamma \varepsilon_{\theta} (\mathbf{O}_i, \Lambda_i^k, k) + \mathcal{N}(0, \sigma^2 I))\\
\mathcal{L} = \text{MSE}(\varepsilon^k, \varepsilon_{\theta} (\mathbf{O}_i, \Lambda_i^0 + \varepsilon^k, k))
$$

## Key design decisions

<center>
<img src='{{"assets/images/Diffusion Policy/diff2.png" | relative_url}}' style="max-width: 100%; width: auto;">
<figcaption style="text-align: center;"></figcaption>
</center>

### Network architecture options
Diffusion policy에 사용되는 두 가지 네트워크 구조를 소개한다.
- CNN based diffusion policy\\
  observation의 conditioning은 FiLM conditioning을 사용한다.

  $$
  \text{FiLM}(\mathbf{x}) = \gamma(\mathbf{c}) \cdot \mathbf{x} + \beta(\mathbf{c})
  $$
  
  Condition $c$에 observation이 들어가고, feature $x$에 action embedding이 들어간다.
  CNN based 방식은 단순하고 성능이 잘 나오지만 action sequence가 빠르게 변화해야 하는 경우 성능이 떨어지는 단점이 있다.
- Time-series diffusion transformer\\
  CNN 모델의 over-smoothing 문제를 해결하기 위해 minGPT에서 사용된 구조를 따라 transformer 기반 DDPM을 사용한다. 

### Visual encoder

ResNet-18을 backbone으로 사용한다. 다만 global average pooling을 spatial softmax pooling으로 변경하여 spatial information을 잃지 않도록 하고, batchnorm을 groupnorm(channel-wise normalization)으로 바꿔 안정적인 학습이 되도록 한다. 그 이유는 batch 크기가 작을 때 batchnorm을 사용하면 평균과 분산의 흐름이 불안정해질 수 있다. 이것이 DDPM에서 사용되는 exponential moving average 방식($\theta\_{\text{EMA}} = \lambda \theta\_{\text{EMA}} + (1 - \lambda) \theta\_{\text{current}}$)에세도 학습의 불안정을 유발할 수 있다.

### Noise schedule

DDPM에서는 다음과 같이 $x_0$에 점진적으로 노이즈를 추가한다.

$$
x_t = \sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

이 $\alpha_t$를 어떻게 설정하는지에 따라 denoising의 정도가 달라진다. 노이즈 제거 속도에 따라 원본 복원의 정도 inference 속도 등에 변화가 생긴다. 방식은 linear, cosine, exponential 등이 있으며 여기서는 iDDPM에서 사용한 Square Cosine Schedule을 이용한다.

### Accelerating inference for real-time control

DDIM에서는 training과 inference에서 denoising 과정을 분리하여 inference의 속도를 높인다. 구체적으로는 non-Markovian process, skip sampling을 통해 이루어지며 DDPM에 비해 20배가량 빠른 inference가 가능하다. Real-world에서는 3080 GPU 기준 100 training iterations, 10 inference iteration에 대해 0.1s의 latency가 나타난다.

## Intriguing properties of diffusion policy

### Model Multi-Modal Action Distributions

<center>
<img src='{{"assets/images/Diffusion Policy/diff3.png" | relative_url}}' style="max-width: 100%; width: auto;">
<figcaption style="text-align: center;"></figcaption>
</center>

Bahavior cloning에서 multi-modal은 중요한 주제로 이야기된다. Stochastic sampling 과정은 이러한 multi-model을 잘 표현하도록 한다. 다양한 해법이 있을 때, 한 가지 방법 뿐만 아니라 골고루 활용하는 것을 알 수 있다.

### Synergy with Position Control

<center>
<img src='{{"assets/images/Diffusion Policy/diff4.png" | relative_url}}' style="max-width: 100%; width: auto;">
<figcaption style="text-align: center;"></figcaption>
</center>

Diffusion policy의 특징적인 부분은 position-control이 velocity-control 보다 더 나은 성능을 보인다는 점이다. 지금까지 대부분의 behavior cloning 방식은 velocity-control을 이용하였다. Position-control이 더 효과적인 이유에 대해서는 다음과 같이 추측한다. 먼저 multi-modal 표현이 position의 경우에 더 강하게 유발된다. 만약 특정 지점으로 가는 경로를 생각하면 다양한 경로(position)를 구상할 수 있다. 그리고 compounding error의 영향을 덜 받는다. Velocity-control의 경우 미세한 오차가 적분 형태로 누적되므로 이후에 큰 오차로 나타날 수 있다. 다른 알고리즘과 비교해 보아도 positional control을 이용했을 때 성능 변화가 뚜렷하다.

### Benefits of Action-Sequence Prediction

Diffusion policy의 경우 action-sequnce prediction에 적합하다. 이전의 알고리즘들이 겪는 문제와 비교하여 diffusion policy의 장점을 확인할 수 있다.
- Temporal action consistency
  T-block의 예제를 보자. Diffusion policy는 한 방향이 설정되면 일관되게 그 방향으로 진행하는 것을 볼 수 있다. 다른 알고리즘들은 각 step마다 행동이 독립적으로 예측되기 때문에 명확하지 않은 구간에서는 action jittering이 발생할 수 있다.
- Robustness to idle actions
  기존 알고리즘은 idle action(유휴 행동, 동일한 action이 계속 주어지는 상황) 취약한 편이다. Idle action은 훈련 데이터에 포함되어 있을 수 있다. 로봇은 이러한 idle action을 학습하여 과적합에 빠지기 쉽다. 가령 물을 따르는 task 일때, 물을 붓는 동안에는 정지해야 한다. 이렇게 된다면 정지 동작이 과적합을 유발하여 계속 로봇이 움직이지 않게 할 수 있다. Diffusion policy는 action sequence predict로 인해 이러한 상황에 좀 더 자유롭다는 듯하다.

### Training Stability

<center>
<img src='{{"assets/images/Diffusion Policy/diff6.png" | relative_url}}' style="max-width: 100%; width: auto;">
<figcaption style="text-align: center;"></figcaption>
</center>

$$
\mathcal{L}_{\text{InfoNCE}} = -\log \left( \frac{e^{-E_{\theta}(\mathbf{o}, \mathbf{a})}}{e^{-E_{\theta}(\mathbf{o}, \mathbf{a})} + \sum_{j=1}^{N_{\text{neg}}} e^{-E_{\theta}(\mathbf{o}, \bar{\mathbf{a}}^j)}} \right)
$$

IBC는 학습 과정에서 안정성이 떨어지는 모습을 보인다. InfoNCE style의 loss function 내의 negative samples가 학습의 안정성에 영향을 주기 때문이다. 만약 negative sample이 정확하지 않은 경우 $Z(\mathbf{o}, \theta)$를 올바르게 추정할 수 없고, 이는 학습의 불안정을 유발한다. 로봇 학습에서의 negative sample은 task를 올바르게 수행하지 못한 action의 예시가 되겠다.

$$
\nabla_a \log p(\mathbf{a}|\mathbf{o}) = -\nabla_a E_{\theta}(\mathbf{a}, \mathbf{o}) - \underbrace{\nabla_a \log Z(\mathbf{o}, \theta)}_{=0} \approx -\varepsilon_{\theta}(\mathbf{a}, \mathbf{o})
$$

Diffusion policy의 경우 $Z(\mathbf{o}, \theta)$를 예측할 필요가 없기 때문에 안정적인 학습이 가능한 것으로 보인다.

### Connections to Control Theory

Diffusion policy가 실제로 linear system에서 feedback policy $\mathbf{a} = -\mathbf{K} s$로 수렴한다는 것을 보인다. 다음의 linear system을 생각해 보자.

$$
s_{t+1} = \mathbf{A} s_t + \mathbf{B} a_t + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(0, \mathbf{\Sigma}_w).
$$

그리고 $T_p=1$로 두면 DDPM의 loss function을 사용했을 때,

$$
\mathcal{L} = \text{MSE}(\varepsilon^k, \varepsilon_{\theta} (s_t, -\mathbf{K} s_t + \varepsilon^k, k))
$$

를 최소화하는

$$
\varepsilon_{\theta}(s, a, k) = \frac{1}{\sigma_k} [a + \mathbf{K} s],
$$

<details>
<summary>유도 과정</summary>
<div markdown="1">
### **📌 유도 과정: Diffusion Policy에서 최적의 Denoiser**
Diffusion Policy의 학습 과정에서 **최적의 노이즈 제거 모델 \\( \epsilon_\theta(s, a, k) \\) 가 다음과 같이 주어진다:**  
\\[
\epsilon_\theta(s, a, k) = \frac{1}{\sigma_k} [a + K s]
\\]
이제 **이 식이 어떻게 유도되는지** 하나씩 차근차근 증명해보자. 🚀  

---

## **1️⃣ 문제 설정: 선형 동적 시스템 (Linear Dynamical System)**
우리는 **선형 시스템에서의 행동 예측 문제**를 고려한다:
\\[
s_{t+1} = A s_t + B a_t + w_t, \quad w_t \sim \mathcal{N}(0, \Sigma_w)
\\]
- \\( s_t \\) : 현재 상태  
- \\( a_t \\) : 현재 행동  
- \\( w_t \\) : 가우시안 노이즈  

이제, 우리가 학습하고자 하는 정책은 **선형 피드백 정책(Linear Feedback Policy) \\( a = -K s \\) 를 모방하는 것**이다.

---

## **2️⃣ Diffusion Process 적용**
Diffusion Policy는 **노이즈를 점진적으로 제거하면서 최적의 행동을 예측하는 방식**을 사용한다.  
따라서, **노이즈화된 행동(action with noise)** 을 다음과 같이 정의할 수 있다:

\\[
a_t = -K s_t + \epsilon_k
\\]

여기서:
- **\\( \epsilon_k \\)** : Diffusion Process에서 추가된 노이즈  
- **\\( k \\) 번째 denoising step에서 제거해야 하는 노이즈**  

즉, **Diffusion Policy의 목표는 \\( \epsilon_k \\) 를 제거하여 \\( a = -K s \\) 로 수렴하도록 학습하는 것!**  

---

## **3️⃣ Mean Squared Error (MSE) 손실 함수**
Diffusion Policy의 학습 과정에서, **최적의 denoiser \\( \epsilon_\theta(s, a, k) \\) 를 찾기 위해 MSE 손실 함수를 최소화한다.**  
즉, 다음을 최소화하는 \\( \epsilon_\theta(s, a, k) \\) 를 찾아야 한다:

\\[
L = MSE(\epsilon_k, \epsilon_\theta(s, a, k))
\\]

손실 함수 정의:
\\[
L = \mathbb{E} \left[ \|\epsilon_k - \epsilon_\theta(s, a, k) \|^2 \right]
\\]

최적의 denoiser는 **조건부 기대값(Conditional Expectation)** 에 의해 결정된다:
\\[
\epsilon_\theta(s, a, k) = \mathbb{E} [\epsilon_k | s, a]
\\]

---

## **4️⃣ 조건부 기대값 \\( \mathbb{E} [\epsilon_k | s, a] \\) 계산**
우리는 다음과 같은 행동 모델을 가지고 있다:
\\[
a = -K s + \epsilon_k
\\]

이를 정리하면:
\\[
\epsilon_k = a + K s
\\]

따라서, 조건부 기대값을 구하면:
\\[
\mathbb{E} [\epsilon_k | s, a] = \mathbb{E} [a + K s | s, a]
\\]

여기서 **\\( a \\) 와 \\( s \\) 는 주어진 상태이므로 상수처럼 처리 가능**하므로:
\\[
\mathbb{E} [\epsilon_k | s, a] = a + K s
\\]

🚀 **즉, 최적의 denoiser는 다음과 같이 주어진다!**  
\\[
\epsilon_\theta(s, a, k) = a + K s
\\]

---

## **5️⃣ 노이즈 스케일링 반영 (\\( \sigma_k \\) 적용)**
Diffusion Process에서 **노이즈는 단계별 스케일링을 가지며, \\( \sigma_k \\) 에 의해 조절된다.**  
따라서, denoiser는 노이즈의 스케일을 고려하여 다음과 같이 정규화된다:

\\[
\epsilon_\theta(s, a, k) = \frac{1}{\sigma_k} (a + K s)
\\]

✅ **최종 결과:**  
\\[
\epsilon_\theta(s, a, k) = \frac{1}{\sigma_k} [a + K s]
\\]

---

## **📌 최종 요약**
1. **행동 모델**: \\( a = -K s + \epsilon_k \\)  
2. **MSE 손실을 최소화하기 위해 \\( \epsilon_k \\) 의 조건부 기대값을 계산**  
3. **결과적으로 \\( \epsilon_\theta(s, a, k) = a + K s \\) 를 얻음**  
4. **노이즈 스케일링을 반영하여 \\( \frac{1}{\sigma_k} \\) 를 곱해 최종 식 도출!**  

🚀 **결론:** Diffusion Policy는 **노이즈를 제거하면서 선형 피드백 제어 정책을 올바르게 학습할 수 있음을 수식적으로 확인할 수 있다!** 😊
</div>
</details>

따라서 denoising을 통해 global minima $\mathbf{a} = -\mathbf{K} s$로 수렴할 수 있다. $T_p>1$인 경우도 같은 결과를 준다.

## Evaluation


## Realworld evaluation


## Realworld bimanual tasks


## Related work


## Limitations and future work

몇 가지 한계점이 있다. 결국 behavior cloning이기 때문에 데이터에 따라 suboptimal한 해를 보일 수 있다. 그리고 이전 알고리즘들에 비해 계산량이 무겁고 추론 속도로 인해 latency가 존재한다.

## Conclusion
