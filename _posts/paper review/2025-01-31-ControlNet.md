---
title: "[ControlNet] Adding Conditional Control to Text-to-Image Diffusion Models"
last_modified_at: 2025-01-31
categories:
  - paper_review
tags:
  - ControlNet
  - Conditional Control
  - ICCV
  - Text-to-Image
excerpt: "ControlNet paper review"
use_math: true
classes: wide
---

> ICCV 2023. [[Paper](https://arxiv.org/abs/2302.05543)] [[Github](https://github.com/lllyasviel/ControlNet)]  
> Lvmin Zhang, Anyi Rao, and Maneesh Agrawala  
> 10 Feb 2023

## Summary

ControlNet은 large, pretrained text-to-image 모델에 spatial conditioning을 더해주는 모델이다. Zero-convolution이라는 구조를 통해 parameter를 0부터 키워가며 노이즈가 finetuning을 방해하는 것을 막는다. 다양한 conditioning(edge, segmentation, human pose 등)에 적용된다. 50K 미만, 1M 이상의 다양한 데이터셋 풀에서도 robust하다.

## Introduction

특정 conditioning을 주는 것은 데이터의 부족 등으로 어려운 과제이다. 가령, stable diffusion의 학습데이터 50억 개(LAION-5B)로 학습시켰다. 부족한 수량의 데이터로 fine-tuning을 하게 된다면 과적합이나 catastrophic forgetting이 발생할 수 있다. 이 논문에서는 ControlNet을 제시한다. ControlNet은 Stable Diffusion과 같은 pretrained text-to-image 모델에 conditional control을 주는 network architecture이다. NVIDIA RTX 3090Ti GPU로도 충분히 학습가능하다.

## Related work
### Finetuning neural network
- HyperNetwork
  
  NLP 분야에서 제시된 HyperNetwork는 작은 RNN을 학습하여 더 큰 모델의 weight에 영향을 주도록 한다. 이를 Stable Diffusion에 적용하여 그림 스타일을 바꾸는 연구가 있다.
  
- Adapter

  Adapter 또한 NLP에서 pretrained transformer를 다른 task에 customize하기 위해 새로운 module을 embedding한다. CV에서는 Incremental learning이나 domain adaptation에 사용된다. Vision transformer와 결합한 연구도 있다. T2I-Adapter는 Stable Diffusion과 결합하여 외부 condition을 준다.

- LoRA
  
  LoRA는 low intrinsic dimension을 이용하여 catasrophic forgetting을 방지한다.

- Zero-initialized layers

  ControlNet에서 사용되는 구조이다. Weight initialization에 대해서는 gaussian initialization이 zero-initialization보다 덜 risky한 것으로 알려져 있다.

### Image diffusion

주로 text-to-image generation 모델 연구 및 conditioning을 주는 연구로 구분된다.

### Image-to-image translation


## Method

### ControlNet

<center>
<img src='{{"assets/images/ControlNet/controlnet1.png" | relative_url}}' width="40%">
<figcaption style="text-align: center;"></figcaption>
</center>

ControlNet은 기본적으로 fixed pretrained Stable Diffusion과 trainable Stable Diffusion encoder with zero convolution decoder로 나뉜다. Block 단위에서의 연산은 다음과 같이 trainable copy 앞뒤로 zero convolution을 배치하게 되는데, zero convolution은 초기값이 0인 1x1 convolution이다. 따라서 학습 초기 단계에서는 ControlNet 구조가 Stable Diffusion의 출력에 영향을 미치지 못한다. 결과적으로 초기 단계에서 weight initialization에 의한 random noise가 불필요한 gradient를 생성하는 것을 방지하게 된다. 수식으로 표현하면 다음과 같다.

$$
y_c = \mathcal{F}(x; \Theta) + \mathcal{Z}\big(\mathcal{F}(x + \mathcal{Z}(c; \Theta_1); \Theta_c); \Theta_2\big)
$$

### ControlNet for text-to-image diffusion

<center>
<img src='{{"assets/images/ControlNet/controlnet2.png" | relative_url}}' width="40%">
<figcaption style="text-align: center;"></figcaption>
</center>

Stable Diffusion은 encoder, decoder 각각 12개의 block을 가지고 있고, middle block 1개를 포함하면 총 25개의 block으로 구성된다. 구체적으로 encoder, decoder에서의 8개 block은 up/down-sampling을 위한 block이다. Middle block과 encoder, decoder의 나머지 block들은 main block으로 각각 4개의 resnet layer와 2개의 ViT로 구성되어 있다. 각 ViT는 여러 개의 cross-attention과 self-attention 구조로 되어 있다. Text prompt는 CLIP text encoder로 처리되며, positional encoding으로 timestep이 함께 입력으로 들어간다.

ControlNet은 계산 측면에서도 효율적인데, 원래 Stable Diffusion과 비교하면 NVIDIA A100 PCIE 40GB 환경에서 23%의 GPU memory 감소 및 학습 iteration 당 34%의 시간 감소가 있었다.

그리고 Stable Diffusion가 사용했던 pre-processing 형식을 맞추기 위해 512x512 image-space condition $c_i$를 간단한 convolution network에 통과시켜 64x64 feature-space condition vector $c_f$로 맞추어준다. 참고로 이 network도 full model과 같이 학습된다.

### Training

Network는 $z_t$에 더해진 noise를 학습한다. 학습 과정에서 $c_t$의 50%를 빈 string으로 학습하는데 이는 edge, pose, depth와 같은 conditioning의 semantics를 더 잘 이해하도록 한다. 

<details>
<summary>[GPT]왜 $c_t$ 일부를 막는것이 semantic 학습에 도움이 되는지?</summary>
<div markdown="1">

### **ControlNet에서 Text Prompt $c_t$의 절반을 가리고 학습하는 이유**
---
### ✅ **1. 문제 정의**
ControlNet은 기존의 Stable Diffusion과 다르게, **Edge, Pose, Depth 등의 추가 조건(Conditioning Input)을 이용하여 이미지 생성을 컨트롤**하는 모델이다.  

하지만, 만약 ControlNet이 **조건(Conditioning Input)에 너무 의존하면**, 모델이 **텍스트 프롬프트($c_t$)를 제대로 해석하지 못하고, 단순히 주어진 조건(예: 엣지 맵)만을 따라가는 현상**이 발생할 수 있다.  

👉 이를 해결하기 위해 **ControlNet 학습 시 텍스트 프롬프트의 일부를 마스킹(masking)하여, 모델이 조건(condition)만으로도 의미(semantics)를 추론하도록 유도**하는 방법을 사용한다.

---

### ✅ **2. 학습 과정에서 절반의 텍스트 프롬프트 $c_t$를 가리는 이유**
ControlNet 논문에서는 학습 과정에서 **텍스트 프롬프트 $c_t$의 절반을 랜덤하게 제거**하여 학습하는 기법을 사용한다.  
이렇게 하면, 모델은 **텍스트 프롬프트 없이도, 주어진 조건(condition)에서 의미(semantics)를 추론하는 능력**을 갖게 된다.

이러한 방식이 어떻게 작용하는지 수학적 원리와 예제를 통해 설명하겠다.

---

### ✅ **3. 수학적 원리**
#### **(1) 일반적인 Stable Diffusion의 Latent Space에서의 Sampling**
Stable Diffusion은 아래와 같은 **노이즈 제거 모델 $\epsilon_\theta$**을 학습한다.

$$
x_{t-1} = x_t - \epsilon_\theta(x_t, c_t)
$$

- $x_t$ : 현재 시점의 Latent Variable (노이즈가 추가된 이미지)
- $c_t$ : 텍스트 프롬프트
- $\epsilon_\theta(x_t, c_t)$ : 모델이 예측한 노이즈

즉, 모델은 텍스트 프롬프트 $c_t$를 기반으로 $x_t$에서 **올바른 노이즈를 제거**하여, 원하는 이미지로 변환해야 한다.

---

#### **(2) ControlNet에서의 Conditional Control**
ControlNet에서는 단순한 텍스트 프롬프트 $c_t$ 뿐만 아니라, **추가적인 조건(condition) $c_{cond}$ (예: 엣지 맵, 포즈, 심도 정보 등)**을 함께 활용한다.

즉, 노이즈 제거 과정이 아래와 같이 확장된다.

$$
x_{t-1} = x_t - \epsilon_\theta(x_t, c_t, c_{cond})
$$

여기서 $c_{cond}$가 **Edge, Pose, Depth 등의 조건 입력**에 해당한다.

---

#### **(3) 텍스트 프롬프트의 절반을 제거하는 효과**
모델이 학습할 때, 텍스트 프롬프트 $c_t$의 절반을 제거하면:

1. **텍스트 없이도 조건 $c_{cond}$ 만으로 의미를 이해하도록 유도**  
   $$
   x_{t-1} = x_t - \epsilon_\theta(x_t, \emptyset, c_{cond})
   $$
   → 이 경우, 모델은 $c_{cond}$ (Edge, Pose, Depth) 만으로도 올바른 출력을 생성해야 한다.

2. **조건 $c_{cond}$ 의 의미(semantics)를 더욱 깊게 학습**  
   - 만약 텍스트가 항상 존재했다면, 모델이 텍스트에 의존하여 생성할 수 있다.
   - 그러나 텍스트가 없는 경우, 모델은 **Edge, Pose, Depth만을 보고 해당 이미지가 어떤 의미인지 유추해야 함**.
   - 즉, 모델이 단순히 **엣지 검출을 따라가는 것이 아니라, 엣지의 의미(예: 얼굴 윤곽 vs 배경 윤곽)를 더 잘 구별할 수 있도록 학습됨**.

3. **텍스트 + 조건을 함께 활용하는 능력 증가**  
   $$
   x_{t-1} = x_t - \epsilon_\theta(x_t, c_t, c_{cond})
   $$
   - 이제 모델이 텍스트와 조건을 함께 사용할 때도, 단순히 텍스트에만 의존하지 않고, **조건을 더 깊게 활용하면서 최적의 결과를 생성**할 수 있음.

---

### ✅ **4. 예제: 왜 이런 학습 방식이 효과적인가?**
#### **(1) 엣지 검출 (Canny Edge) 예제**
- 만약 ControlNet이 텍스트 프롬프트를 항상 참고하면:
  - **"a cat sitting on a table"**이라는 프롬프트를 보고, 고양이의 윤곽이 어딘지 신경 쓰지 않고 이미지를 생성.
  - 엣지 정보가 변형되었을 때 올바르게 대응하지 못할 가능성이 높음.

- 하지만 **학습 과정에서 텍스트 프롬프트의 일부가 제거되면**:
  - 모델은 단순히 "고양이"라는 단어를 보고 생성하는 것이 아니라, **엣지만 보고 이것이 고양이인지, 배경인지, 다른 사물인지 스스로 이해하는 능력을 학습**.
  - 즉, 모델이 **엣지의 의미를 더 깊게 학습**하게 됨.

---

#### **(2) 포즈 추정 (Pose Estimation) 예제**
- 만약 ControlNet이 텍스트 프롬프트에 의존하면:
  - **"a dancing person"**이라는 프롬프트를 보고 포즈를 무시하고 춤을 추는 사람을 생성하려 할 수도 있음.
  - 하지만 **학습 과정에서 텍스트가 가려지면**, 모델은 포즈만 보고 "이 포즈가 춤을 추는 사람인지, 아니면 서 있는 사람인지" 이해해야 함.

👉 **결과적으로, 포즈의 의미(semantics)를 더 정확하게 파악하게 됨**.

---

### ✅ **5. 결론**
👉 **ControlNet이 텍스트 프롬프트 $c_t$의 절반을 가리고 학습하면, 모델이 Edge, Pose, Depth 등의 조건을 더욱 깊이 이해할 수 있다.**  

👉 **이러한 방식 덕분에, 모델이 단순히 텍스트에 의존하지 않고, 각 조건의 의미를 스스로 파악할 수 있도록 유도할 수 있다.**  

👉 **결과적으로, ControlNet이 더 일반화된 조건 기반 이미지 생성을 수행할 수 있도록 학습됨.** 🚀
</div>
</details>

$$
\mathcal{L} = \mathbb{E}_{x_0, t, c_t, c_f, \epsilon \sim \mathcal{N}(0, 1)} \left[ \left\| \epsilon - \epsilon_\theta(z_t, t, c_t, c_f) \right\|_2^2 \right]
$$

<center>
<img src='{{"assets/images/ControlNet/controlnet3.png" | relative_url}}' width="40%">
<figcaption style="text-align: center;"></figcaption>
</center>

추가적으로 확인되는 현상은 zero convolution의 특성(초기 학습에는 큰 gradient를 만들어 내지 못함) 때문에 어느 순간 비약적인 변화를 보여준다고 하는데(10K steps 부근)이를 sudden convergence phenomenon으로 명명한다.

### Inference

<center>
<img src='{{"assets/images/ControlNet/controlnet4.png" | relative_url}}' width="40%">
<figcaption style="text-align: center;"></figcaption>
</center>

$$
\epsilon_{\text{prd}} = \epsilon_{\text{uc}} + \beta_{\text{cfg}} (\epsilon_{\text{c}} - \epsilon_{\text{uc}})
$$

Stable Diffusion에서는 높은 quality의 이미지를 뽑아내기 위해 CFG(classifier free guidance)를 사용한다. Unconditional output과 conditional output을 weighted sum하는 것인데 높은 $\beta_{\text{cfg}}$에 대해서는 텍스트의 세부 사항을 정확히 반영한다.
ControlNet에서는 conditioning image를 $\epsilon_{\text{uc}}, \epsilon_{\text{c}}$ 모두 혹은 $\epsilon_{\text{c}}$에만 더해줄 수 있다. 극단적으로 prompt가 주어지지 않았다고 가정했을 때, 둘 모두에 더했을 때 CFG guidance의 효과사 상쇄된다(그림 b, 이 부분은 정확히 이해가 되지 않는데, $\epsilon_{\text{uc}}=\epsilon_{\text{c}}$가 되어 $\beta_{\text{cfg}}$가 더 이상 무의미하기 때문이라 판단함). $\epsilon_{\text{c}}$에만 더하게 되면 conditional image의 guidance가 해상도 별로 나뉘어지지 않고 conditioning image의 영향이 지배적이어서 텍스쳐가 무너진 것을 확인할 수 있다(그림 c).
ControlNet에서는 conditioning image를 Stable Diffusion에 더해줄 때, block의 size에 따라 다른 weight를 곱한 뒤 더해준다. 구체적으로는 $w_i=64/h_i$ 의 형태로, block의 size가 작을 수록(middle block에 가까울수록) 큰 weight를 곱해주게 된다. 해상도가 낮아짐에 따라 정보의 손실로 control signal이 강하지 않으면 conditional image가 제대로 반영되지 않을 것이다. 해상도가 높아진 경우, 텍스쳐 및 세부 정보를 위해 conditional image의 신호를 적게 주입하게 된다.

<details>
<summary >[GPT]좀 더 자세히</summary>
<div markdown="1">
# 📌 **Classifier-Free Guidance (CFG) 개요**
Stable Diffusion 및 ControlNet에서 **Classifier-Free Guidance (CFG)**는 주어진 조건(예: 텍스트 프롬프트, 컨트롤 맵)의 영향을 조절하는 핵심 기법이다.

## **1. Stable Diffusion (LDM)에서의 CFG**
Stable Diffusion에서는 **CFG를 활용하여 조건을 강화하고 이미지 품질을 향상**시킨다.

### **1.1 CFG 수식**
Stable Diffusion에서 노이즈 예측 모델 $ \epsilon_{\theta} $ 는 두 가지 방식으로 샘플링된다.

1. **무조건부 샘플링 (Unconditional, $ \epsilon_{\theta}(\mathbf{x}_t) $ )**  
   - 조건 없이 샘플링된 결과  
2. **조건부 샘플링 (Conditional, $ \epsilon_{\theta}(\mathbf{x}_t, c) $ )**  
   - 텍스트 프롬프트 등의 조건을 포함한 결과  

이 두 결과를 CFG 스케일 $ w $ 에 따라 다음과 같이 결합한다:

$$
\hat{\epsilon}_t = \epsilon_{\theta}(\mathbf{x}_t) + w \cdot (\epsilon_{\theta}(\mathbf{x}_t, c) - \epsilon_{\theta}(\mathbf{x}_t))
$$

여기서:
- $ \hat{\epsilon}_t $ : 최종적으로 적용된 노이즈 예측 값
- $ \mathbf{x}_t $ : 확산 과정 중의 샘플
- $ w $ : CFG 스케일 (보통 1~10 범위, 값이 클수록 조건을 강하게 반영)

### **1.2 예제**
#### **(1) CFG = 1 (일반적인 설정)**
일반적인 샘플링 방식이며, 조건을 약간 반영하지만 완전히 따르지는 않는다.

#### **(2) CFG = 0 (무조건부 샘플링)**
조건을 무시하고 랜덤한 샘플을 생성한다.

#### **(3) CFG = 10 (강한 조건부 샘플링)**
텍스트 프롬프트나 컨트롤 신호를 강하게 반영하지만, 과도한 CFG 값은 아티팩트(이상 현상)를 초래할 수 있다.

---

## **2. ControlNet에서 CFG Resolution Weighting**
ControlNet은 특정 조건(예: 포즈, 엣지 맵, 깊이 맵 등)을 활용하여 Stable Diffusion의 출력을 제어한다. 그러나 **저해상도의 컨트롤 신호는 충분한 정보를 제공하지 못하는 문제**가 있다.

### **2.1 해상도에 따른 문제**
- **저해상도 입력 (Low-resolution Condition)**  
  - 컨트롤 신호가 단순해지고, 이미지의 구조를 정확히 반영하기 어렵다.
  - 예: $ 64 \times 64 $ 포즈 맵 → 세부 구조 손실 발생
- **고해상도 입력 (High-resolution Condition)**  
  - 세부 정보는 증가하지만, 제어 신호가 너무 강하면 모델의 표현력이 제한될 수 있다.

따라서 **해상도에 따라 CFG 강도를 조절하는 기법**이 필요하다.

### **2.2 CFG Resolution Weighting 수식**
ControlNet에서는 해상도에 따라 CFG 가중치를 동적으로 조정한다.

$$
w_{\text{eff}} = f(\text{resolution}) \cdot w
$$

여기서:
- $ w_{\text{eff}} $ : 해상도에 따라 조정된 CFG 스케일
- $ f(\text{resolution}) $ : 해상도에 따른 가중치 함수

일반적으로 $ f(\text{resolution}) $ 은 다음과 같이 설정될 수 있다:

$$
f(\text{resolution}) = 1 + \alpha \cdot \log_2 \left( \frac{\text{resolution}}{256} \right)
$$

여기서:
- $ \alpha $ : 해상도 증가에 따른 조정 계수 (예: $ 0.5 $)
- $ \text{resolution} $ : ControlNet 입력의 해상도 (예: $ 512 $ 또는 $ 1024 $)

### **2.3 예제**
#### **(1) 저해상도 입력 (256×256)**
- $ f(256) = 1 + 0.5 \cdot \log_2(1) = 1.0 $
- $ w_{\text{eff}} = 1.0 \cdot w $ (변화 없음)

#### **(2) 고해상도 입력 (1024×1024)**
- $ f(1024) = 1 + 0.5 \cdot \log_2(4) = 2.0 $
- $ w_{\text{eff}} = 2.0 \cdot w $ (강한 제어 적용)

이를 통해 **고해상도에서는 CFG 강도를 더욱 증가시켜 세부 구조를 유지하도록 한다**.

---

## ✅ **결론: ControlNet에서 CFG Resolution Weighting의 역할**
- ControlNet의 성능을 향상시키기 위한 장치로 볼 수 있다.
- 기존 Stable Diffusion 모델에서도 CFG를 사용하지만, ControlNet에서는 이를 **해상도에 따라 조정하여 더 정밀한 제어가 가능하도록 확장**하였다.
- 이 기법 덕분에 **낮은 해상도에서도 컨트롤 신호를 효과적으로 반영하면서, 고해상도에서는 디테일을 유지**할 수 있다.
</div>
</details>

<center>
<img src='{{"assets/images/ControlNet/controlnet5.png" | relative_url}}' width="40%">
<figcaption style="text-align: center;"></figcaption>
</center>

만약 두 가지 이상의 conditioning이 들어가는 경우, ControlNet의 output을 더하여 Stable Diffusion에 전달하게 된다.

## Experiments

<center>
<img src='{{"assets/images/ControlNet/controlnet6.png" | relative_url}}' width="80%">
<figcaption style="text-align: center;">Prompt 없이 다양한 condition을 입력으로 주었을 때의 output</figcaption>
</center>


<center>
<img src='{{"assets/images/ControlNet/controlnet7.png" | relative_url}}' width="80%">
<figcaption style="text-align: center;">Zero-convolution에 대한 ablation study</figcaption>
</center>


<center>
<img src='{{"assets/images/ControlNet/controlnet8.png" | relative_url}}' width="40%">
<figcaption style="text-align: center;">위에서부터 순서대로 데이터셋 크기에 따른 결과, 애매한 conditioning에 따른 다양한 해석, ControlNet의 transfer inference</figcaption>
</center>