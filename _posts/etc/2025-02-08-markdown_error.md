---
title: "Markdown 수식이 잘 적용되지 않는 경우"
last_modified_at: 2025-02-08
categories:
  - etc
tags:
excerpt: "Markdown 수식 오류"
use_math: true
classes: wide
---

Markdown에서 수식을 사용할 때 종종 수식이 잘 적용되지 않는 경우가 있다.

$\mathbb{E}_\pi$ and $\mathbb{E}_\mu$

이런 경우 보통 $\*$, $\\_$, $\|$ 등에서 오류가 난 경우가 많다.

간단히 앞에 백슬래쉬(\\)를 붙여주면 해결된다.

- Before

> Input
> ```latex
$\mathbb{E}_\pi$ and $\mathbb{E}_\mu$
```
> Output\\
> $\mathbb{E}_\pi$ and $\mathbb{E}_\mu$

- After

> Input
> ```latex
$\mathbb{E}\_\pi$ and $\mathbb{E}\_\mu$
```
> Output\\
> $\mathbb{E}\_\pi$ and $\mathbb{E}\_\mu$