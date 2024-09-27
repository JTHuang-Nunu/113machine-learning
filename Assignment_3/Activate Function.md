![網路抓的Activate Function](https://ik.imagekit.io/pjspqa5eh/obsidian/240810-7_Ubs9QfdmkD.png)

## 摘要

- **ReLU** 用於隱藏層，主要用於引入非線性並加速訓練。
- **Softmax** 用於輸出層，主要用於多分類問題，輸出每個類別的機率。

<br/>

## Activate Function

### 1. Sigmoid 函數

Sigmoid 將輸入映射到 \(0\) 和 \(1\) 之间，是一種 S 曲獻。
$\sigma(x) = \frac{1}{1 + e^{-x}}$
通常用作二元分類

### 2. Tanh 函數

Tanh 將輸入映射到 \(-1\) 和 \(1\) 之间，也是 S 曲線，但更對稱一些。
$\text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

### 3. ReLU (Recitified Linear Unit)函數

ReLU 是目前最常用的激活函数之一，負數會映射為 0，正數則保持不變。
它可以幫助加速訓練以及減少梯度消失問題。
$\text{ReLU}(x) = \max(0, x)$

### 4. Softmax 函數

Softmax 通常用於 NN 網路的輸出層，將潛空間的向量轉成機率分布，解決分類問題
$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$
