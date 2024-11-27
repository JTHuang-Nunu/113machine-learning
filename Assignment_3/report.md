## Acitvate Function

`Acitvate Function.md` 文件描述了激活函数的作用，以及常見的激活函数有哪些，以及它們的公式。

## main.ipynb

```python
X, y = make_classification(n_samples=1000, n_features=num_feature, n_classes=2, random_state=42)
```

隨機產生 1000 筆資料，每筆資料有 20 個特徵，共有 2 個類別。

```python
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

```

用 Linear 的方式，將輸入資料的 20 個特徵，通過一個線性層，再用 Sigmoid Acitivate Function 將輸出值轉換為 0~1 之間的機率值。

```bash
Epoch [100/1000], Loss: 0.5407
Epoch [200/1000], Loss: 0.4644
Epoch [300/1000], Loss: 0.4243
Epoch [400/1000], Loss: 0.3996
Epoch [500/1000], Loss: 0.3829
Epoch [600/1000], Loss: 0.3709
Epoch [700/1000], Loss: 0.3620
Epoch [800/1000], Loss: 0.3550
Epoch [900/1000], Loss: 0.3495
Epoch [1000/1000], Loss: 0.3451
Accuracy: 0.8400
```

最後得到的結果 > **Accuracy: 0.8400**
