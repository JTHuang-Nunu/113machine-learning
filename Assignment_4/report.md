# main.ipynb

在 Iris 資料集中使用 `Classification` 以及 `Regression` 進行分類。

## Classification

### Model

```python
class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(4, 16)  # 輸入層到隱藏層
        self.fc2 = nn.Linear(16, 8)  # 隱藏層到隱藏層
        self.fc3 = nn.Linear(8, 3)   # 隱藏層到輸出層
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

model_cls = ClassificationModel()
criterion_cls = nn.CrossEntropyLoss()
optimizer_cls = optim.Adam(model_cls.parameters(), lr=0.01)
```

> NN： 4 -> 16 -> 8 -> 3 取 softmax

Epoch [100/100], Loss: 0.5742

### Validation

以各項指標驗證模型，`Accuracy`、`Precision`、`Recall`、`F1 Score` 以及 `Confusion Matrix`

```python
accuracy = accuracy_score(actual.cpu(), predicted.cpu())
precision = precision_score(actual.cpu(), predicted.cpu(), average='weighted')
recall = recall_score(actual.cpu(), predicted.cpu(), average='weighted')
f1 = f1_score(actual.cpu(), predicted.cpu(), average='weighted')
```

```bash
Accuracy: 1.0000

Precision: 1.0000

Recall: 1.0000

F1 Score: 1.0000

Confusion Matrix:
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
```

## Regression

### Model

```python
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(4, 16)  # 輸入層到隱藏層
        self.fc2 = nn.Linear(16, 8)  # 隱藏層到隱藏層
        self.fc3 = nn.Linear(8, 1)   # 隱藏層到輸出層

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model_cls = ClassificationModel()
criterion_cls = nn.CrossEntropyLoss()
optimizer_cls = optim.Adam(model_cls.parameters(), lr=0.01)
```

> NN： 4 -> 16 -> 8 -> 1 取 RMSE

Epoch [200/100], Loss: 0.1005

### Validation

以 RMSE 及 R² 指標驗證模型，

```python
rmse_reg = mean_squared_error(y_true_reg.numpy(), y_pred_reg.numpy())
r2_reg = r2_score(y_true_reg.numpy(), y_pred_reg.numpy())
```

使用 rsme 以及 r2_score 驗證模型

```bash
Regression Model RMSE: 0.0945

Regression Model R²: 0.8647
```
