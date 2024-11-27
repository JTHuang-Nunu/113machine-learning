## Intro

CIFAR-10 資料集包含 60,000 張 32×32 像素的彩色圖像，分為 10 個不同的類別。

## Implement

### data_generator

生成新的資料集
依照 batch_size 及 batch loop 的多寡來生成新資料

```python
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

data_generator = datagen.flow(x_train, y_train, batch_size=10000)

augmented_images, augmented_labels = [],[]
# ImageDataGenerator.flow() -> generate the rotate image
for i in range(5):  # run 5 batch
    x_batch, y_batch = next(data_generator)
    augmented_images.extend(x_batch)
    augmented_labels.extend(y_batch)
```

### VGG16

output Feature vector

```python
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
x = Flatten()(base_model.output)
feature_model = Model(inputs=base_model.input, outputs=x)
feature_model.summary()
```

### StandardScaler

標準化資料

### SVM

rbf(Polynomial kernel function): 目前最主流的 kernel，用泰勒級數收斂
C: 容忍度，可以通過 GridSearchCV 的方式來找出最適合的 C 值

Question- 為何不用設定有多少種類
Answer- SVM 用於二元分類問題，在多個類別標籤的情況下，它會自動使用多分類策略處理

```python
svm = SVC(kernel='rbf', C=10, gamma='scale')  # 使用 RBF 核函數
svm.fit(x_train_features, y_train_augment)

# Estimate model
y_pred = svm.predict(x_test_features)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

```python
# 定義參數範圍
param_grid = {'C': [0.1, 1, 10, 100, 1000]}

# 定義模型
model = SVC(kernel='linear')

# 網格搜索尋找最佳 C 值
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(x_train_features, y_train)
```

## Summary

SVM 不能用 GPU Training，Training 時間相對長

Data Augment 擴大資料集未必助於提高 Accuracy
原先 trainset=50,000→100,000→75,000
Accuracy: 0.62→ 0.6192→0.606
