# VAE

> File: [VAE.ipynb]

## Model

```python
# 針對 CIFAR10 的 VAE 模型
class VAE(nn.Module):
    def __init__(self, latent_dim):
            super(VAE, self).__init__()
            self.latent_dim = latent_dim

            # Encoder network (CNN)
            self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)  # 32x32x3 -> 16x16x32
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 16x16x32 -> 8x8x64
            self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 8x8x64 -> 4x4x128
            self.fc1 = nn.Linear(128*4*4, 512)
            self.fc2_mu = nn.Linear(512, latent_dim)  # For mean
            self.fc2_logvar = nn.Linear(512, latent_dim)  # For log-variance

            # Decoder network (CNN)
            self.fc3 = nn.Linear(latent_dim, 512)
            self.fc4 = nn.Linear(512, 128*4*4)
            self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 4x4x128 -> 8x8x64
            self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 8x8x64 -> 16x16x32
            self.conv6 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)  # 16x16x32 -> 32x32x3
```

## Result

### CIFAR10

**First Run**

> Epoch [1/10], Loss: 663.7647904101563
>
> Epoch [2/10], Loss: 567.719385703125
>
> Epoch [3/10], Loss: 553.1078017382813
>
> Epoch [4/10], Loss: 548.7949794335938
>
> Epoch [5/10], Loss: 546.4576111523437
>
> Epoch [6/10], Loss: 544.1890346191407
>
> Epoch [7/10], Loss: 542.3913061132813
>
> Epoch [8/10], Loss: 541.2759715625
>
> Epoch [9/10], Loss: 540.4163854882812
>
> Epoch [10/10], Loss: 539.7007494921875

**Second Run** 接續 First Run 模型

> Epoch [1/10], Loss: 538.9394563574218
>
> Epoch [2/10], Loss: 538.2353040039062
>
> Epoch [3/10], Loss: 537.6349412890625
>
> Epoch [4/10], Loss: 537.1351985449219
>
> Epoch [5/10], Loss: 536.681393515625
>
> Epoch [6/10], Loss: 536.2931744140625
>
> Epoch [7/10], Loss: 535.8672916796875
>
> Epoch [8/10], Loss: 535.5696532226563
>
> Epoch [9/10], Loss: 535.2271116992188
>
> Epoch [10/10], Loss: 534.9387535839844

生成 20 張圖片，成果不太好，因為 CIFAR 本身的圖片較為複雜，而且像素較低，所以生成的圖片也不太好看清楚。
![alt text](image.png)
