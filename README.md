# SCL-AlertMsg_Classification
한국 코로나 재난문자 분류하기

**Conda 환경 구성**

```jsx
conda create -n SCL python=3.8
conda activate SCL
```

**PyTorch 설치**

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

```jsx
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
```
Stable(1.8.1) / Windows / Conda / Python / CUDA 11.1

**Pytorch 설치 확인 코드**

```python
import torch
x = torch.rand(5, 3)
print(x)
print(torch.cuda.is_available())
```
tensor([[0.2574, 0.2301, 0.6204],
        [0.5290, 0.7206, 0.0804],
        [0.6010, 0.8442, 0.3168],
        [0.7276, 0.2248, 0.9528],
        [0.3123, 0.8787, 0.4104]])
True