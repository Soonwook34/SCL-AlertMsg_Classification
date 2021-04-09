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

**Pytorch 설치 확인 코드**

```python
import torch
x = torch.rand(5, 3)
print(x)
print(torch.cuda.is_available())
```