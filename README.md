**Conda 환경 구성**

```bash
conda create -n SCL python=3.8
conda activate SCL
```

**PyTorch 설치** ([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/))

설치 버전: Stable(1.8.1) / Windows / Conda / Python / CUDA 11.1

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
```

**Pytorch 설치 확인 코드** (test/test_pytorch.py)

```python
import torch
x = torch.rand(5, 3)
print(x)
print(f"CUDA 사용가능 여부: {torch.cuda.is_available()}")
print(f"CUDA 현재 GPU 번호: {torch.cuda.current_device()}")
print(f"CUDA에서 사용가능한 GPU 수: {torch.cuda.device_count()}")
print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
```

```bash
tensor([[0.7301, 0.4986, 0.6058],
        [0.9202, 0.8034, 0.4961],
        [0.1361, 0.7180, 0.5290],
        [0.0906, 0.4042, 0.6823],
        [0.2407, 0.6745, 0.3194]])
CUDA 사용가능 여부: True
CUDA 현재 GPU 번호: 0
CUDA에서 사용가능한 GPU 수: 1
GPU 이름: NVIDIA GeForce GTX 1660 Ti
```

**KoBERT에 필요한 라이브러리 설치**

```bash
pip install mxnet gluonnlp pandas tqdm sentencepiece transformers==3
```

**KoBERT 설치**

```bash
pip install git+https://git@github.com/SKTBrain/KoBERT.git@master
```

**Kobert 예제**

- test/test_kobert.py

    SKT Brain의 [KoBERT](https://github.com/SKTBrain/KoBERT) 예제 [Naver Sentiment Analysis Fine-Tuning with pytorch](https://colab.research.google.com/github/SKTBrain/KoBERT/blob/master/scripts/NSMC/naver_review_classifications_pytorch_kobert.ipynb) 사용

    Google CoLAB 버전으로 만들어진 코드를 PyCharm에서 실행 가능하도록 변경함