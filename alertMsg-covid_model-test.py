import torch
from torch.utils.data import Dataset
from kobert.utils import get_tokenizer
import numpy as np
import pandas as pd
import gluonnlp as nlp
from kobert.pytorch_kobert import get_pytorch_kobert_model
from tqdm import tqdm

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i],))

    def __len__(self):
        return (len(self.labels))


device = None
# GPU 사용 시
if torch.cuda.is_available():
    print("GPU 사용...")
    device = torch.device("cuda")
# CPU 사용 시
else:
    print("CPU 사용...")
    device = torch.device("cpu")

model = torch.load("model_covid-classification.pt")
model.to(device)
model.eval()

bertmodel, vocab = get_pytorch_kobert_model()
# 기본 Bert Tokenizer 사용
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
max_len = 64

# 테스트 문장 예측
# {0: '보건소방문', 1: '캠페인', 2: '확진자발생'}
test_sentence = "[순천시청] 코로나19 감염이 인근(목포, 광주)에서 지속 발생하고 있습니다. 개개인이 방역주체가 되어 마스크 착용 등 방역수칙을 반드시 준수 바랍니다. "
test_label = 1

unseen_test = pd.DataFrame([[test_sentence, test_label]], columns = [['MESSAGE', 'CATEGORY']])
unseen_values = unseen_test.values
test_set = BERTDataset(unseen_values, 0, 1, tok, max_len, True, False)
test_input = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=5)

for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_input)):
  token_ids = token_ids.long().to(device)
  segment_ids = segment_ids.long().to(device)
  valid_length= valid_length
  out = model(token_ids, valid_length, segment_ids)
  print(out)

