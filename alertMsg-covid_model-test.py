import torch
from torch import nn
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


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=3,  # 분류 개수에 따라 수정
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


def testModel():
    torch.multiprocessing.freeze_support()
    torch.cuda.empty_cache()

    bertmodel, vocab = get_pytorch_kobert_model()
    # 기본 Bert Tokenizer 사용
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    max_len = 64

    device = None
    model = BERTClassifier(bertmodel, dr_rate=0.5)
    # GPU 사용 시
    if torch.cuda.is_available():
        print("GPU 사용...")
        device = torch.device("cuda")
        model.load_state_dict(torch.load("model_covid-classification_state-dict.pt"), strict=False)
        model.to(device)
    # CPU 사용 시
    else:
        print("CPU 사용...")
        device = torch.device("cpu")
        model.load_state_dict(torch.load("model_covid-classification_state-dict.pt", map_location=device))
    # model = torch.load("model_covid-classification.pt")
    model.eval()

    # 테스트 문장 예측
    mapping_dict = {0: '보건소방문', 1: '캠페인', 2: '확진자발생'}
    # test_sentence = "[순천시청] 코로나19 감염이 인근(목포, 광주)에서 지속 발생하고 있습니다. 개개인이 방역주체가 되어 마스크 착용 등 방역수칙을 반드시 준수 바랍니다. "
    # test_sentence = "[진주시청]6.29 20:38~22:35 국가대표고기집 호탄점을 이용하신 분은 진주시보건소(749-5714)로 연락바랍니다. "
    # test_sentence = "[의정부시청] 코로나19 추가 확진자 발생【양주시 거주(2명)】 의정부시 관내(확진자) 동선은 시 홈페이지 및 블로그 참고 바랍니다. "
    # test_sentence = "[서울시청] 04.12.(월) 00시 기준 서울시 신규 확진자 162명 발생. 자치구별 현황 및 동선 등은 bityl.co/6I7T 참고하시기 바랍니다. ☎120"
    # test_sentence = "[용인시청] 4월10일 확진자18명[(용인 2243~2260번) ▶처인구2, 기흥구3, 수지구12, 수원시1] 발생하였습니다. corona.yongin.go.kr"
    # test_sentence = "[수원시청] 4.2~4.4 수원시 팔달구 매산로2가 소재 \'차이나 24시 노래 유흥주점\' 방문자는 가까운 보건소에서 코로나19 검사를 받으시기 바랍니다."
    # test_sentence = "[중대본]4.2.~9. 성남시 분당구 소재 도우미 이용 노래방 방문자·근무자는 가까운 보건소 선별진료소에서 코로나19 검사를 받으시기 바랍니다(☎120,1339)"
    test_label = 1
    tttt = [["[중대본]4.2.~9. 성남시 분당구 소재 도우미 이용 노래방 방문자·근무자는 가까운 보건소 선별진료소에서 코로나19 검사를 받으시기 바랍니다(☎120,1339)", 0],
            ["[용인시청] 4월10일 확진자18명[(용인 2243~2260번) ▶처인구2, 기흥구3, 수지구12, 수원시1] 발생하였습니다. corona.yongin.go.kr", 2]]
    # unseen_test = pd.DataFrame([[test_sentence, test_label]], columns=[['MESSAGE', 'CATEGORY']])
    unseen_test = pd.DataFrame(tttt, columns=[['MESSAGE', 'CATEGORY']])
    unseen_values = unseen_test.values
    test_set = BERTDataset(unseen_values, 0, 1, tok, max_len, True, False)
    test_input = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=5)

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_input)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)

        max_vals, max_indices = torch.max(out, 1)
        train_acc = (max_indices == label).sum().data.cpu().numpy() / max_indices.size()[0]
        indices = out.data.max(1).indices
        print(f"\n{tttt[batch_id][0]} : {mapping_dict[indices.item()]}")
        print(train_acc)


if __name__ == '__main__':
    testModel()
