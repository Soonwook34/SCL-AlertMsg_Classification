from gluonnlp.data import SentencepieceTokenizer, BERTSPTokenizer, BERTSentenceTransform
from kobert.utils import get_tokenizer

from kobert.pytorch_kobert import get_pytorch_kobert_model

bertmodel, vocab = get_pytorch_kobert_model()
tokenizer = get_tokenizer()
sampleText = "[순천시청] 코로나19 감염이 인근(목포, 광주)에서 지속 발생하고 있습니다. 개개인이 방역주체가 되어 마스크 착용 등 방역수칙을 반드시 준수 바랍니다. "
sampleText = "마치 미국애니에서 튀어나온듯한 창의력없는 로봇디자인부터가,고개를 젖게한다"
print(vocab)

tok = BERTSPTokenizer(tokenizer, vocab, lower=False)
print(tok)
print(tok(sampleText))
transform = BERTSentenceTransform(tok, max_seq_length=32, pad=True, pair=False)
print(transform(sampleText))

sp = SentencepieceTokenizer(tokenizer)
print(sp)
print(sp(sampleText))

#
# transform = BERTSentenceTransform(tok, max_seq_length=32, pad=True, pair=False)
# transform2 = BERTSentenceTransform(sp, max_seq_length=32, vocab=None, pad=True, pair=False)
# print(transform("한국어 모델을 공유합니다."))
# print(transform2("한국어 모델을 공유합니다."))
