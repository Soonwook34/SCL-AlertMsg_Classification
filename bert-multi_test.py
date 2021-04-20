import torch
from torch import nn
from torch.nn import DataParallel, BCEWithLogitsLoss
from torch.utils.data import Dataset
from pytorch_transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BertModel
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm
import pandas as pd
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup


class BertForMultiLabelSequenceClassification():
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """

    def __init__(self, config, num_labels=2):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


model = BertForMultiLabelSequenceClassification(config="bert-base-multilingual-cased")
print(model)
