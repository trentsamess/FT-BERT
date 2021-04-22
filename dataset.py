import pandas as pd

import torch

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam


def prepare_dataset(dframe):
    dataset = {}
    for sentence in range(len(dframe)):
        labels, text = dframe['labels'][sentence], dframe['text'][sentence]
        labels, text = labels.split(' '), text.split(' ')
        idx = [sentence + 1] * len(labels)
        if len(labels) == len(text):
            if type(dataset) == dict:
                dataset = {'idx': idx, 'text': text, 'labels': labels}
                dataset = pd.DataFrame(dataset)
            else:
                sentence_dict = {'idx': idx, 'text': text, 'labels': labels}
                sentence_dict = pd.DataFrame(sentence_dict)
                dataset = pd.concat([dataset, sentence_dict], ignore_index=True)
                print(len(dataset))
    dataset


class SentenceGetter(object):

    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["text"].values.tolist(),
                                                     s["labels"].values.tolist())]
        self.grouped = self.dataset.groupby("idx").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


getter = SentenceGetter(dataset)


def sentences_labels(getter):
    sentences = [" ".join([s[0] for s in sent]) for sent in getter.sentences]
    labels = [[s[1] for s in sent] for sent in getter.sentences]
    return sentences, labels


def tags_idx(dataset):
    tags_vals = list(set(dataset["labels"].values))
    tag2idx = {t: i for i, t in enumerate(tags_vals)}
    return tag2idx


MAX_LEN = 40
bs = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()


def get_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    return tokenizer


def get_tokenized_texts(sentences, tokenizer):
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    return tokenized_texts


def get_input_ids(tokenizer, tokenized_texts):
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    return input_ids


def get_attention_masks(input_ids):
    attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]
    return attention_masks


def make_a_split(input_ids,tags,attention_masks):
    tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                                random_state=2018, test_size=0.1)
    tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                 random_state=2018, test_size=0.1)
    return tr_inputs, val_inputs, tr_tags, val_tags, tr_masks, val_masks


def turn_to_tensor(tr_inputs, val_inputs, tr_tags, val_tags, tr_masks, val_masks):
    tr_inputs = torch.tensor(tr_inputs)
    val_inputs = torch.tensor(val_inputs)
    tr_tags = torch.tensor(tr_tags)
    val_tags = torch.tensor(val_tags)
    tr_masks = torch.tensor(tr_masks)
    val_masks = torch.tensor(val_masks)
    return tr_inputs, val_inputs, tr_tags, val_tags, tr_masks, val_masks


def train_load(tr_inputs, tr_masks, tr_tags):
    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)
    return train_dataloader


def valid_load(val_inputs, val_masks, val_tags):
    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)
    return valid_dataloader
