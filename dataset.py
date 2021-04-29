import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

from sklearn.model_selection import train_test_split

from keras.preprocessing.sequence import pad_sequences

import pandas as pd
from transformers import (
    BertTokenizer,
)


def import_data(path):
    data = pd.read_csv(path, encoding="ISO-8859-1", error_bad_lines=False)
    return data


def split_data(data):
    training, testing = train_test_split(data, test_size=0.2)
    return training, testing


def convert_dataframe_to_data(input_data):
    dataset = []
    for sentence in range(len(input_data.index)):
        labels, text = input_data['labels'][input_data.index[sentence]], input_data['text'][input_data.index[sentence]]
        labels, text = labels.split(' '), text.split(' ')
    if len(labels) == len(text):
        sentence_tuple = (text, labels)
        dataset.append(sentence_tuple)
    return dataset


def tags_and_tag_to_idx(train_data, test_data):
    tags = set()
    for pair in train_data:
        tags = set.union(tags, set(pair[1]))
    for pair in test_data:
        tags = set.union(tags, set(pair[1]))
    tag_to_idx = {t: i for i, t in enumerate(tags)}
    tag_to_idx['<PAD>'] = -1
    return tags, tag_to_idx


max_seq_length = 160
batch_size = 8
epochs = 3


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)


def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        tokenized_sentence.extend(tokenized_word)

        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


def prepair_train_data(train_data, tag_to_idx):
    tokenized_sents_and_labels = [
        tokenize_and_preserve_labels(*sent_lab) for sent_lab in train_data
    ]

    print(f"Tokens and tags after BERT tokenization:\n{tokenized_sents_and_labels[3]}\n\n")

    input_ids = pad_sequences(
        [
            tokenizer.convert_tokens_to_ids(sent_lab[0])
            for sent_lab in tokenized_sents_and_labels
        ],
        maxlen=max_seq_length,
        dtype="long",
        value=0.0,
        truncating="post",
        padding="post",
    )

    tags = pad_sequences(
        [
            [tag_to_idx.get(l) for l in sent_lab[1]]
            for sent_lab in tokenized_sents_and_labels
        ],
        maxlen=max_seq_length,
        value=tag_to_idx["<PAD>"],
        padding="post",
        dtype="long",
        truncating="post",
    )

    # converting to tensors
    train_inputs = torch.tensor(input_ids)
    train_tags = torch.tensor(tags)

    # getting masks to know where pad characters are
    train_masks = (train_inputs > 0).float()

    print(f"Sentence ids:\n{train_inputs[3]}\n\n")
    print(f"Tags:\n{train_tags[3]}\n\n")
    print(f"Masks:\n{train_masks[3]}\n\n")

    train_tensor_dataset = TensorDataset(train_inputs, train_masks, train_tags)
    train_sampler = RandomSampler(train_tensor_dataset)
    train_data_loader = DataLoader(
        train_tensor_dataset, sampler=train_sampler, batch_size=batch_size
    )
    return train_tensor_dataset, train_sampler, train_data_loader


def prepair_test_data(test_data, tag_to_idx):
    tokenized_sents_and_labels = [
        tokenize_and_preserve_labels(*sent_lab) for sent_lab in test_data
    ]

    input_ids = pad_sequences(
        [
            tokenizer.convert_tokens_to_ids(sent_lab[0])
            for sent_lab in tokenized_sents_and_labels
        ],
        maxlen=max_seq_length,
        dtype="long",
        value=0.0,
        truncating="post",
        padding="post",
    )

    tags = pad_sequences(
        [
            [tag_to_idx.get(l) for l in sent_lab[1]]
            for sent_lab in tokenized_sents_and_labels
        ],
        maxlen=max_seq_length,
        value=tag_to_idx["<PAD>"],
        padding="post",
        dtype="long",
        truncating="post",
    )

    test_inputs = torch.tensor(input_ids)
    test_tags = torch.tensor(tags)
    test_masks = (test_inputs > 0).float()

    test_tensor_dataset = TensorDataset(test_inputs, test_masks, test_tags)
    test_data_loader = DataLoader(test_tensor_dataset, batch_size=1)
    return test_tensor_dataset, test_data_loader

