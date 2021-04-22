import numpy as np
import tokenizer as tokenizer
from sklearn.metrics import f1_score
from torch.optim import Adam

import torch

from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from tqdm import tqdm, trange

from dataset import SentenceGetter

model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx))

FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def train(train_dataloader, valid_dataloader, tags_vals, device):
    epochs = 5
    max_grad_norm = 1.0

    for _ in trange(epochs, desc="Epoch"):
        # TRAIN loop
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # forward pass
            loss = model(b_input_ids, token_type_ids=None,
                         attention_mask=b_input_mask, labels=b_labels)
            # backward pass
            loss.backward()
            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            model.zero_grad()
        # print train loss per epoch
        print("Train loss: {}".format(tr_loss / nb_tr_steps))
        # VALIDATION on validation set
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions, true_labels = [], []
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                      attention_mask=b_input_mask, labels=b_labels)
                logits = model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_mask)
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1
        eval_loss = eval_loss / nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
        pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
        valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
        print("F1-Score: {}".format(f1_score(pred_tags, valid_tags, average='weighted')))


def predict(sentence, MAX_LEN):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = []
    tags_vals = ['B-tim',
                  'I-per',
                  'I-org',
                  'B-gpe',
                  'B-art',
                  'I-art',
                  'I-nat',
                  'I-geo',
                  'B-per',
                  'B-org',
                  'O',
                  'B-geo',
                  'B-nat',
                  'B-eve',
                  'I-tim',
                  'I-eve',
                  'I-gpe']
    tokenized_text = tokenizer.tokenize(sentence)
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(tokenized_text)],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]
    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)
    with torch.no_grad():
        model.eval()
        inputs = inputs.to(device)
        masks = masks.to(device)
        logits = model(inputs, token_type_ids=None,
                       attention_mask=masks)
    logits = logits.detach().cpu().numpy()
    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    pred_tags = [[tags_vals[p_i] for p_i in p] for p in predictions]
    amount = len(sentence.split(' '))
    return pred_tags[0][:amount]
