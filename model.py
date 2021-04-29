import torch
from tqdm import trange, tqdm_notebook

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from transformers import (
    BertForTokenClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    AdamW,
)

max_seq_length = 160
batch_size = 8
epochs = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased",
    # num_labels=len(tag_to_idx) - 1,  # excluding padding tag
    output_attentions=False,
    output_hidden_states=False,
)

FULL_FINETUNING = True  # We will be updating weights, set to False if you want to train only a linear classifier

if FULL_FINETUNING:
    param_optimizer = list(bert_model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.0,
        },
    ]
else:  # Just training the linear classifier on top of BERT and keeping all other weights fixed
    param_optimizer = list(bert_model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-8)

# Total number of training steps is number of batches * number of epochs.
# total_steps = len(train_data_loader) * epochs

# Create the learning rate scheduler.train_tensor_dataset, train_sampler, train_data_loader


# scheduler = get_linear_schedule_with_warmup(
    # optimizer, num_warmup_steps=0, num_training_steps=total_steps
# )


max_grad_norm = 1.0  # for gradient clipping


def bert_model_train(
    bert_model, train_data_loader, epochs, max_grad_norm, optimizer, scheduler
):
    loss_values = []

    for _ in trange(epochs, desc="Epoch"):
        # Put the model into training mode.
        bert_model.train()
        # Reset the total loss for this epoch.
        total_loss = 0

        # Training loop
        epoch_iterator = tqdm_notebook(train_data_loader)
        for step, batch in enumerate(epoch_iterator):
            # add batch to gpu
            batch = tuple(tup.to(device) for tup in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # Clearing previously calculated gradients before performing a backward pass
            bert_model.zero_grad()

            # forward pass
            # This will return the loss (rather than the model output)
            # because we have provided the `labels`. Otherwise it returns labels
            outputs = bert_model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )

            # get the loss. Transformer models' output is a tuple
            loss = outputs[0]

            # Perform a backward pass to calculate the gradients
            loss.backward()

            # Add to total train loss
            total_loss += loss.item()

            # Clip the norm of the gradient
            # This is to help prevent the "exploding gradients" problem
            torch.nn.utils.clip_grad_norm_(
                parameters=bert_model.parameters(), max_norm=max_grad_norm
            )

            # update parameters
            optimizer.step()

            # Update the learning rate
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_data_loader)
        print(f"Average train loss: {avg_train_loss}")

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

    sns.set(style="darkgrid")
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    plt.plot(loss_values, label="training loss")

    plt.title("Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()


def one_sentence_prediction_bert(test_sentence, bert_model, tokenizer, tag_values):
    tokenized_sentence = tokenizer.encode(test_sentence)
    input_ids = torch.tensor([tokenized_sentence]).cuda()

    with torch.no_grad():
        output = bert_model(input_ids)
    label_indices = np.argmax(output[0].to("cpu").numpy(), axis=2)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.to("cpu").numpy()[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(tag_values[label_idx])
            new_tokens.append(token)

    return zip(new_tokens, new_labels)
