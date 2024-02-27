import os
import sys
import logging
import datasets

import torch.nn as nn

import pandas as pd
import numpy as np

from transformers import BertTokenizerFast, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.model_selection import train_test_split

# train = pd.read_csv("./corpus/imdb/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
# test = pd.read_csv("./corpus/imdb/testData.tsv", header=0, delimiter="\t", quoting=3)
train = pd.read_csv("F:\\deeplearning\\test2\\imdb_sentiment_analysis_torch\\labeledTrainData.tsv", header=0,
                    delimiter="\t", quoting=3)
test = pd.read_csv("F:\\deeplearning\\test2\\imdb_sentiment_analysis_torch\\testData.tsv", header=0,
                   delimiter="\t", quoting=3)

class BertScratch(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
#问题是在这一句，这个地方没有将不同的模型带入一个分类，我现在像的是，将不同的输入，改为不同的对应的函数，仅在输入为训练的时候输入
#我在python中常见的是，在train——loop中包含loss，这个在这个位置包含，然后稍微有一点输入错误，或者说，这个类的本质就是一个训练池。    
        loss_fct = nn.CrossEntropyLoss()
        if labels != None:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        else:
            loss = None
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))
#日志文件
    train, val = train_test_split(train, test_size=.2)

    train_dict = {'label': train["sentiment"], 'text': train['review']}
    val_dict = {'label': val["sentiment"], 'text': val['review']}
    test_dict = {"text": test['review']}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True)


    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)
#到这个地方完成了基本的读取文件
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#在这个地方就是自定义的模型BertScratch了
    model = BertScratch.from_pretrained('bert-base-uncased')

    metric = datasets.load_metric("accuracy")
#在中国地方加载了文件的类，类似于goole的pickle文件

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    training_args = TrainingArguments(
        output_dir='./checkpoint',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=6,  # batch size per device during training
        per_device_eval_batch_size=12,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=100,
        save_strategy="no",
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=tokenized_train,  # training dataset
        eval_dataset=tokenized_val,  # evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
#这一句就是普通的训练模型
    trainer.train()
#目前是这一句有问题，tokenized_test文件未具有一个特定的属性，导致其中的函数调用此属性时，发生问题
    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    print(test_pred)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("F:\\deeplearning\\test2\\imdb_sentiment_analysis_torch\\result\\bert_scratch.csv", index=False, quoting=3)
    logging.info('result saved!')
   #目前是这一句有问题，tokenized_test文件未具有一个特定的属性，导致其中的函数调用此属性时，发生问题 