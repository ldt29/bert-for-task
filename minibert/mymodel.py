import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from bert import BertModel
from utils import *
class BertForSequenceClassification(torch.nn.Module):
    def __init__(self, model_name_or_path, num_labels):
        super(BertForSequenceClassification,self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(model_name_or_path)
        config=self.bert.config
        # define the classifier layer, and dropout function
        # todo 
        self.dropout=nn.Dropout(config.hidden_dropout_prob)
        self.classifier=nn.Linear(config.hidden_size,num_labels)


    def forward(self,input_ids=None,attention_mask=None,labels=None,):
        # if labels is None, return the logits; otherwise, return a tuple (loss, logits)

        # add the classifier layer to the pooled output of bert layer, and calculate the loss w.r.t. labels if possible
        # todo 
        # define the output
        outputs=self.bert(input_ids,attention_mask)
        pooled_output=outputs["pooler_output"]
        pooled_output=self.dropout(pooled_output)
        logits=self.classifier(pooled_output)
        if labels is not None:
            loss_fct=CrossEntropyLoss()
            loss=loss_fct(logits.view(-1,self.num_labels),labels.view(-1))
            return loss,logits
        else:
            return logits
        


class BertForCLSMLM(torch.nn.Module):
    def __init__(self, model_name_or_path, num_labels):
        super(BertForCLSMLM,self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(model_name_or_path)
        self.config=config=self.bert.config

        # define the layers needed
        # todo 
        self.dropout=nn.Dropout(config.hidden_dropout_prob)
        self.classifier=nn.Linear(config.hidden_size,num_labels)


        self.mlm_dense=nn.Linear(config.hidden_size,config.hidden_size)
        self.mlm_norm=nn.LayerNorm(config.hidden_size)
        self.mlm_activation=nn.ReLU()
        self.mlm_classifier=nn.Linear(config.hidden_size,config.vocab_size)
        self.mlm_loss_fct=CrossEntropyLoss()
        self.cls_loss_fct=CrossEntropyLoss()


    def forward(self,input_ids=None,attention_mask=None,labels=None,mlm_input_ids=None,mlm_attention_mask=None,mlm_labels=None):
        # if labels is None, return the classification logits; 
        # if the labels is not None but mlm_labels is None, return a tuple (cls_loss, classification logits)
        # if the labels and mlm_labels are not None, return a triple (cls_loss, mlm_loss, classification logits)
        # in our code, it is not possible that the labels is None while mlm_labels is not None

        # MLM part 
        # -- add the dense layer with nonlinearity and layer normalization to the sequence output of bert, and then map it into vocab_size with another dense layer
        # -- calculate the loss w.r.t. mlm_labels if mlm_labels is not None

        # todo 
        outputs = self.bert(input_ids,attention_mask)
        pooled_output=self.dropout(outputs["pooler_output"])
        logits=self.classifier(pooled_output)
        if labels is None:
            return logits
        else:
            cls_loss=self.cls_loss_fct(logits.view(-1,self.num_labels),labels.view(-1))
            if mlm_labels is None:
                return cls_loss,logits
            else:
                mlm_outputs=self.bert(mlm_input_ids,mlm_attention_mask)
                mlm_sequence_output=mlm_outputs["last_hidden_state"]
                mlm_sequence_output=self.mlm_norm(mlm_sequence_output)
                mlm_sequence_output=self.mlm_activation(mlm_sequence_output)
                mlm_sequence_output=self.mlm_dense(mlm_sequence_output)
                mlm_logits=self.mlm_classifier(mlm_sequence_output)
                mlm_loss=self.mlm_loss_fct(mlm_logits.view(-1,self.config.vocab_size),mlm_labels.view(-1))
                return cls_loss,mlm_loss,logits
                

