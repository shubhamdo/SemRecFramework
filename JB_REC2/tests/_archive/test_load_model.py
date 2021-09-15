

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForNextSentencePrediction

BERT_CLASS = BertForNextSentencePrediction

# Make sure all the files are in same folder, i.e vocab , config and bin file
PRE_TRAINED_MODEL_NAME_OR_PATH = 'H:/Thesis/Output_Data/model'

model = BERT_CLASS.from_pretrained(PRE_TRAINED_MODEL_NAME_OR_PATH, cache_dir=None)