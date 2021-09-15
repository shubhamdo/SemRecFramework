# https://www.sbert.net/docs/training/overview.
# paraphrase-distilroberta-base-v1
# sentence-transformers/paraphrase-xlm-r-multilingual-v1
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, util, models
from sentence_transformers.evaluation import BinaryClassificationEvaluator, EmbeddingSimilarityEvaluator
import pandas as pd

# from transformers import BertTokenizer, BertForMaskedLM


word_embedding_model = models.Transformer('sentence-transformers/paraphrase-distilroberta-base-v1', max_seq_length=512)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
# model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# [x] Import Training Data
negativeRecords = pd.read_csv(filepath_or_buffer="H:/Thesis/Output_Data/TrainingData/25/Negative_TrainingData.csv"
                              , sep=";", names=['inputA', 'inputB', 'similarity'])
positiveRecords = pd.read_csv(filepath_or_buffer="H:/Thesis/Output_Data/TrainingData/25/Positive_TrainingData.csv"
                              , sep=";", names=['inputA', 'inputB', 'similarity'])

# [x] Create Input Samples for Training the Semantic Model i.e SBERT
train_examples = []
neg = [InputExample(texts=[x[0], x[1]], label=x[2]) for i, x in negativeRecords.iterrows()]
pos = [InputExample(texts=[x[0], x[1]], label=x[2]) for i, x in positiveRecords.iterrows()]
train_examples = neg + pos
train_examples = train_examples[0:len(train_examples)-40]
dev_examples = train_examples[0:100]
test_examples = train_examples[len(train_examples)-40:]
# train_examples = train_examples[:10]
del neg
del pos
# train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
#                   InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)

"""
# [x] labels need to be 0 for dissimilar pairs and 1 for similar pairs.
evalNegativeRecords = negativeRecords.copy()
evalPositiveRecords = positiveRecords.copy()
evalPositiveRecords['similarity'] = 1
evalNegativeRecords['similarity'] = 0
frames = [evalPositiveRecords, evalNegativeRecords]
evalTestingDataSet = pd.concat(frames)

# [x] Find Evaluation Code
evaluator = BinaryClassificationEvaluator(evalTestingDataSet['inputA'].tolist(), evalTestingDataSet['inputB'].tolist(),
                                          evalTestingDataSet['similarity'].tolist(), batch_size=2,
                                          show_progress_bar=True,
                                          write_csv=True, name="25_Model")
"""

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_examples, name='sts-dev')
# [x] Create Code for Running the Training

# Tune the model
train_loss = losses.CosineSimilarityLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=2, warmup_steps=100, evaluator=evaluator, output_path="H:/Thesis/Output_Data/TrainingData/25/Test")

# [x] Save the model
# model.save(path="H:/Thesis/Output_Data/TrainingData/25/Test", model_name="paraphrase-distilroberta-base-jobdesc-v1")

# Concatenation of Negative and Positive Records
# frames = [negativeRecords, positiveRecords]
# testingDataSet = pd.concat(frames)

# Two lists of sentences
# Compute embedding for both lists
