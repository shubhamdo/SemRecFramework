import transformers
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, util, models
from sentence_transformers.evaluation import BinaryClassificationEvaluator, EmbeddingSimilarityEvaluator
import pandas as pd
from JB_REC2.connections.mongoConnection import insertCollection, getCollection


word_embedding_model = models.Transformer('sentence-transformers/paraphrase-distilroberta-base-v1', max_seq_length=512)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# [x] Import Training Data
# Change the path here
# datapath = "H:/Thesis/Output_Data/TrainingData/25/Positive_TrainingData.csv"
# datapath1 = "H:/Thesis/Output_Data/TrainingData/25/Negative_TrainingData.csv"
# negativeRecords = pd.read_csv(filepath_or_buffer=datapath
#                               , sep=";", names=['inputA', 'inputB', 'similarity'])
# positiveRecords = pd.read_csv(filepath_or_buffer=datapath1
#                               , sep=";", names=['inputA', 'inputB', 'similarity'])

negativeRecords = getCollection("trainingData", "negativeSamples")
positiveRecords = getCollection("trainingData", "positiveSamples")

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


evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_examples, name='sts-dev')

# [x] Create Code for Running the Training
# Tune the model
# Change Model Path
# savemodepath = "H:/Thesis/Output_Data/TrainingData/25"
train_loss = losses.CosineSimilarityLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=2, warmup_steps=100,
          optimizer_class = transformers.AdamW, evaluator=evaluator)
          # output_path=savemodepath + "/Test")


