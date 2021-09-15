# https://www.sbert.net/docs/training/overview.
# paraphrase-distilroberta-base-v1
# sentence-transformers/paraphrase-xlm-r-multilingual-v1
from sentence_transformers import SentenceTransformer, util, models
from flask import Flask, render_template, request



# [x] Import Model
word_embedding_model = models.Transformer(model_name_or_path='H:/Thesis/Output_Data/TrainingData/25/Test'
                                          , max_seq_length=512)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# [x] Pass the Records
# sentences1 = [positiveRecords['inputA'].iloc[0]]
# sentences2 = [positiveRecords['inputB'].iloc[0]]
sentences1 = ["Some Text was in the this"]
sentences2 = ["Some Text was in the this"]

# [x] Write Inference Code
# embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings1 = model.encode(sentences1, output_value = 'token_embeddings') # convert_to_tensor=True)
embeddings2 = model.encode(sentences1, output_value = 'token_embeddings') # convert_to_tensor=True)
# embeddings2 = model.encode(sentences2, convert_to_tensor=True)

# [x] Get Cosin value
cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
ag = []
# Output the pairs with their score
for i in range(len(sentences1)):
    # print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
    print("{} \t\n {} \t\n Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
    ag.append(cosine_scores[i][i])

