# https://www.sbert.net/docs/training/overview.
# paraphrase-distilroberta-base-v1
# sentence-transformers/paraphrase-xlm-r-multilingual-v1
from sentence_transformers import SentenceTransformer, util, models
from flask import Flask, render_template, request

app = Flask(__name__, template_folder="JB_REC")


@app.route('/test', methods=['POST'])
def upload():
    data = request.json
    print(data)
    a = str(data['sentence1'])
    b = str(data['sentence2'])

    print(a, b)
    # [x] Import Model
    word_embedding_model = models.Transformer(model_name_or_path='H:/Thesis/Output_Data/TrainingData/25/Test'
                                              , max_seq_length=512)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # [x] Pass the Records
    # sentences1 = [positiveRecords['inputA'].iloc[0]]
    # sentences2 = [positiveRecords['inputB'].iloc[0]]
    sentences1 = [a]
    sentences2 = [b]

    # [x] Write Inference Code
    # embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings1 = model.encode(sentences1, output_value = 'sentence_embedding') # convert_to_tensor=True)
    embeddings2 = model.encode(sentences1, output_value = 'sentence_embedding') # convert_to_tensor=True)
    # embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    # [x] Get Cosin value
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    ag = []
    # Output the pairs with their score
    for i in range(len(sentences1)):
        # print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
        print("{} \t\n {} \t\n Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
        ag.append(cosine_scores[i][i])
    return str(ag)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
