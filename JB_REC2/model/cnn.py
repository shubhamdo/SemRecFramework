import pandas as pd
import numpy as np

# [x] Import Training Data
from tensorflow import keras

negativeRecords = pd.read_csv(filepath_or_buffer="H:/Thesis/Output_Data/TrainingData/25/Negative_TrainingData.csv"
                              , sep=";", names=['inputA', 'inputB', 'similarity'])
positiveRecords = pd.read_csv(filepath_or_buffer="H:/Thesis/Output_Data/TrainingData/25/Positive_TrainingData.csv"
                              , sep=";", names=['inputA', 'inputB', 'similarity'])

# Create Categorical
negativeRecords['class'] = 0
# negativeRecords = negativeRecords[:5000]
positiveRecords['class'] = 1
# positiveRecords = positiveRecords[:5000]
data = pd.concat([negativeRecords, positiveRecords]).reset_index(drop=True)
data = data.drop(['similarity'], axis=1)

import nltk
import tqdm
import pandas as pd
import numpy as np
import tensorflow as tf
import re, nltk, gensim
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Input, \
    BatchNormalization, Bidirectional, concatenate, Dropout, Conv1D, \
    MaxPooling1D, Flatten, add, Lambda
import tensorflow.keras.backend as K

nltk.download('stopwords')
nltk.download('wordnet')


def load_data(dataset):
    train = pd.read_csv(dataset)
    train.dropna(axis=0, inplace=True)
    return train


# data = load_data('C:/Users/shubham/Downloads/train.csv/train.csv')
# data = data[:10000]


# Creating two list one for left and another for the right question
def list_data(train):
    q1 = pd.Series(train.question1.tolist()).astype(str)
    q2 = pd.Series(train.question2.tolist()).astype(str)
    return q1, q2


q1, q2 = data['inputA'], data['inputB']
# q1, q2 = data['inputA'], data['inputB']
# q1, q2 = list_data(data)
# data['class'].value_counts()
data['class'].value_counts()


def text_clean(corpus):
    cleaned_corpus = pd.Series()
    for row in corpus:
        qs_list = []
        for word in row.split():
            word = word.lower()
            word = re.sub(r"[^a-zA-Z0-9^.']", " ", word)
            p1 = re.sub(pattern='[^a-zA-Z0-9]', repl=' ', string=word)
            qs_list.append(p1)
        cleaned_corpus = cleaned_corpus.append(pd.Series(' '.join(qs_list)))
    return cleaned_corpus


all_corpus = q1.append(q2)
all_corpus = text_clean(all_corpus)


# The data is in format like all q1 are the in the starting
# rows of all_corpus
# then once q1 gets finished, q2 starts. So again
# separating q1 and q2 and merging them into a data frame.
def clean_data(all_corpus, q1, q2, train):
    q1 = all_corpus[0:q1.shape[0]]
    q2 = all_corpus[q2.shape[0]::]
    data_out = pd.DataFrame({'q1': q1, 'q2': q2})
    data_out.index = list(range(0, len(data_out)))
    # data_out['output'] = train['class']
    data_out['output'] = train['class']
    return data_out


data_new = clean_data(all_corpus, q1, q2, data)


# creating word to index using keras tokenizer
def word_to_index(all_corpus):
    lines = []
    for key in all_corpus:
        lines.append(key)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return (tokenizer.word_index)


word2index = word_to_index(all_corpus)
index2word = dict((v, k) for k, v in word2index.items())


# Loading pre-trained word vectors
def load_embedding(EMBEDDING_FILE, embedding_dim):
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
    # w2v = dict(zip(word2vec_model.wv.index2word,
    #                word2vec_model.wv.syn0))

    # This will be the embedding matrix
    embeddings = 1 * np.random.randn(len(word2index)
                                     + 1, embedding_dim)
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix
    words = list(word2vec_model.index_to_key)
    # for word, index in word2index.items():
    for word, index in word2index.items():
        if word in words:
            embeddings[index] = word2vec_model.word_vec(word)
    return embeddings


embedding_dim = 300
EMBEDDING_FILE = 'C:/Users/shubham/Downloads/GoogleNews-vectors-negative300.bin.gz'

embeddings = load_embedding(EMBEDDING_FILE, embedding_dim)


def max_length(all_corpus):
    lines = []
    max_len = -1
    for key in all_corpus:
        for d in key:
            if len(d.split()) > max_len:
                max_len = len(d.split())
    return max_len


max_len = max_length(all_corpus)

# If len is not equal to max_len then doing post padding
max_len = 512


def create_train_data(dataset, max_length, column):
    X1 = list()
    for idx in range(len(dataset)):
        for words in (data_new.iloc[idx][[column]].values):
            numeric_seq = [word2index[word] for word in words.split() if word in word2index]
            in_seq = numeric_seq
            in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
            X1.append(in_seq)
    return X1


q1 = np.array(create_train_data(data_new, max_len, 'q1'))
q2 = np.array(create_train_data(data_new, max_len, 'q2'))


def split_train_test(q1, q2, data):
    X = np.stack((q1, q2), axis=1)
    X_train, X_test, y_train, y_test = X[:-10], \
                                       X[-10:], list(data['class'])[:-10], list(data['class'])[-10:]
    # X[-10:], list(data['class'])[:-10], list(data['class'])[-10:]
    train_q1 = X_train[:, 0]
    train_q2 = X_train[:, 1]
    test_q1 = X_test[:, 0]
    test_q2 = X_test[:, 1]
    return train_q1, train_q2, test_q1, test_q2, \
           y_train, y_test, X_train, X_test


data['class'] = data['class'].astype('float32')
train_q1, train_q2, test_q1, test_q2, y_train, \
y_test, X_train, X_test = split_train_test(q1, q2, data)
y_train = np.array(y_train)
y_test = np.array(y_test)


# Cosine distance
def cosine_distance(output):
    x, y = output[0], output[1]
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)


def euclidean_distance(output):
    x, y = output[0], output[1]
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def contrastive_loss(y_true, y_pred):
    margin = 1
    return y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0))


def accuracy(y_true, y_pred):
    return K.mean(tf.cast(tf.equal(y_true, tf.cast(y_pred < 0.5, dtype=tf.float32)), dtype=tf.float32))


def cnn_model(input_shape, embeddings, embedding_dim):
    model_input = Input(shape=(input_shape,))
    layer = Embedding(len(embeddings),
                      embedding_dim,
                      weights=[embeddings],
                      input_length=max_len,
                      trainable=False)(model_input)
    layer = Conv1D(filters=64, kernel_size=3, activation='relu')(layer)
    layer = MaxPooling1D(pool_size=2)(layer)
    layer = Dropout(0.2)(layer)
    layer = Conv1D(filters=64, kernel_size=2, activation='relu')(layer)
    layer = MaxPooling1D(pool_size=2)(layer)
    layer = Dropout(0.2)(layer)
    layer = Conv1D(filters=64, kernel_size=2, activation='relu')(layer)
    layer = MaxPooling1D(pool_size=2)(layer)
    output = Flatten()(layer)

    model = Model(inputs=model_input, outputs=output)
    model.summary()
    return model


# max_len = 512
cnn_model = cnn_model(max_len, embeddings, embedding_dim)

input_q1 = Input(shape=(max_len,))
input_q2 = Input(shape=(max_len,))

left_out = cnn_model(input_q1)
right_out = cnn_model(input_q2)

output = Lambda(euclidean_distance, name='euclidean_distance')([left_out, right_out])

cnn_model = Model(inputs=[input_q1, input_q2], outputs=output)
cnn_model.summary()

cnn_model.compile(loss=contrastive_loss, optimizer='adam', metrics=[accuracy])
# cnn_model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=[accuracy])

# callback = [ModelCheckpoint('question_pairs_weights_cnn.h5', monitor='accuracy', save_best_only=True, mode='max')]

history = cnn_model.fit([train_q1, train_q2],
                        y_train,
                        epochs=10,
                        batch_size=10)
# callbacks=callback)
y_pred=cnn_model.predict([test_q1,test_q2])
data_new_test=data_new[-10:]
data_new_test['Y_prediction']=[i for i in y_pred]
data_new_test


def pearson_correlation(y_true, y_pred):
    # Pearson's correlation coefficient = covariance(X, Y) / (stdv(X) * stdv(Y))
    fs_pred = y_pred - K.mean(y_pred)
    fs_true = y_true - K.mean(y_true)
    covariance = K.mean(fs_true * fs_pred)

    stdv_true = K.std(y_true)
    stdv_pred = K.std(y_pred)

    return covariance / (stdv_true * stdv_pred)


import pandas as pd
from scipy.stats import pearsonr

y_test = y_test.astype(int)
y_pred = y_pred.astype(int)

corr, _ = pearsonr(y_test, y_pred)
# pearson_correlation(y_test, y_pred)