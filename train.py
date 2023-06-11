import click
import os
from time import time
import pickle
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.models import Model

from keras_preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import load_model

import matplotlib.pyplot as plt


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60

    return f'{h}:{m:>02}:{s:>05.2f}'


def data_generator(descriptions, features, word_to_idx, max_len, batch, vocab_size):
    x1, x2, y = [], [], []
    n = 0

    while True:
        for key, desc_list in descriptions.items():
            n += 1
            feature = features[key + '.jpg']

            for desc in desc_list:
                seq = [word_to_idx[word] for word in desc.split(' ') if word in word_to_idx]

                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_len)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    x1.append(feature)
                    x2.append(in_seq)
                    y.append(out_seq)

            if n == batch:
                yield ([np.array(x1), np.array(x2)], np.array(y))
                x1, x2, y = [], [], []
                n = 0


@click.command()
@click.option('--model_name', default='caption_model.h5', help='Model name')
@click.option('--epochs', default=10, help='Epochs')
@click.option('--train_batch', default=8, help='Train batch')
@click.option('--test_batch', default=4, help='Test batch')
def run(model_name, epochs, train_batch, test_batch):
    # Load pkl
    with open('idx_to_word.pkl', 'rb') as f:
        idx_to_word = pickle.load(f)
    with open('word_to_idx.pkl', 'rb') as f:
        word_to_idx = pickle.load(f)
    with open('embeddings_index.pkl', 'rb') as f:
        embeddings_index = pickle.load(f)
    with open('train_captions.pkl', 'rb') as f:
        train_captions = pickle.load(f)
    with open('test_captions.pkl', 'rb') as f:
        test_captions = pickle.load(f)
    with open('train_encoding.pkl', 'rb') as f:
        train_encoding = pickle.load(f)
    with open('test_encoding.pkl', 'rb') as f:
        test_encoding = pickle.load(f)
    with open('information.pkl', 'rb') as f:
        try:
            information = pickle.load(f)
            embedding_dim = information['embedding_dim']
            output_dim = information['output_dim']
            vocab_size = information['vocab_size']
            max_len = information['max_len']
        except Exception as e:
            print(e)

    # Generate embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in word_to_idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    # print(embedding_matrix.shape)

    # Design model
    inputs1 = tf.keras.layers.Input(shape=(output_dim,))
    fe1 = tf.keras.layers.Dropout(0.5)(inputs1)
    fe2 = tf.keras.layers.Dense(256, activation='relu')(fe1)
    inputs2 = tf.keras.layers.Input(shape=(max_len,))
    se1 = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = tf.keras.layers.Dropout(0.5)(se1)
    se3 = tf.keras.layers.LSTM(256)(se2)
    # decoder1 = tf.keras.layers.add([fe2, se3])
    decoder1 = tf.keras.layers.concatenate([fe2, se3])
    decoder2 = tf.keras.layers.Dense(256, activation='relu')(decoder1)
    outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')(decoder2)
    caption_model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # caption_model.summary()

    caption_model.layers[2].set_weights([embedding_matrix])
    caption_model.layers[2].trainable = False
    caption_model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Train
    train_steps = len(train_captions) // train_batch
    test_steps = len(test_captions) // test_batch
    if not os.path.exists(model_name):
        checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='min')
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        callbacks_list = [checkpoint, es]

        train_generator = data_generator(train_captions, train_encoding, word_to_idx, max_len, train_batch, vocab_size)
        test_generator = data_generator(test_captions, test_encoding, word_to_idx, max_len, test_batch, vocab_size)

        start = time()
        history = caption_model.fit_generator(generator=train_generator, steps_per_epoch=train_steps, validation_data=test_generator,
                            validation_steps=test_steps,
                            epochs=epochs, callbacks=callbacks_list, workers=1)

        print(f'Training took:', {hms_string(time() - start)})

        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        train_len = np.arange(len(train_loss))
        val_len = np.arange(len(val_loss))
        plt.plot(train_len, train_loss, marker='.', c='blue', label="Train-set Loss")
        plt.plot(val_len, val_loss, marker='.', c='red', label="Validation-set Loss")
        plt.legend(loc='upper right')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        # plt.show()
        plt.title('Loss')
        plt.savefig('./loss.png')
        plt.clf()
    else:
        caption_model = load_model(model_name)

    # Test
    start = 'startseq'
    stop = 'endseq'

    for i in range(5):
        z = random.randint(0, 1000)
        pic = list(test_encoding.keys())[z]
        image = test_encoding[pic].reshape((1, output_dim))

        x = plt.imread('data/' + pic)

        in_text = start
        for i in range(max_len):
            sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
            sequence = pad_sequences([sequence], maxlen=max_len)
            yhat = caption_model.predict([image, sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = idx_to_word[yhat]
            in_text += ' ' + word

            if word == stop:
                break
        # print(test_captions)
        print(test_captions[pic[:-4]])

        title = in_text.split()
        title = title[1:-1]
        title = ' '.join(title)

        print('Caption:', title)

        plt.rc('font', family='Malgun Gothic')
        plt.imshow(x)
        plt.title(title)
        plt.show()


if __name__ == '__main__':
    run()