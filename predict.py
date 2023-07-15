import click
import pickle
import numpy as np
import tensorflow as tf

from keras_preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.models import Model


def encode_image(img, width, height, encode_model, preprocess_input, output_dim):
    img = img.resize((width, height), Image.ANTIALIAS)
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = encode_model.predict(x)
    x = np.reshape(x, output_dim)

    return x


@click.command()
@click.option('--file_name', default='data/000000574154.jpg', help='Input file name')
@click.option('--model_name', default='caption_model.h5', help='Model name')
def run(file_name, model_name):
    start = 'startseq'
    stop = 'endseq'

    with open('idx_to_word.pkl', 'rb') as f:
        idx_to_word = pickle.load(f)
    with open('word_to_idx.pkl', 'rb') as f:
        word_to_idx = pickle.load(f)
    with open('information.pkl', 'rb') as f:
        try:
            information = pickle.load(f)
            output_dim = information['output_dim']
            max_len = information['max_len']
            width = information['width']
            height = information['height']
            preprocess_input = information['preprocess_input']
        except Exception as e:
            print(e)

    x = Image.open(file_name)
    x.load()

    encode_model = InceptionV3(weights='imagenet')
    encode_model = Model(encode_model.input, encode_model.layers[-2].output)

    img = encode_image(x, width, height, encode_model, preprocess_input, output_dim).reshape((1, output_dim))

    caption_model = load_model(model_name)
    in_text = start
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len)
        yhat = caption_model.predict([img, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word[yhat]
        in_text += ' ' + word

        if word == stop:
            break

    title = in_text.split()
    title = title[1:-1]
    title = ' '.join(title)

    plt.rc('font', family='Malgun Gothic')
    plt.imshow(x)
    plt.title(title)
    plt.axis('off')
    plt.show()

    print('Caption:', title)


if __name__ == '__main__':
    run()
