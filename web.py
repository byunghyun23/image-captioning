import gradio as gr
import numpy as np
import pickle
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.models import Model
from keras_preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import load_model
from PIL import Image
import tensorflow as tf


def encode_image(img, width, height, encode_model, preprocess_input, output_dim):
    img = img.resize((width, height), Image.ANTIALIAS)
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = encode_model.predict(x)
    x = np.reshape(x, output_dim)

    return x


def upload_image(image):
    img = encode_image(image, width, height, encode_model, preprocess_input, output_dim).reshape((1, output_dim))

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

    caption = in_text.split()
    caption = caption[1:-1]
    caption = ' '.join(caption)

    return caption


### Setting
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

model_name = 'caption_model.h5'

encode_model = InceptionV3(weights='imagenet')
encode_model = Model(encode_model.input, encode_model.layers[-2].output)

caption_model = load_model(model_name)


### Gradio
title = 'Image Captioning'
description = 'Ref: https://github.com/byunghyun23/image-captioning'
image_input = gr.inputs.Image(label='Input image', type='pil')
output_text = gr.outputs.Textbox(label='Caption')
custom_css = '#component-12 {display: none;} #component-1 {display: flex; justify-content: center; align-items: center;} img.svelte-ms5bsk {width: unset;}'

iface = gr.Interface(fn=upload_image, inputs=image_input, outputs=output_text,
                     title=title, description=description, css=custom_css)
iface.launch(server_port=8080)