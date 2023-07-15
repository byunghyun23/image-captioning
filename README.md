# Image Captioning Using Inception-v3 and LSTM

## Introduction
This is a TensorFlow implementation for Image Captioning using Inception-v3 and LSTM.

![image](https://github.com/byunghyun23/image-captioning/blob/main/assets/fig1.png)
![image](https://github.com/byunghyun23/image-captioning/blob/main/assets/fig2.png)

## Dataset
For training the model, you need to download the MS COCO dataset [link1](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=261) or [link2](https://cocodataset.org).  
The name of the json file used in this project is MSCOCO_train_val_Korean.

## Download images
Before running, you need to create a directory like "data" to store data.  
You can see that the data has been downloaded to the directory you created by running
```
python download.py
```

## Embedding
We use GloVe for Word Embedding.  
You can get files like this
```
embeddings_index.pkl
information.pkl
glove.model
```
by running
```
python embedding.py
```

## Preprocessing
We obtained the features of images using Inception-v3 and use it as input for learning the captioning model.  
So you can label image features and captions and you can also get a dataset classified as training and test by running
```
python preprocessing.py
```
The generated files are:
```
idx_to_word.pkl
word_to_idx.pkl
train_captions.pkl
test_captions.pkl
train_encoding.pkl
test_encoding.pkl
```

## Train
```
python train.py
```
After training, the following model is created.
```
caption_model.h5
```

## Predict
You can get the caption of an image by running
```
python predict.py --file_name file_name
```

## Demo
Also, you can also use the model using Gradio by running
```
python web.py
```
![image](https://github.com/byunghyun23/image-captioning/blob/main/assets/fig3.PNG)

