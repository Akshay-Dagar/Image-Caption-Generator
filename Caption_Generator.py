import numpy as np
import pandas as pd
from keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
from keras.layers import GlobalAvgPool2D,Input,Dense,Dropout,Embedding,LSTM,Add
from keras.models import Model,load_model
from keras.preprocessing import sequence
import pickle
import keras
import tensorflow as tf

session=tf.Session()
keras.backend.set_session(session)


model=load_model("model.h5")
model._make_predict_function()                     #to avoid an error

#Loading the Feature Extractor (Resnet50) Model:

ftr_extractor_model=ResNet50(include_top=False,input_shape=(224,224,3))
output_layer=GlobalAvgPool2D()(ftr_extractor_model.output)
ftr_extractor_model=Model(ftr_extractor_model.input,output_layer)

ftr_extractor_model._make_predict_function() 

#graph=tf.get_default_graph()

#Loading the word2idx and idx2word mappings:

with open("word2idx.pkl",'rb') as f:
    word2idx=pickle.load(f)
    
with open("idx2word.pkl",'rb') as f:
    idx2word=pickle.load(f)

#Function to extract image features using Resnet50 Model:

def encode_image(file):

    img=cv2.imread(file)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(224,224))
    img=preprocess_input(img)                      

    ftrs=ftr_extractor_model.predict(img.reshape(1,img.shape[0],img.shape[1],img.shape[2])).flatten()

    ftrs=ftrs.reshape(1,ftrs.shape[0])
        
    return ftrs

#Function to generate Captions given the extracted features:

def predict(ftrs): 

    maxlen=30
        
    input_string='<s>'                            
        
    for i in range(maxlen):                     
            
        input_seq=[word2idx[word] for word in input_string.split(' ')]            
            
        input_sequence=np.array(sequence.pad_sequences([input_seq],maxlen=maxlen,padding='post')[0])
        input_sequence=input_sequence.reshape((1,input_sequence.shape[0]))
            
        pred=model.predict([ftrs,input_sequence])
        idx=pred.argmax()                         
        word=idx2word[idx]                        
            
        input_string+=' '+word
            
        if word=='</s>':                          
            break
        
    predicted_caption=input_string.split(' ')[1:-1]         
    predicted_caption=' '.join(predicted_caption)           
        
    return predicted_caption

#Function to call the other 2 functions:

def predict_captions(file):

    with session.as_default():
        with session.graph.as_default():

            ftrs=encode_image(file)
            caption=predict(ftrs)
                    
            return caption

