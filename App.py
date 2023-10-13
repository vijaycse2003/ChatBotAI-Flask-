from flask import Flask,render_template,redirect,request,jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
import random
import joblib as jl



app=Flask(__name__)
app.secret_key="12345"






with open('content.json') as content:
    data1=json.load(content)

tag=[]
inputs=[]
responses={}

for intent in data1['intents']:
    responses[intent['tag']]=intent['responses']
    for lines in intent['patterns']:
        inputs.append(lines)
        tag.append(intent['tag'])

data=pd.DataFrame({"inputs":inputs,"tags":tag})


@app.route('/')
def Home():
    return render_template('Home.html',titleName='HackAI Chatbot | Main Page')





@app.route('/chat',methods=['GET','POST'])
def ChatBot():
   
    model=jl.load('predict_model.joblib')
    le=LabelEncoder()
    le.fit_transform(data['tags'])
    tokenizer=Tokenizer(num_words=2000)
    res=0    
    if request.method == 'POST':
        message=str(request.form['userReply'])
        bot_response=generate_res(tokenizer,message,le,model)
        print('Message  -->',message)
    
        while True:

           if message!='':
               bot_response = str(bot_response)      
               print(bot_response)
               return jsonify({'status':'OK','answer':bot_response})
           elif message == "bye":
               bot_response='Hope to see you soon'
               print(bot_response)
               return jsonify({'status':'OK','answer':bot_response})
               break
         


    return render_template('chat.html',titleName='HackAI Chatbot | Chat')




def generate_res(tokenizer,prediction_input,le,model):
    
    texts_p=[]
    prediction_input=[letters.lower() for letters in prediction_input ]
    prediction_input=''.join(prediction_input)
    texts_p.append(prediction_input)
    print('Texts plain :',texts_p)
    print('Prediction Input  :',prediction_input)
    prediction_input=tokenizer.texts_to_sequences(texts_p)
    prediction_input=np.array( prediction_input).reshape(-1)
    prediction_input=pad_sequences([prediction_input],9)
    output=model.predict(prediction_input)
    print('Before output',output)
    output=output.argmax()
    output=output 
    print('After output',output)
    
    response_tag=le.inverse_transform([output])[0]
    res=random.choice(responses[response_tag])
    print(f"HackAI ({response_tag}) :",res)
    return res


if __name__ == "__main__":
    app.run(debug=True)