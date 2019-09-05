# Load libraries
from flask import Flask, request, jsonify, render_template
import pandas as pd
import keras
from keras.models import load_model
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras import backend as K

from tensorflow.keras.preprocessing.text import Tokenizer
epochs=50
batch_size=128
max_words=100000
max_seq_size=256

train_df = pd.read_csv('resources/train.csv', nrows = 100000)
test_df = pd.read_csv('resources/test.csv')


transformer = Tokenizer(lower = True, filters='', num_words=max_words)
transformer.fit_on_texts(list(train_df["comment_text"].values) + list(test_df["comment_text"].values))


with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model('test.h5')

# instantiate flask 
app = Flask(__name__)

# load the model

graph = K.get_session().graph
    
@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        data = request.form.get('text')
        # Make prediction
        df = pd.DataFrame([str(data)], columns=['content'])
        print(df.head())
        pred = model.predict(data_df=df)
        print(pred)
        return render_template('index.html', prediction=pred['toxicity'][0])
    return render_template('index.html', prediction='')


# creating predict url and only allowing post requests.
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    with graph.as_default():
        s = request.args.get('s')
        score = model.predict([transformer.texts_to_sequences([s])])[0][0]
    return jsonify({'score': float(score)})
    

    

if __name__ == '__main__':
    app.run(debug=True, port=5001)
