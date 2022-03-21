from flask import Flask,render_template, request
from tensorflow import keras
import librosa
import numpy as np

emos = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def extract_mfcc(filename):
    y ,sr = librosa.load(filename, duration=5, offset=0.5)
    mfcc =np.mean(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40).T,axis=0)
    return mfcc

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload')
def upload_file1():
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file2():
   if request.method == 'POST':
      filename = './data/' + request.files['file'].filename

      loaded_model = keras.models.load_model('only_bangla.h5')
      X_mfcc = extract_mfcc(filename)
      X = [ x for x in X_mfcc]
      X= np.array(X)
      x_valid=np.expand_dims(X, -1)
      x_valid = np.expand_dims(x_valid,axis=0)
      
      Y_pred = loaded_model.predict(x_valid)
      output = [np.argmax(i) for i in Y_pred]

      return render_template('output.html', data=[emos[output[0]]])
    #   return emos[output[0]]

if __name__=="__main__":
    app.run(debug=True)