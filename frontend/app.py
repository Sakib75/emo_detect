from flask import Flask,render_template, request,redirect
from tensorflow import keras
import librosa
import numpy as np
np.set_printoptions(suppress=True)

emos = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def extract_mfcc(filename):
    y ,sr = librosa.load(filename, duration=5, offset=0.5)
    mfcc =np.mean(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40).T,axis=0)
    return mfcc

app = Flask(__name__)

@app.route('/')
def index():
    return redirect('/upload')

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
      Y_pred_list = Y_pred.tolist()[0]
      Y_pred_list = ['{:f}'.format(x) for x in Y_pred_list]


      output = [np.argmax(i) for i in Y_pred]

      final_output = dict()
      final_output['emotional_state'] = emos[output[0]]
      final_output['overall'] = dict(zip(emos, Y_pred_list))
      print(final_output)
      return render_template('output.html', data=final_output)
    #   return emos[output[0]]

if __name__=="__main__":
    app.run(debug=True)