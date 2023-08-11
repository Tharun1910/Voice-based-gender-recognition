from flask import Flask, render_template, request
import pandas as pd
import joblib
from itertools import cycle
import librosa.display
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
from keras.models import load_model



# Load the machine learning model
model = load_model("cnn.h5")

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     # Get the uploaded file from the HTML form
#     uploaded_file = request.files['file']

app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    # Get the uploaded CSV file
    file = request.files['file']
    if not file:
        return render_template('index.html', message='No file uploaded')

    # sns.set_theme(style="white", palette=None)
    color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    y, sr = librosa.load(file)
    # y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    # pd.Series(y_trimmed).plot(figsize=(10, 5),
    #               lw=1,
    #               title='Raw Audio Trimmed Example',
    #               color=color_pal[1])
    feature = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    data=pd.DataFrame(feature,columns=['feature'])
    x=np.array(data['feature'].tolist())
    x=x.reshape(-1, 40, 1)

    # Make predictions using the machine learning model
    predictions = model.predict(x)
    
    if(predictions[0][0] > predictions[0][1]):
        predicted = "FEMALE"
    else:
        predicted = "MALE"
        
    # Return the predicted values as a string
    return render_template('index.html', message=predicted)

if __name__ == '__main__':
    app.run(debug=True)
