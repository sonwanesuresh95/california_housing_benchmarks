from flask import Flask, render_template, request
from sklearn.utils import shuffle
import pandas as pd

from utils import predict_instance

app = Flask(__name__)

# create 5 test samples
data = pd.read_csv('./data/test.csv')
data = shuffle(data).head(5).to_html()
html_file = open('./templates/test_samples.html', 'w')
html_file.write(data)
html_file.close()


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    example = request.form['example']
    prediction = predict_instance(example)
    message = 'Predicted housing price is {}'.format(prediction)
    return render_template('predict.html',message=message)


if __name__ == '__main__':
    app.run(debug=True)
