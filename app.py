from flask import Flask, request, render_template, redirect
from flask_bootstrap import Bootstrap
import os
import pandas as pd
from werkzeug.utils import secure_filename
from model import tpotClassifier, tpotRegressor, autoviml
import joblib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/home/noobie/Bvicam-project'
app.config['Download_Folder'] = '/home/noobie/Bvicam-project/static'
app.config['SECRET_KEY'] = 'LOL'
Bootstrap(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/withcode', methods=['GET', 'POST'])
def withcode():
    if request.method == 'POST':
        target = request.form['targetValue']
        inputfile = request.files['inputfile']
        type = request.form['Type']
        orig_name = inputfile.filename
        filename = secure_filename(orig_name)
        fullpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        inputfile.save(fullpath)
        df = pd.read_csv(fullpath)
        if type == "Classifier":
            model, score = tpotClassifier(df, target)
            joblib.dump(model, 'Yourmodel.pkl')
        else:
            model, score = tpotRegressor(df, target)
            joblib.dump(model, 'Yourmodel.pkl')
        return redirect('/')
    return render_template('with-code.html')


@app.route('/withoutcode', methods=['GET', 'POST'])
def withoutcode():
    if request.method == 'POST':
        target = request.form['targetValue']
        inputfile = request.files['inputfile']
        orig_name = inputfile.filename
        filename = secure_filename(orig_name)
        fullpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        inputfile.save(fullpath)
        df = pd.read_csv(fullpath)
        model,score = autoviml(df, target)
        joblib.dump(model, 'Yourmodel.pkl')
        return redirect('/')
    return render_template('without-code.html')


@app.route('/pklusage')
def pkl_usage():
    return render_template('pkl_usage.html')


@app.route('/advanced', methods=['GET', 'POST'])
def advanced():
    if request.method == 'POST':
        target = request.form['targetValue']
        inputfile = request.files['inputfile']
        type = request.form['Type']
        orig_name = inputfile.filename
        filename = secure_filename(orig_name)
        fullpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        inputfile.save(fullpath)
        df = pd.read_csv(fullpath)
        if type == "Classifier":
            model, score = tpotClassifier(df, target)
            joblib.dump(model, 'Yourmodel.pkl')
        else:
            model, score = tpotRegressor(df, target)
            joblib.dump(model, 'Yourmodel.pkl')
        return redirect('/')
    return render_template('Advanced.html')

@app.route('/download')
def download():
    pass

if __name__ == "__main__":
    app.run(debug=True, port=3000)
