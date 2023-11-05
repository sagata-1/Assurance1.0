from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, make_response, session
from flask_session import Session
from flask_wtf import FlaskForm
from wtforms import SubmitField
from numpy import loadtxt
import pandas as pd
from numpy import sort
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image

from lsopt.tree import OptimalTreeClassifier ## M-OCT propsed by Liu & Allen
from lsopt.tree import BinNodePenaltyOptimalTreeClassifier ## BNP-OCT propsed by Liu & Allen
# from lsopt.tree import OldOptimalTreeClassifier ## OCT proposed by Bertsimas & Dunn

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree 

import graphviz

app = Flask(__name__)

# Configure session to use filesystem
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY
Session(app)

training_complete = 0
names = []
fraud_data = pd.DataFrame()

@app.route("/", methods=['GET', "POST"])
def index():
    global training_complete
    training_complete = 0
    if request.method == 'GET':
        return render_template('index.html')
    else:
        return render_template('dataset.html')

@app.route("/dataset", methods=['GET', 'POST'])
def dataset():        
    global training_complete
    global X_train
    global y_train
    global X_test
    global y_test
    global names
    global fraud_data
    global select_X_train
    global selection_model
    print(training_complete)
    print(request.form)
    if request.method == 'GET':
        return render_template('dataset.html')
    elif training_complete == 1:
        training_complete = 2
        return render_template("optfeatures.html")
    elif training_complete == 3:
        training_complete = 4
        return render_template("training.html", names=names)
    else:
        # Find optimal parameters should be flashed here
        # Load data
        if training_complete == 0:
            print(request.files)
            fraud_data = pd.read_csv(request.files["image"])
            print(fraud_data)
            X = fraud_data.iloc[:, 0:47].to_numpy()
            y = fraud_data["PotentialFraud"].apply(lambda val: 0 if val == "Not-Fraud" else 1).to_numpy()
            # Split into train and test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
            # Update user that feature selection has started
            training_complete = 1
            return redirect("/dataset", code=307)
        # Feature select with XGBoost
        if training_complete == 2:
            model = XGBClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions = [round(value) for value in y_pred]    
            accuracy = accuracy_score(y_test, predictions)
            print("Accuracy: %.2f%%" % (accuracy * 100.0))
            thresholds = sort(model.feature_importances_)
            max_acc = -1
            true_thresh = -1
            for thresh in thresholds:
                selection = SelectFromModel(model, threshold=thresh, prefit=True)
                select_X_train = selection.transform(X_train)
                
                selection_model = XGBClassifier()
                selection_model.fit(select_X_train, y_train)
                
                select_X_test = selection.transform(X_test)
                y_pred = selection_model.predict(select_X_test)
                
                predictions = [round(value) for value in y_pred]
                
                accuracy = accuracy_score(y_test, predictions)
                if accuracy >= max_acc and select_X_train.shape[1] < 20:
                    max_acc = accuracy
                    true_thresh = thresh
                print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
            selection = SelectFromModel(model, threshold=true_thresh, prefit=True)
            select_X_train = selection.transform(X_train)

            # Initialize model
            # OCT parameters
            max_depth = 3
            min_samples_leaf = 10
            alpha = 0.00005
            time_limit = 10 # minute
            mip_gap_tol = 0.01  # optimal gap percentage
            mip_focus = 'balance'
            mip_polish_time = None
            warm_start = False
            log_file = None
            fp_heur = True
            backtrack = "bestb"
            
            # Construct BNP-OCT classifier
            selection_model = BinNodePenaltyOptimalTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,
                                            alpha=alpha,
                                            criterion="gini",
                                            solver="gurobi",
                                            time_limit=time_limit,
                                            verbose=True,
                                            warm_start=warm_start,
                                            log_file=log_file,
                                            solver_options={'mip_cuts': 'auto',
                                                            'mip_gap_tol': mip_gap_tol,
                                                            'mip_focus': mip_focus,
                                                            'mip_polish_time': mip_polish_time,
                                                            }
                                            )
            
            # Feature names
            for i in range(len(model.feature_importances_)):
                if model.feature_importances_[i] >= true_thresh:
                    names.append(fraud_data.iloc[:, [i]].columns[0])
            print(len(names))
            training_complete = 3
            return redirect("/dataset", code=307)
        else:
            
            # Train model
            selection_model.fit(select_X_train, y_train)
            # Make prediction
            # selection = SelectFromModel(model, threshold=true_thresh, prefit=True)
            # select_X_test = selection.transform(X_test)
            y_pred = selection_model.predict(X=select_X_train)
            y_pred_prob = selection_model.predict_proba(X=select_X_train)
            
            # Check confusion matrix
            print("Confusion Matrix :")
            print(confusion_matrix(y_true=y_train,
                                y_pred=y_pred))

            print(classification_report(y_true=y_train,
                                        y_pred=y_pred))
            
            # Plot Optimal Tree
            feature_names = names
            class_names = ['Not-Fraud', 'Fraud']

            dot_data = tree.export_graphviz(selection_model,
                                            out_file=None,
                                            feature_names=feature_names,
                                            class_names=class_names,
                                            label='all',
                                            impurity=True,
                                            node_ids=True,
                                            filled=True,
                                            rounded=True,
                                            leaves_parallel=True,
                                            special_characters=False)

            graph = graphviz.Source(dot_data)
            graph.format = 'png'
            graph.render(filename='optimal_tree_fraud', directory='static/img/', view=False)
            return render_template('trueoutput.html')
@app.route('/trueoutput', methods=["GET"])
def true_output():
    return render_template('trueoutput.html')

@app.route("/dataset2", methods=['GET'])
def dataset2():
    return render_template('datasettwo.html')

@app.route("/dataset3", methods=['GET'])
def dataset3():
    return render_template('datasetthree.html')
    
@app.route("/decisiontree", methods=['GET'])
def decision_tree():
    return render_template('decisiontree.html')

@app.route("/decisiontree2", methods=['GET'])
def decision_tree2():
    return render_template('decisiontreetwo.html')

@app.route("/decisiontree3", methods=['GET'])
def decision_tree3():
    return render_template('decisiontreethree.html')

@app.route("/dataset1report", methods=['POST'])
def dataset1report():
    return render_template('datarep.html')

@app.route("/dataset2report", methods=['GET'])
def dataset2report():
    return render_template('datasetreporttwo.html')

@app.route("/dataset3report", methods=['GET'])
def dataset3report():
    return render_template('datasetreportthree.html')

@app.route("/data", methods=['GET'])
def data():
    return render_template('data.html')

@app.route("/dashboard.html", methods=['GET'])
def dashboard():
    return render_template('dashboard.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404