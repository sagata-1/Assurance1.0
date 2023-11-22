from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, make_response, session
from flask_session import Session
from flaskwebgui import FlaskUI
import numpy as np
from numpy import loadtxt
import pandas as pd
from numpy import sort
import os
import time
import random
import graphviz

app = Flask(__name__)

# # Configure session to use filesystem
# app.config["SESSION_PERMANENT"] = False
# app.config["SESSION_TYPE"] = "filesystem"
# SECRET_KEY = os.urandom(32)
# app.config['SECRET_KEY'] = SECRET_KEY
# Session(app)

ui = FlaskUI(app=app, server="flask")

# Variable to keep track of page changes
training_complete = 0
# Legacy varialble- to keep track of features selected for (might be used again)
names = []
inputData = pd.DataFrame()
# global selection_model
large_value = 1000000000

# Global variables
number_of_features = 48
number_of_runs = 5410
number_of_classes = 2
number_of_tree_levels = 3
number_of_nodes = 1 + sum(2 ** i for i in range(1, number_of_tree_levels))
number_of_leaves = 2 ** number_of_tree_levels
chromosome_length = 2 * number_of_nodes + number_of_leaves
# Placeholder for personTree array
person_tree = np.zeros(chromosome_length)

# Initialize arrays and variables
feature_data = np.zeros((number_of_runs, number_of_features))
min_feature_value = np.zeros(number_of_features)
max_feature_value = np.zeros(number_of_features)
response_categories = np.zeros(number_of_runs)
prediction_category = 0

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
    global inputData
    global large_value

    # Global variables
    global number_of_features
    global number_of_runs
    global number_of_classes
    global number_of_tree_levels
    global number_of_nodes
    global number_of_leaves
    global chromosome_length
    # Placeholder for personTree array
    global person_tree

    # Initialize arrays and variables
    global feature_data
    global min_feature_value
    global max_feature_value
    global response_categories
    global prediction_category

    # Debug statements- checking that we're transitioning between pages
    print(training_complete)
    print(request.form)
    
    # Page that renders where you can upload the data
    if request.method == 'GET':
        return render_template('dataset.html')
    # Page that indicates training is occurring
    elif training_complete == 2:
        training_complete = 3
        return render_template("training.html", names=names)
    else:
        # Load data in, then indicate that training has started
        if training_complete == 0:
            print(request.files)
            inputData = pd.read_csv(request.files["image"])
            print(inputData)
            dict_conversion = {"Not-Fraud": 1, "Fraud": 2}
            for i in range(number_of_runs):
                response_categories[i] = dict_conversion[inputData.iloc[i, 48]]
                for j in range(number_of_features):
                    feature_data[i, j] = inputData.iloc[i, j]
            # Calculate mins and maxes for the features
            for j in range(number_of_features):
                min_feature_value[j] = np.min(feature_data[:, j])
                max_feature_value[j] = np.max(feature_data[:, j])
            training_complete = 2
            return redirect("/dataset", code=307)
        else:
            # Train model
            start = time.time()
            eng_vec = np.zeros(chromosome_length)
            val = deterministic_ga(chromosome_length, 30, 30, eng_vec)
            print(eng_vec)
            print(f"{val:.2f}")
            end = time.time()
            print(f"Running time: {end - start}")
            dot_data = array_to_dot(eng_vec, inputData)
            # graph = graphviz.Source(dot_data)
            # graph.format = 'png'
            # graph.render(filename='tree', directory='static/img/', view=False)
            # Backend check that everything is working fine
            print(eng_vec)
            # Page that shows trained tree
            return render_template('trueoutput.html')

# Page that shows trained tree (available by get, so currently just for checking)
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

# 404 return page
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

# Function  to convert tree in array form into a form that can be read by graphviz to get a graph
def array_to_dot(array, data):
    # Node features and values
    temp_arr = array[:-8]
    # Leaf nodes- Fraud / Not-Fraud
    classes = list(map(int, array[14:]))
    count = 0
    dot_str = 'digraph G {\n'
    dot_str += 'node [shape="box", style="filled, rounded", color="orange", fontname="helvetica"];\ngraph [ranksep=equally, splines=polyline];\nedge [fontname="helvetica"];\n'
    for index, value in enumerate(temp_arr):
        # Create a node for each element in the array
        if index % 2 == 0:
            feature = int(value)
            dot_str += f'    {count} [label="Node #{count}\n Feature {list(data.columns.values)[feature]} '
        else:
            dot_str += f'<= {value}"];\n'
        
            # Assuming a binary tree stored in a typical array format (as that's what the grad student is using)
            left_child_index = 2 * count + 1
            right_child_index = 2 * count + 2
            
            # Add edges if children exist
            if left_child_index <= len(temp_arr):
                dot_str += f'    {count} -> {left_child_index};\n'
            if right_child_index <= len(temp_arr):
                dot_str += f'    {count} -> {right_child_index};\n'
            count += 1
    for i in range(7, len(classes) + 7):
        if classes[i - 7] == 1:
            dot_str += f'    {i} [label="Not-Fraud"]\n'
        else:
            dot_str += f'    {i} [label="Fraud", fillcolor="#74baed"]\n'
    # dot_str += '{rank=same ; 7; 8; 9; 10; 11; 12; 13; 14}\n'
    dot_str += '}'
    return dot_str

# Name is self-explanatory, estimate the error of each tree in the GA
def estimate_prediction_error(pool, chromosome_length, person_tree, error_value):
    number_of_runs = len(response_categories)
    number_of_features = feature_data.shape[1]
    insensitivity = 0
    imprecision = 0
    not_fraud = 1
    fraud = 1

    for i in range(number_of_runs):
        individual_point = feature_data[i, :]
        # Make prediction with current tree
        prediction_categor = tree_model_predict(chromosome_length, individual_point, person_tree, prediction_category)
        # print(prediction_categor, response_categories[i], individual_point)
        if int(response_categories[i]) == 1:
            if int(prediction_categor) != int(response_categories[i]) and int(response_categories[i]) == 1:
                imprecision += 1
            not_fraud += 1
        elif int(response_categories[i]) == 2:
            if int(prediction_categor) != int(response_categories[i]):
                insensitivity += 1
            fraud += 1
        # Legacy code, might be used again
        # if int(prediction_categor) != int(response_categories[i]):
        #     error_value += 1

    # TODO, consider node complexity
    # print(error_value)
    # print(number_of_runs)
    
    # error_value /= (number_of_runs)
    # Compute error value with respect to sensitivity and precision (proprotion of not-fraud and fraud-cases identified, as a weighted average)
    error_value = (((imprecision / not_fraud) + (insensitivity / fraud)) / 2)
    print("Test", (imprecision / not_fraud), (insensitivity / fraud))
        
    return (error_value, 0)

# Make prediction with current tree
def tree_model_predict(chromosome_length, individual_point, person_tree, prediction_category):
    prediction_categor = 0
    if number_of_tree_levels == 2:
        # Top node
        if individual_point[int(person_tree[0]) - 1] < person_tree[1]:
            # Second node
            if individual_point[int(person_tree[2]) - 1] < person_tree[3]:
                prediction_categor = person_tree[6]
            else:
                prediction_categor = person_tree[7]
        else:
            # Third node
            if individual_point[int(person_tree[4]) - 1] < person_tree[5]:
                prediction_categor = person_tree[8]
            else:
                prediction_categor = person_tree[9]

    elif number_of_tree_levels == 3:
        # Node 1
        if individual_point[int(person_tree[0]) - 1] < person_tree[1]:
            # Node 2
            if individual_point[int(person_tree[2]) - 1] < person_tree[3]:
                # Node 4
                if individual_point[int(person_tree[6]) - 1] < person_tree[7]:
                    prediction_categor = person_tree[14]
                else:
                    prediction_categor = person_tree[15]
            # Node 5
            else:
                if individual_point[int(person_tree[8]) - 1] < person_tree[9]:
                    prediction_categor = person_tree[16]
                else:
                    prediction_categor = person_tree[17]
        # Node 3
        else:
            if individual_point[int(person_tree[4]) - 1] < person_tree[5]:
                # Node 6
                if individual_point[int(person_tree[10]) - 1] < person_tree[11]:
                    prediction_categor = person_tree[18]
                else:
                    prediction_categor = person_tree[19]
            # Node 7
            else:
                if individual_point[int(person_tree[12]) - 1] < person_tree[13]:
                    prediction_categor = person_tree[20]
                else:
                    prediction_categor = person_tree[21]

    return prediction_categor

# Convert the zero vector array into a tree
def class_tree_translate_to_engineering(number_decision_variables, x_vector, engineering_x_vector):
    for i in range(number_of_nodes):
        # Odd values in vector are splitting variables. Even values are splitting values.
        # Splitting variables for single variable splits
        engineering_x_vector[(2 * i)] =  int(1 + int(x_vector[(2 * i)] * number_of_features))
        # Splitting values for single variable splits
        feature_index = int(engineering_x_vector[(2 * i)] - 1)  # Adjusting for 0-based index
        # print(feature_index)
        engineering_x_vector[(2 * i) + 1] = x_vector[2 * (i) + 1] * (max_feature_value[feature_index] - min_feature_value[feature_index]) + min_feature_value[feature_index]
            

    # Decide which class for each leaf
    for i in range(number_of_leaves):
        engineering_x_vector[2 * number_of_nodes + i] = int(x_vector[(2 * (number_of_nodes)) + i] * number_of_classes) + 1

    return engineering_x_vector

def class_tree_function(number_decision_variables, x_vector, a4_translate_to_engineering):
    # Part 1: Interpret the [0,1] hypercube vector as a solution.
    engineering_x_vector = a4_translate_to_engineering(number_decision_variables, x_vector)

    # Part 2: Evaluate the solution.
    a4_function = sum(i * (engineering_x_vector[i - 1] ** 4) for i in range(1, number_decision_variables + 1))

    return a4_function

# Fitness value estimation- constructs tree, then evaluates it (currently the main candidate for parallelization)
def a4_function(pool, number_decision_variables, x_vector, class_tree_translate_to_engineering, estimate_prediction_error, engineering_x_vector):
    # Part 1: Interpret the [0,1] hypercube vector as a solution.
    class_tree_translate_to_engineering(number_decision_variables, x_vector, engineering_x_vector)

    # Part 2: Evaluate the solution.- fitness valuation
    error_value = estimate_prediction_error(pool, number_decision_variables, engineering_x_vector, 0)

    return error_value

# Translate final tree into array form
def a4_translate_to_engineering(number_decision_variables, x_vector, class_tree_translate_to_engineering, engineering_x_vector):
    class_tree_translate_to_engineering(number_decision_variables, x_vector, engineering_x_vector)
    # return

    # If we need to include the alternative calculation, then I'll uncomment and use the following lines:
    # for i in range(number_decision_variables):
    #     engineering_x_vector[i] = (x_vector[i] - 0.5) * 2.56
    return engineering_x_vector

def a4_translate_from_engineering(number_decision_variables, engineering_x_vector):
    x_vector = [0] * number_decision_variables

    for i in range(number_decision_variables):
        x_vector[i] = ((engineering_x_vector[i] + 1.28) / 2.56)

    return x_vector

# The GA algorithm
def deterministic_ga(number_decision_variables, number_in_population, number_of_generations, engineering_x_vector):
    # Legacy, possibly used when parallelizing
    pool = 0
    # Define the scalar variables
    e_elitist = int(0.1 * number_in_population)
    m_immigrant = int(0.1 * number_in_population)
    probability_bernoulli = 0.8
    big_number = 1000000000  # This should be bigger than any relevant objective value.

    # Define the vectors and matrices
    current_objective_values = np.zeros(number_in_population)
    normalizer = np.zeros(number_in_population)
    next_objective_values = np.zeros(number_in_population)
    # Initialise first generation
    temp = ()
    current_generation = np.random.rand(number_in_population, number_decision_variables)
    next_generation = np.zeros((number_in_population, number_decision_variables))
    x_vector = np.zeros(number_decision_variables)
    first_child = np.zeros(number_decision_variables)
    second_child = np.zeros(number_decision_variables)

    for g_index in range(number_of_generations):
        # Evaluate the current generation (fitness score)
        for i_index in range(number_in_population):
            x_vector = current_generation[i_index, :]
            temp = a4_function(pool, number_decision_variables, x_vector, class_tree_translate_to_engineering, estimate_prediction_error, engineering_x_vector)
            print(temp)
            current_objective_values[i_index] = temp[0]
            normalizer[i_index] = temp[1]
        # Sort the population
        sort_index = np.argsort(current_objective_values)
        current_generation = current_generation[sort_index, :]
        current_objective_values = current_objective_values[sort_index]
        normalizer = normalizer[sort_index]

        # Make elitist subset keeping top e_elitist solutions
        next_generation[:e_elitist, :] = current_generation[:e_elitist, :]

        # Make m_immigrant immigrant subset or massive mutants
        next_generation[e_elitist:e_elitist + m_immigrant, :] = np.random.rand(m_immigrant, number_decision_variables)

        # Fill remainder with crossover solutions
        for i_index in range(e_elitist + m_immigrant, number_in_population):
            first_parent_index = random.randint(0, number_in_population - 1)
            second_parent_index = random.randint(0, number_in_population - 1)

            # Perform Bernoulli crossover to make children
            for j_index in range(number_decision_variables):
                if random.random() < probability_bernoulli:
                    first_child[j_index] = current_generation[first_parent_index, j_index]
                    second_child[j_index] = current_generation[second_parent_index, j_index]
                else:
                    second_child[j_index] = current_generation[first_parent_index, j_index]
                    first_child[j_index] = current_generation[second_parent_index, j_index]
            # Tournament select the best child for the next generation
            temp = a4_function(pool, number_decision_variables, first_child, class_tree_translate_to_engineering, estimate_prediction_error, engineering_x_vector)
            temp_1 = a4_function(pool, number_decision_variables, second_child, class_tree_translate_to_engineering, estimate_prediction_error, engineering_x_vector)
            first_child_value = temp[0]
            second_child_value = temp_1[0]

            if first_child_value < second_child_value:
                next_generation[i_index, :] = first_child
            else:
                next_generation[i_index, :] = second_child
                
        # Possible set up for depth variability, currently not in use
        # if (g_index == 25):
        #     number_of_nodes = 1 + sum(2 ** i for i in range(1, number_of_tree_levels))
        #     number_of_leaves = 2 ** number_of_tree_levels
        #     number_decision_variables = 2 * number_of_nodes + number_of_leaves
        
        # Copy over the current generation
        current_generation = next_generation.copy()

    # Evaluate the last generation
    for i_index in range(number_in_population):
        x_vector = current_generation[i_index, :]
        temp = a4_function(pool, number_decision_variables, x_vector, class_tree_translate_to_engineering, estimate_prediction_error, engineering_x_vector)
        current_objective_values[i_index] = temp[0]
        normalizer[i_index] = temp[1]

    # Sort the population
    sort_index = np.argsort(current_objective_values)
    current_generation = current_generation[sort_index, :]
    current_objective_values = current_objective_values[sort_index]
    normalizer = normalizer[sort_index]

    x_vector = current_generation[0, :]
    a4_translate_to_engineering(number_decision_variables, x_vector, class_tree_translate_to_engineering, engineering_x_vector)
    # pool.close()
    # pool.join()
    return current_objective_values[0] - normalizer[0]

if __name__ == "__main__":
    ui.run()