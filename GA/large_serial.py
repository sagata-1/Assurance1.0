import random
import numpy as np
from numpy import sort
import pandas as pd
import multiprocessing as mp
import time
import gc
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
import graphviz

# Assuming that inputData is a pandas DataFrame that contains the required data
inputData = pd.read_csv("../data/grouped(AutoRecovered)_2.csv")
oversampled_fraud = inputData[inputData["PotentialFraud"] == "Fraud"]
for _ in range(3):
    oversampled_fraud = pd.concat([oversampled_fraud, oversampled_fraud], ignore_index=True)
oversampled_fraud = pd.concat([oversampled_fraud, inputData[inputData["PotentialFraud"] == "Fraud"]], ignore_index=True)
inputData = pd.concat([inputData, oversampled_fraud], ignore_index=True)
# inputData = pd.concat([inputData, inputData], ignore_index=True)

# Large value, should be larger than anything objective function can hit
large_value = 1000000000

# Global variables
number_of_features = len(inputData.columns) - 1
print(number_of_features)
number_of_runs = len(inputData.index)
print(number_of_runs)
number_of_classes = 2
number_of_tree_levels = 4
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
# x_vector = np.zeros(chromosome_length)

def main():
    error_value = 0
    global response_categories
    global feature_data
    global number_of_features
    # Read in the data
    dict_conversion = {"Not-Fraud": 1, "Fraud": 2}
    reverse_conversion = {1:"Not-Fraud" , 2: "Fraud"}
    for i in range(number_of_runs):
        response_categories[i] = dict_conversion[inputData.iloc[i, number_of_features]]
        for j in range(number_of_features):
            feature_data[i, j] = inputData.iloc[i, j]
    # Calculate mins and maxes for the features
    for j in range(number_of_features):
        min_feature_value[j] = np.min(feature_data[:, j])
        max_feature_value[j] = np.max(feature_data[:, j])
    
    # Preprocess classes for fitting into 0 - 1 instead of 1 - 2
    # y = np.zeros(number_of_runs)
    # for i in range(number_of_runs):
    #     y[i] = int(response_categories[i] - 1)
    # # Split data for xgboost feature selection
    # X_train, X_test, y_train, y_test = train_test_split(feature_data, y, test_size=0.2, random_state=7)
    # # Feature selection with xgboost
    # model = XGBClassifier()
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # predictions = [round(value) for value in y_pred]    
    # accuracy = accuracy_score(y_test, predictions)
    # print("Accuracy: %.2f%%" % (accuracy * 100.0))  
    
    # thresholds = sort(model.feature_importances_)
    # max_acc = -1
    # true_thresh = -1
    # for thresh in thresholds:
    #     selection = SelectFromModel(model, threshold=thresh, prefit=True)
    #     select_X_train = selection.transform(X_train)
        
    #     selection_model = XGBClassifier()
    #     selection_model.fit(select_X_train, y_train)
        
    #     select_X_test = selection.transform(X_test)
    #     y_pred = selection_model.predict(select_X_test)
        
    #     predictions = [round(value) for value in y_pred]
        
    #     accuracy = accuracy_score(y_test, predictions)
    #     if accuracy >= max_acc:
    #         max_acc = accuracy
    #         true_thresh = thresh
    #     print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
    # selection = SelectFromModel(model, threshold=true_thresh, prefit=True)
    # feature_data = selection.transform(feature_data)
    # # selection_model = XGBClassifier()
    # # selection_model.fit(feature_data, y)
    # # y_pred = selection_model.predict(feature_data)
    # # predictions = [round(value) for value in y_pred]
    # # inputData["Predictions"] = predictions
    # # result = inputData[["PotentialFraud", "Predictions"]]
    # # result.to_csv("predictions.csv", sep=',', index=False, encoding='utf-8')
    # names = []
    # for i in range(len(model.feature_importances_)):
    #     if model.feature_importances_[i] >= true_thresh:
    #         names.append(inputData.iloc[:, [i]].columns[0])
    # print(len(names))
    # number_of_features = len(names)

    # # Define chromosomeVec and individualPoint with example values
    # chromosome_vec = np.array([0.7, 0.338, 0.9, 0.7, 0.8, 0.7, 0.1, 0.1, 0.5, 0.8])  # Example values
    # individual_point = np.array([5.1, 3.5, 1.4, 0.2])  # Example values

    # Function calls
    # class_tree_translate_to_engineering(chromosome_length, chromosome_vec, person_tree)
    # print(tree_model_predict(chromosome_length, individual_point, person_tree, 10))
    # pool = mp.Pool(1)
    # error_value = estimate_prediction_error(pool, chromosome_length, person_tree, 0)
    # pool.close()
    # pool.join()
    # print(person_tree)
    # print(error_value)
    start = time.time()
    eng_vec = np.zeros(chromosome_length)
    val = deterministic_ga(chromosome_length, 20, 20, eng_vec)
    print(eng_vec)
    print(f"{val[0]:.2f}")
    end = time.time()
    print(f"Running time: {end - start}")
    dot_data = array_to_dot(eng_vec, inputData)
    # print(dot_data)
    with open("tree.dot", "w") as f:
        f.write(dot_data)
    # s = graphviz.Source(dot_data, format="png")
    # s.render('tree')
    graph = graphviz.Source(dot_data)
    graph.format = 'png'
    graph.render(filename='tree', directory='', view=True)
    # Make predictions and append them to the pandas dataframe and store it in a new file
    predictions = []
    for i in range(number_of_runs):
        predictions.append(reverse_conversion[int(tree_model_predict(chromosome_length,feature_data[i, :], eng_vec, prediction_category))])
    # print(predictions)
    inputData["Predictions"] = predictions
    result = inputData[["PotentialFraud", "Predictions"]]
    accuracy = accuracy_score(inputData["PotentialFraud"], inputData["Predictions"])
    print("Final Accuracy:", accuracy)
    print("Not-Fraud and Fraud incorrect proportions", val[1])
    result.to_csv("predictions.csv", sep=',', index=False, encoding='utf-8')

# def parallel_estimate_prediction_error(chromosome_length, person_tree, error_value, individual_point, response_category):
#     prediction_categor = tree_model_predict(chromosome_length, individual_point, person_tree, prediction_category)
#     # print(prediction_categor, response_category)
#     if prediction_categor != response_category:
#         return 1
#     return 0

# def estimate_prediction_error(pool, chromosome_length, person_tree, error_value):
#     errors = pool.starmap(parallel_estimate_prediction_error, [(chromosome_length, person_tree, error_value, feature_data[i, :], response_categories[i]) for i in range(len(response_categories))])
#     return (sum(errors) / len(response_categories) + 0.01 *(person_tree[3]), 0.01 * person_tree[3])

def array_to_dot(array, data):
    # Node features and values
    if number_of_tree_levels == 3:
        temp_arr = array[:-8]
    else:
        temp_arr = array[:-16]
    # Leaf nodes- Fraud / Not-Fraud
    if number_of_tree_levels == 3:
        classes = list(map(int, array[14:]))
    else:
        classes = list(map(int, array[30:]))
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
    if number_of_tree_levels == 3:
        for i in range(7, len(classes) + 7):
            if classes[i - 7] == 1:
                dot_str += f'    {i} [label="Not-Fraud"]\n'
            else:
                dot_str += f'    {i} [label="Fraud", fillcolor="#74baed"]\n'
    else:
        for i in range(15, len(classes) + 15):
            if classes[i - 15] == 1:
                dot_str += f'    {i} [label="Not-Fraud"]\n'
            else:
                dot_str += f'    {i} [label="Fraud", fillcolor="#74baed"]\n'
    # dot_str += '{rank=same ; 7; 8; 9; 10; 11; 12; 13; 14}\n'
    dot_str += '}'
    return dot_str

def estimate_prediction_error(pool, chromosome_length, person_tree, error_value):
    number_of_runs = len(response_categories)
    number_of_features = feature_data.shape[1]
    insensitivity = 0
    imprecision = 0
    not_fraud = 0
    fraud = 0
    inacc = 0

    for i in range(number_of_runs):
        individual_point = feature_data[i, :]

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
        if int(prediction_categor) != int(response_categories[i]):
            inacc += 1

    # print(error_value)
    # print(number_of_runs)
    
    # error_value /= (number_of_runs)
    # error_value = (((imprecision / not_fraud) + (insensitivity / fraud)) / 2)
    error_value = 0.3* (((imprecision / not_fraud) + (insensitivity / fraud)) / 2)+ (inacc / number_of_runs)
    # error_value = 1 - ((not_fraud - imprecision) / ((not_fraud - imprecision) + 0.5 *(insensitivity + imprecision))) # f1-score
    # error_value = inacc / number_of_runs
    # error_value = 0.4*(insensitivity / fraud) + (inacc / number_of_runs)
    # error_value =  0.7*(insensitivity / fraud) + (inacc / number_of_runs)
    # print("Test", (imprecision / not_fraud), (insensitivity / fraud))
        
    return (error_value, 0.3* (((imprecision / not_fraud) + (insensitivity / fraud)) / 2), (imprecision / not_fraud), (insensitivity / fraud))

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
    if number_of_tree_levels == 4:
        # Node 1
        if individual_point[int(person_tree[0]) - 1] < person_tree[1]:
            # Node 2
            if individual_point[int(person_tree[2]) - 1] < person_tree[3]:
                # Node 4
                if individual_point[int(person_tree[6]) - 1] < person_tree[7]:
                    # Node 8
                    if individual_point[int(person_tree[14]) - 1] < person_tree[15]:
                        prediction_categor = person_tree[30]
                    else:
                        prediction_categor = person_tree[31]
                    # Node 9
                else:
                    if individual_point[int(person_tree[16]) - 1] < person_tree[17]:
                        prediction_categor = person_tree[32]
                    else:
                        prediction_categor = person_tree[33]
            # Node 5
            else:
                if individual_point[int(person_tree[8]) - 1] < person_tree[9]:
                    # Node 10
                    if individual_point[int(person_tree[18]) - 1] < person_tree[19]:
                        prediction_categor = person_tree[34]
                    else:
                        prediction_categor = person_tree[35]
                    # Node 11
                else:
                    if individual_point[int(person_tree[20]) - 1] < person_tree[21]:
                        prediction_categor = person_tree[36]
                    else:
                        prediction_categor = person_tree[37]
        # Node 3
        else:
            if individual_point[int(person_tree[4]) - 1] < person_tree[5]:
                # Node 6
                if individual_point[int(person_tree[10]) - 1] < person_tree[11]:
                    # Node 12
                    if individual_point[int(person_tree[22]) - 1] < person_tree[23]:
                        prediction_categor = person_tree[38]
                    else:
                        prediction_categor = person_tree[39]
                    # Node 13
                else:
                    if individual_point[int(person_tree[24]) - 1] < person_tree[25]:
                        prediction_categor = person_tree[40]
                    else:
                        prediction_categor = person_tree[41]
                # Node 7
            else:
                if individual_point[int(person_tree[12]) - 1] < person_tree[13]:
                    # Node 14
                    if individual_point[int(person_tree[26]) - 1] < person_tree[27]:
                        prediction_categor = person_tree[42]
                    else:
                        prediction_categor = person_tree[43]
                    # Node 15
                else:
                    if individual_point[int(person_tree[28]) - 1] < person_tree[29]:
                        prediction_categor = person_tree[44]
                    else:
                        prediction_categor = person_tree[45]


    return prediction_categor

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

def a4_function(pool, number_decision_variables, x_vector, class_tree_translate_to_engineering, estimate_prediction_error, engineering_x_vector):
    # Part 1: Interpret the [0,1] hypercube vector as a solution.
    class_tree_translate_to_engineering(number_decision_variables, x_vector, engineering_x_vector)

    # Part 2: Evaluate the solution.- fitness valuation
    error_value = estimate_prediction_error(pool, number_decision_variables, engineering_x_vector, 0)

    return error_value

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


def deterministic_ga(number_decision_variables, number_in_population, number_of_generations, engineering_x_vector):
    # Legacy
    pool = 0
    # Define the scalar variables
    e_elitist = int(0.1 * number_in_population)
    m_immigrant = int(0.1 * number_in_population)
    probability_bernoulli = 0.8
    big_number = 1000000000  # This should be bigger than any relevant objective value.

    # Define the vectors and matrices
    current_objective_values = np.zeros(number_in_population)
    normalizer = np.zeros(number_in_population)
    proportions = np.zeros((number_in_population, 2))
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
            proportions[i_index] = (temp[2], temp[3])
        # Sort the population
        sort_index = np.argsort(current_objective_values)
        current_generation = current_generation[sort_index, :]
        current_objective_values = current_objective_values[sort_index]
        normalizer = normalizer[sort_index]
        proportions = proportions[sort_index]

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
                    first_child[j_index] = (current_generation[first_parent_index, j_index] + current_generation[second_parent_index, j_index]) / 2
                    second_child[j_index] = (current_generation[first_parent_index, j_index] + current_generation[second_parent_index, j_index]) / 2
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

        # Copy over the current generation
        current_generation = next_generation.copy()

    # Evaluate the last generation
    for i_index in range(number_in_population):
        x_vector = current_generation[i_index, :]
        temp = a4_function(pool, number_decision_variables, x_vector, class_tree_translate_to_engineering, estimate_prediction_error, engineering_x_vector)
        current_objective_values[i_index] = temp[0]
        normalizer[i_index] = temp[1]
        proportions[i_index] = (temp[2], temp[3])

    # Sort the population
    sort_index = np.argsort(current_objective_values)
    current_generation = current_generation[sort_index, :]
    current_objective_values = current_objective_values[sort_index]
    normalizer = normalizer[sort_index]
    proportions = proportions[sort_index]

    x_vector = current_generation[0, :]
    a4_translate_to_engineering(number_decision_variables, x_vector, class_tree_translate_to_engineering, engineering_x_vector)
    return (current_objective_values[0] - normalizer[0], proportions[0])


if __name__ == "__main__":
    main()