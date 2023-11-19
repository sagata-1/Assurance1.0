import random
import numpy as np
import pandas as pd
from basicGA import deterministic_ga
import multiprocessing as mp
import time
import gc

# Assuming that inputData is a pandas DataFrame that contains the required data
inputData = pd.read_csv("../data/grouped(AutoRecovered)_2.csv")

# Large value, should be larger than anything objective function can hit
large_value = 1000000000

# Global variables
number_of_features = 48
number_of_runs = 5000
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
# x_vector = np.zeros(chromosome_length)

def main():
    error_value = 0
    # Read in the data
    dict_conversion = {"Not-Fraud": 1, "Fraud": 2}
    for i in range(number_of_runs):
        response_categories[i] = dict_conversion[inputData.iloc[i, 48]]
        for j in range(number_of_features):
            feature_data[i, j] = inputData.iloc[i, j]
    # Calculate mins and maxes for the features
    for j in range(number_of_features):
        min_feature_value[j] = np.min(feature_data[:, j])
        max_feature_value[j] = np.max(feature_data[:, j])

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
    val = deterministic_ga(chromosome_length, 50, 50, eng_vec)
    print(eng_vec)
    print(f"{val:.2f}")
    end = time.time()
    print(f"Running time: {end - start}")

# def parallel_estimate_prediction_error(chromosome_length, person_tree, error_value, individual_point, response_category):
#     prediction_categor = tree_model_predict(chromosome_length, individual_point, person_tree, prediction_category)
#     # print(prediction_categor, response_category)
#     if prediction_categor != response_category:
#         return 1
#     return 0

# def estimate_prediction_error(pool, chromosome_length, person_tree, error_value):
#     errors = pool.starmap(parallel_estimate_prediction_error, [(chromosome_length, person_tree, error_value, feature_data[i, :], response_categories[i]) for i in range(len(response_categories))])
#     return (sum(errors) / len(response_categories) + 0.01 *(person_tree[3]), 0.01 * person_tree[3])

def estimate_prediction_error(pool, chromosome_length, person_tree, error_value):
    number_of_runs = len(response_categories)
    number_of_features = feature_data.shape[1]

    for i in range(number_of_runs):
        individual_point = feature_data[i, :]

        prediction_categor = tree_model_predict(chromosome_length, individual_point, person_tree, prediction_category)
        # print(prediction_categor, response_categories[i], individual_point)

        if int(prediction_categor) != int(response_categories[i]):
            error_value += 1

    # TODO, consider node complexity
    # print(error_value)
    # print(number_of_runs)
    
    error_value /= (number_of_runs)
    return (error_value, 0)

def tree_model_predict(chromosome_length, individual_point, person_tree, prediction_category):
    if number_of_tree_levels == 2:
        # Top node
        if individual_point[int(person_tree[0]) - 1] < person_tree[1]:
            # Second node
            if individual_point[int(person_tree[2]) - 1] < person_tree[3]:
                prediction_category = person_tree[6]
            else:
                prediction_category = person_tree[7]
        else:
            # Third node
            if individual_point[int(person_tree[4]) - 1] < person_tree[5]:
                prediction_category = person_tree[8]
            else:
                prediction_category = person_tree[9]

    elif number_of_tree_levels == 3:
        # Node 1
        if individual_point[int(person_tree[0]) - 1] < person_tree[1]:
            # Node 2
            if individual_point[int(person_tree[2]) - 1] < person_tree[3]:
                # Node 4
                if individual_point[int(person_tree[6]) - 1] < person_tree[7]:
                    prediction_category = person_tree[14]
                else:
                    prediction_category = person_tree[15]
            # Node 5
            else:
                if individual_point[int(person_tree[8]) - 1] < person_tree[9]:
                    prediction_category = person_tree[16]
                else:
                    prediction_category = person_tree[17]
        # Node 3
        else:
            if individual_point[int(person_tree[4]) - 1] < person_tree[5]:
                # Node 6
                if individual_point[int(person_tree[10]) - 1] < person_tree[11]:
                    prediction_category = person_tree[18]
                else:
                    prediction_category = person_tree[19]
            # Node 7
            else:
                if individual_point[int(person_tree[12]) - 1] < person_tree[13]:
                    prediction_category = person_tree[20]
                else:
                    prediction_category = person_tree[21]

    return prediction_category

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
    # Assuming a4Function and a4TranslateToEngineering are defined elsewhere
    # number_of_tree_levels = 2
    # number_of_nodes = 1 + sum(2 ** i for i in range(1, number_of_tree_levels))
    # number_of_leaves = 2 ** number_of_tree_levels
    # number_decision_variables = number_decision_variables
    pool = mp.Pool(mp.cpu_count())
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
    pool.close()
    pool.join()
    return current_objective_values[0] - normalizer[0]
    # if current_objective_values[0] < 0 and current_objective_values[0] >= -1:
    #     val =  -1 * current_objective_values[0]
    # elif current_objective_values[0] > 1:
    #     val =  (current_objective_values[0] / current_objective_values[0])
    # elif current_objective_values[0] < -1:
    #     val = (current_objective_values[0] / current_objective_values[0])
    # else:
    #     val = current_objective_values[0]
    # if val > max_val:
    #     max_val = val


if __name__ == "__main__":
    main()