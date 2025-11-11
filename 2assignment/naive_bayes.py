import numpy as np


def naive_bayes(training_file, test_file):
    # Training Data
    training_data = {}
    count = 0
    with open(training_file, 'r') as file:
        
        for line in file:
            # Parsing data
            clean_line = line.strip()
            data = clean_line.split()

            # Turning strings into integers/floats depending on attribute/class label
            for i in range(len(data) - 1):
                data[i] = float(data[i]) # attribute
            data[len(data) - 1] = int(data[len(data) - 1]) # class label

            # If does not have 9 columns, then return
            if (len(data) != 9):
                print("There is training data exceeding the normal number of columns and cannot be processed.")
                return
            else:
                class_label = data[8]
                if class_label not in training_data:
                    training_data[class_label] = {}
                    training_data[class_label]["appearances"] = 1
                    for attribute in range(1, 9):
                        if attribute not in training_data[class_label]:
                            training_data[class_label][attribute] = {}
                            training_data[class_label][attribute]["data"] = []
                            training_data[class_label][attribute]["data"].append(data[attribute-1])
                else:
                    training_data[class_label]["appearances"] += 1
                    for attribute in range(1, 9):
                        training_data[class_label][attribute]["data"].append(data[attribute-1])
                count += 1
    for class_label in training_data:
        training_data[class_label]["probability"] = training_data[class_label]["appearances"] / count
                        

    # Test Data
    test_data = {}
    with open(test_file, 'r') as file:
        
        id = 1
        for line in file:
            clean_line = line.strip()
            data = clean_line.split()

            # Turning strings into integers/floats depending on attribute/class label
            for i in range(len(data) - 1):
                data[i] = float(data[i]) # attribute
            data[len(data) - 1] = int(data[len(data) - 1]) # class label

            if (len(data) != 9):
                print("There is training data exceeding the normal number of columns and cannot be processed.")
                return
            else:
                class_label = data[8]
                if id not in test_data:
                    test_data[id] = {}
                test_data[id]["data"] = data
                id += 1
    calculate_gaussian(training_data, test_data)
            
def calculate_means(training_data):
    for class_label in training_data:
        for attribute in range(1, 9):
            sum = 0
            sample_size = 0
            for data in training_data[class_label][attribute]["data"]:
                sample_size += 1
                sum += data
            mean = sum / sample_size
            training_data[class_label][attribute]["mean"] = mean       
            

def calculate_stdevs(training_data):
    for class_label in range(1, len(training_data) + 1):
        for attribute in range(1, 9):
            squared_deviation_sum = 0
            sample_size = -1
            for data in training_data[class_label][attribute]["data"]:
                sample_size += 1
                squared_deviation_sum += (data - training_data[class_label][attribute]["mean"]) ** 2
            squared_deviation_mean = squared_deviation_sum / sample_size
            stdev = squared_deviation_mean ** 0.5
            if stdev > 0.01:
                training_data[class_label][attribute]["stdev"] = stdev
            else:
                training_data[class_label][attribute]["stdev"] = 0.01
            print(f"Class {class_label}, attribute {attribute}, mean = {training_data[class_label][attribute]["mean"] : .2f}, std = {training_data[class_label][attribute]["stdev"] : .2f}")

def calculate_gaussian(training_data, test_data):
    calculate_means(training_data)
    calculate_stdevs(training_data)
    accuracy_sum = 0
    for id in test_data:
        class_probabilities = {}
        for class_label in range(1, 11):
            total_probability = np.log(training_data[class_label]["probability"])
            for attribute in range(1, 9):
                mean = training_data[class_label][attribute]["mean"]
                stdev = training_data[class_label][attribute]["stdev"]
                data_pt = test_data[id]["data"][attribute-1]
                normalization_constant = 1 / np.sqrt(2 * np.pi * (stdev ** 2))
                exponential_term =  np.exp(- (((data_pt - mean) ** 2) / (2 * (stdev ** 2))))
                gaussian_likelihood = normalization_constant * exponential_term
                if gaussian_likelihood == 0:
                    gaussian_likelihood = 1E-9
                log_gaussian_likelihood = np.log(gaussian_likelihood)
                total_probability += log_gaussian_likelihood 
            class_probabilities[class_label] = np.exp(total_probability)

        evidence = sum(class_probabilities.values())
        best_likelihood_class = max(class_probabilities, key=class_probabilities.get)
        best_likelihood = 0
        if evidence > 0:
            best_likelihood = class_probabilities[best_likelihood_class] / evidence
        true_class = test_data[id]["data"][8]
        accuracy = 0.0
        if true_class == best_likelihood_class:
            accuracy = 1.0
        print(f"ID={id : 5d}, predicted={best_likelihood_class : 3d}, probability = {best_likelihood : .4f}, true={true_class}, accuracy={accuracy : 4.2f}")
        accuracy_sum += accuracy
    overall_accuracy = accuracy_sum / len(test_data)
    print(f"classification accuracy={overall_accuracy : 6.4f}")
    