from linear_regression import *


# When you test your code, you can change this line to reflect where the 
# dataset directory is located on your machine.
dataset_directory = r"C:\Users\malwa\GitHub Repos\cse4309\Assignment 3"

# When you test your code, you can select the dataset you want to use 
# by modifying the next lines

training_file = "f1.txt"
test_file = "f1.txt"
#training_file = "boston_housing_training.txt"
#test_file = "boston_housing_test.txt"

training_file = dataset_directory + "/" + training_file
test_file = dataset_directory + "/" + test_file

# When you test your code, you can select the hyperparameters you want to use 
# by modifying the next lines
degree = 1
lambda1 = 0

linear_regression(training_file, test_file, degree, lambda1)
