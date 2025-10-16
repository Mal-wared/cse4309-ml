from naive_bayes import *


# When you test your code, you can change this line to reflect where the 
# dataset directory is located on your machine.
dataset_directory = r"C:\Users\malwa\GitHub Repos\cse4309\Assignment 2"

# When you test your code, you can select the dataset you want to use 
# by commenting out the other dataset names.

#dataset = "pendigits"
#dataset = "satellite"
dataset = "yeast"

training_file = dataset_directory + "/" + dataset + "_training.txt"
test_file = dataset_directory + "/" + dataset + "_test.txt"

naive_bayes(training_file, test_file)