# Task 10 (10 pts)

# Computes the average and standard deviation of each column of numbers contained in some specified file

# Invoke the following command:
# python file_stats.py pathname

# pathname is the path name of the data file; should be a file stored on the local computer (not justin the current directory)

# The specified file will contain data in a tabular format, so that each value is a number, and values are separated by whitespace

import sys
import statistics

def main(pathname):
    data = []
    map_file_data(pathname, data)

    transposed_data = [[0 for _ in range(len(data))] for _ in range(len(data[0]))]
    swap_data_rows_and_cols(data, transposed_data)

    means = []
    compute_mean(transposed_data, means)

    stdevs = []
    compute_stdev(transposed_data, stdevs)

    for i in range(len(transposed_data)):
        print(f"Column {i}: mean = {means[i]: .4f}, std = {stdevs[i]: .4f}")

def compute_stdev(data, stdevs):
    for i in range(len(data)):
        stdevs.append(statistics.stdev(data[i]))
    

def compute_mean(data, means):
    for i in range(len(data)):
        sum = 0
        for j in range(len(data[i])):
            sum = sum + data[i][j]
        means.append(sum / len(data[i]))
        
    ''' print neatly
    for i in range(len(means)):
        print(f"Column {i}: mean = {means[i]: .4f}")
    '''

def swap_data_rows_and_cols(data, transposed_data):
    # transposes original data set so that columns are easier to analyze
    for i in range(len(data)):
        for j in range(len(data[i])):
            transposed_data[j][i] = float(data[i][j])

    ''' print neatly
    for line in transposed_data:
        print(line)
    '''

def map_file_data(pathname, data):
    # read file and populate the 'data' variable for future usage
    with open(pathname, 'r') as file:
        for line in file:
            data.append(line.split())

    ''' print data neatly
    for line in data:
        print(line)
    '''

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("ERROR: Incorrect number of arguments. Please utilize the following format: python file_stats.py <pathname>")
    else:
        pathname = sys.argv[1]
        main(pathname)