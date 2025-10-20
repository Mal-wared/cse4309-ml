from decision_tree import *
import sys
import io
import numpy as np
import time


# When you test your code, you can change this line to reflect where the 
# dataset directory is located on your machine.
dataset_directory = r"C:\Users\malwa\GitHub Repos\cse4309\Assignment5"

# When you test your code, you can select the dataset you want to use 
# by modifying the next lines
dataset = "pendigits_string"
#dataset = "satellite"
#dataset = "yeast"


training_file = dataset_directory + "/" + dataset + "_training.txt"
test_file = dataset_directory + "/" + dataset + "_test.txt"

# When you test your code, you can select the function arguments you want to use 
# by modifying the next lines
option = "optimized"
#option = 1
#option = 3
#option = 15
pruning_thr = 50

def run_experiment(training_file, test_file, option, pruning_thr, runs=15):
    """
    Runs the decision_tree function multiple times and captures its output
    to calculate the average accuracy.
    """
    accuracies = []
    print(f"--- Starting Experiment ---")
    print(f"Params: option='{option}', pruning_thr={pruning_thr}, runs={runs}")
    
    start_time = time.time()

    for i in range(runs):
        print(f"  Running iteration {i+1}/{runs}...")

        # --- Redirect stdout to capture print output ---
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        # -----------------------------------------------

        try:
            # Call your function. All its print() statements
            # will go into 'redirected_output' instead of the console.
            decision_tree(training_file, test_file, option, pruning_thr)
        finally:
            # --- Restore stdout ---
            sys.stdout = old_stdout
            # ----------------------

        # --- Parse the captured output ---
        output = redirected_output.getvalue().strip()
        
        # Find the last line of the output
        if not output:
            print(f"    Error: No output captured for run {i+1}.")
            continue

        last_line = output.split('\n')[-1]

        if last_line.startswith('classification accuracy='):
            try:
                acc_str = last_line.split('=')[1]
                acc_val = float(acc_str)
                accuracies.append(acc_val)
            except (IndexError, ValueError):
                print(f"    Error: Could not parse accuracy from line: '{last_line}'")
        else:
            print(f"    Error: Last line was not accuracy: '{last_line}'")
        # ---------------------------------

    end_time = time.time()
    print(f"\nTotal time for experiment: {end_time - start_time:.2f} seconds")

    if accuracies:
        avg_accuracy = np.mean(accuracies)
        min_accuracy = np.min(accuracies)
        max_accuracy = np.max(accuracies)
        median_accuracy = np.median(accuracies)

        print(f"\n--- Experiment Results for {len(accuracies)} runs ---")
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        print(f"Median Accuracy:  {median_accuracy:.4f}  <-- Good for '5 out of 10' rule")
        print(f"Min Accuracy:     {min_accuracy:.4f}")
        print(f"Max Accuracy:     {max_accuracy:.4f}")
        
        print("\nAll accuracies (sorted):")
        print(sorted(accuracies))
    else:
        print("\n--- Experiment Failed ---")
        print("No accuracy results were captured.")

if __name__ == "__main__":
    
    if len(sys.argv) != 5:
        print("Usage: python decision_tree_main.py <training_file> <test_file> <option> <pruning_thr>")
        sys.exit(1)

    training_file = sys.argv[1]
    test_file = sys.argv[2]
    option_arg = sys.argv[3]
    pruning_thr_arg = int(sys.argv[4]) # Pruning threshold needs to be an int

    # For 'optimized', we only need to run it once since it's deterministic
    num_runs = 1 if option_arg == 'optimized' else 15

    run_experiment(training_file, test_file, option_arg, pruning_thr_arg, runs=num_runs)
    

    #decision_tree(training_file, test_file, option, pruning_thr)