import multiprocessing
import subprocess
import sys


def main():
    """
    This method can be used to parallel the process of running the feature attribution methods
    and to run multiple instances.
    Change the number of iterations per instance in the feature attribution method individual file and change 'num = '.
    """
    number_of_runners = 4  # Set the number of runners
    number_of_instances = 100  # Set the number of instances

    # Select one of the feature attribution method files
    python_file = "{lime_implementation," \
                  "kernel_shap_implementation," \
                  "integrated_gradients_docs_only_implementation," \
                  "integrated_gradients_claim_and_docs_implementation}.py"

    base_command = [sys.executable, python_file]
    arguments = [[*base_command, str(i)] for i in range(number_of_instances)]

    with multiprocessing.Pool(number_of_runners) as pool:
        pool.map(
            subprocess.run, arguments
        )


if __name__ == "__main__":
    main()
