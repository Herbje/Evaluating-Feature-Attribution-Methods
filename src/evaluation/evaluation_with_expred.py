import json
import numpy
import scipy
from scipy import stats
from tabulate import tabulate
import matplotlib.pyplot as plt


def read_files(name_set, num):
    """
    Read the .json files from LIME, Kernel SHAP and Integrated Gradients.

    Args:
        name_set: The name of the dataset (test or train).
        num: The number of the instance.

    Returns:
        The ranking from LIME, Kernel SHAP and Integrated Gradients, and the explanation from ExPred.
    """
    # ExPred
    expred = open('/dir/to/your/expred/' + name_set + '/expred_' + str(num) + '.json', 'r')
    expred_data = json.load(expred)
    expred_docs = expred_data['scores_list']
    expred_docs = numpy.asarray(expred_docs, dtype=int)
    expred.close()

    # LIME
    lime = open('/dir/to/your/lime_base/' + name_set + '/lime_base_' + str(num) + '.json', 'r')
    lime_data = json.load(lime)
    lime_ranking = lime_data['scores_list']
    lime_ranking = len(lime_ranking) - stats.rankdata(numpy.asarray(lime_ranking, dtype=float))
    lime.close()

    # Kernel SHAP
    kernel_shap = open('/dir/to/your/kernel_shap/' + name_set + '/kernel_shap_' + str(num) + '.json', 'r')
    kernel_shap_data = json.load(kernel_shap)
    kernel_shap_ranking = kernel_shap_data['scores_list']
    kernel_shap_ranking = len(kernel_shap_ranking) - stats.rankdata(numpy.asarray(kernel_shap_ranking, dtype=float))
    kernel_shap.close()

    # Integrated Gradients
    integrated_gradients = open('/dir/to/your/integrated_gradients/'
                                + name_set + '/integrated_gradients_' + str(num) + '.json', 'r')
    integrated_gradients_data = json.load(integrated_gradients)
    integrated_gradients_ranking = integrated_gradients_data['scores_list']
    integrated_gradients_ranking = numpy.asarray(integrated_gradients_ranking, dtype=float)
    integrated_gradients_ranking = [ig for ig in integrated_gradients_ranking if ig != 0]
    integrated_gradients_ranking = len(integrated_gradients_ranking) - stats.rankdata(integrated_gradients_ranking)
    integrated_gradients.close()

    return expred_docs, lime_ranking, kernel_shap_ranking, integrated_gradients_ranking


def top_n_ranking(top_n, rankings):
    """
    Change the n-highest scores to a 1 and the rest to a 0.

    Args:
        top_n: The top n-highest scores.
        rankings: The rankings for a feature attribution method.

    Returns:
        A list with 1s and 0s, where the 1s are the highest scores.
    """
    result = []
    for i in rankings:
        if i < top_n:
            result.append(1)
        else:
            result.append(0)
    return result


def jaccard_similarity(expred_docs, lime_ranking, kernel_shap_ranking, integrated_gradients_ranking):
    """
    Compute the Jaccard Distance between the feature attribution methods and ExPreds explanation.

    Args:
        expred_docs: A list with ExPreds explanation.
        lime_ranking: A list with LIMEs rankings changed to 1s and 0s.
        kernel_shap_ranking: A list with Kernel SHAPs rankings changed to 1s and 0s.
        integrated_gradients_ranking: A list with Integrated Gradients' rankings changed to 1s and 0s.

    Returns:
        The Jaccard Distances for the methods with ExPred.
    """
    jaccard1 = scipy.spatial.distance.jaccard(lime_ranking, expred_docs)
    jaccard2 = scipy.spatial.distance.jaccard(kernel_shap_ranking, expred_docs)
    jaccard3 = scipy.spatial.distance.jaccard(integrated_gradients_ranking, expred_docs)
    return jaccard1, jaccard2, jaccard3


def create_file(name_of_set, num, current_table, j_1, j_2, j_3):
    """
    Creates and saves a file with the results of the Jaccard Distance.

    Args:
        name_of_set: The name of the dataset (test or train).
        num: The percentage of the total tokens that are selected to become 1s.
        current_table: The results in a table format.
        j_1: The Jaccard Distance between LIME and ExPred.
        j_2: The Jaccard Distance between Kernel SHAP and ExPred.
        j_3: The Jaccard Distance between Integrated Gradients and ExPred.
    """
    with open("/dir/to/your/evaluation/" + name_of_set + "/jaccard_similarity_percentage_" + str(num) + ".txt", "w") as f:
        f.write(current_table)
        f.write("\n\nLIME/ExPred jaccard average " + str(j_1)
                + "\nKernel SHAP/ExPred jaccard average " + str(j_2)
                + "\nIntegrated Gradients/ExPred jaccard average " + str(j_3))
        f.close()


def create_table(dataset, max_num, top=5):
    """
    Compute the Jaccard Distance between the feature attribution methods and ExPreds explanation.

    Args:
        dataset: The name of the dataset (test or train).
        max_num: The amount of instances.
        top: The top n that will be selected.

    Returns:
        A table of the results from the Jaccard Distance computation.
    """
    table_data = []
    jaccard1_mean, jaccard2_mean, jaccard3_mean = [], [], []

    for i in range(max_num):
        expred_docs, lime_ranking, kernel_shap_ranking, integrated_gradients_ranking = read_files(dataset, i)
        # Limit all lengths to the length of Integrated Gradients
        jaccard1, jaccard2, jaccard3 = jaccard_similarity(expred_docs[:len(integrated_gradients_ranking)],
                                                          top_n_ranking(top / 100 * len(integrated_gradients_ranking),
                                                                        lime_ranking[
                                                                        :len(integrated_gradients_ranking)]),
                                                          top_n_ranking(top / 100 * len(integrated_gradients_ranking),
                                                                        kernel_shap_ranking[
                                                                        :len(integrated_gradients_ranking)]),
                                                          top_n_ranking(top / 100 * len(integrated_gradients_ranking),
                                                                        integrated_gradients_ranking))

        table_data.append([i, jaccard1, jaccard2, jaccard3])
        jaccard1_mean.append(jaccard1)
        jaccard2_mean.append(jaccard2)
        jaccard3_mean.append(jaccard3)

    head = ['Instance', "LIME/ExPred jaccard", "Kernel SHAP/ExPred jaccard", "Integrated Gradients/ExPred jaccard"]
    return tabulate(table_data, headers=head, tablefmt="grid"), numpy.mean(jaccard1_mean), \
           numpy.mean(jaccard2_mean), numpy.mean(jaccard3_mean)


if __name__ == '__main__':
    n_list = []
    lime_list = []
    shap_list = []
    ig_list = []
    set_name = 'test_set'

    for i in range(0, 101):  # From 0 to 100 percent
        n_list.append(i)
        table, j1, j2, j3 = create_table(set_name, 100, top=i)
        create_file(set_name, i, table, j1, j2, j3)
        lime_list.append(j1)
        shap_list.append(j2)
        ig_list.append(j3)

    # Create plot
    plt.plot(n_list, lime_list)
    plt.plot(n_list, shap_list)
    plt.plot(n_list, ig_list)
    plt.legend(["LIME", 'Kernel SHAP', "Integrated Gradients"])
    plt.xlabel('Percentage of Selected Highest Scoring Tokens')
    plt.ylabel('Jaccard Distance')
    plt.xlim((0, 100))
    plt.ylim(top=1)
    plt.xticks(range(0, 101, 10))
    plt.savefig('figures/' + set_name + '.pdf')
    plt.show()
