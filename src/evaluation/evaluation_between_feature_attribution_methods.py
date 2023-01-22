import json
import numpy
from scipy import stats
from tabulate import tabulate


def read_files(name_set, num):
    """
    Read the .json files from LIME, Kernel SHAP and Integrated Gradients.

    Args:
        name_set: The name of the dataset (test or train).
        num: The number of the instance.

    Returns:
        The ranking from LIME, Kernel SHAP and Integrated Gradients.
    """
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

    return lime_ranking, kernel_shap_ranking, integrated_gradients_ranking


def kendall_tau(lime_ranking, kernel_shap_ranking, integrated_gradients_ranking):
    """
    Compute Kendall's Tau between all feature attribution methods.
    Between Integrated Gradients and another method limit length of the other method.

    Args:
        lime_ranking: The rankings from LIME.
        kernel_shap_ranking: The rankings from Kernel SHAP.
        integrated_gradients_ranking: The rankings from Integrated Gradients.

    Returns:
        The results from Kendall's Tau (tau and p-value) for every pair of methods.
    """
    tau1, p_value1 = stats.kendalltau(lime_ranking, kernel_shap_ranking)
    tau2, p_value2 = stats.kendalltau(lime_ranking[:len(integrated_gradients_ranking)],
                                      integrated_gradients_ranking)
    tau3, p_value3 = stats.kendalltau(kernel_shap_ranking[:len(integrated_gradients_ranking)],
                                      integrated_gradients_ranking)
    return tau1, p_value1, tau2, p_value2, tau3, p_value3


def create_table(dataset, max_num):
    """
    Compute the Kendall's Tau between the feature attribution methods.

    Args:
        dataset: The name of the dataset (test or train).
        max_num: The amount of instances.

    Returns:
        A table of the results from the Kendall's Tau computation.
    """
    table_data = []
    tau1_mean, p_value1_mean, tau2_mean, p_value2_mean, tau3_mean, p_value3_mean = [], [], [], [], [], []

    for i in range(max_num):
        lime_ranking, kernel_shap_ranking, integrated_gradients_ranking = read_files(dataset, i)
        tau1, p_value1, tau2, p_value2, tau3, p_value3 = kendall_tau(lime_ranking, kernel_shap_ranking,
                                                                     integrated_gradients_ranking)
        table_data.append([i, tau1, p_value1, tau2, p_value2, tau3, p_value3])
        tau1_mean.append(tau1)
        p_value1_mean.append(p_value1)
        tau2_mean.append(tau2)
        p_value2_mean.append(p_value2)
        tau3_mean.append(tau3)
        p_value3_mean.append(p_value3)

    head = ['Instance',
            "LIME/Kernel SHAP tau", "LIME/Kernel SHAP p_value",
            "LIME/Integrated Gradients tau", "LIME/Integrated Gradients p_value",
            "Kernel SHAP/Integrated Gradients tau", "Kernel SHAP/Integrated Gradients p_value"]
    return tabulate(table_data, headers=head, tablefmt="grid"), \
           numpy.mean(tau1_mean), numpy.mean(p_value1_mean), \
           numpy.mean(tau2_mean), numpy.mean(p_value2_mean), \
           numpy.mean(tau3_mean), numpy.mean(p_value3_mean)


if __name__ == '__main__':
    set_name = 'train_set'
    # Store table
    with open("/dir/to/your/evaluation/" + set_name + "/kendall_tau.txt", "w") as f:
        table, t1, p1, t2, p2, t3, p3 = create_table(set_name, 100)
        f.write(table)
        f.write("\n\nLIME/Kernel SHAP tau average " + str(t1)
                + "\nLIME/Kernel SHAP p_value average " + str(p1)
                + "\nLIME/Integrated Gradients tau average " + str(t2)
                + "\nLIME/Integrated Gradients p_value average " + str(p2)
                + "\nKernel SHAP/Integrated Gradients tau average " + str(t3)
                + "\nKernel SHAP/Integrated Gradients p_value average " + str(p3))
        f.close()
