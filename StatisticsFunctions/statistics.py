import numpy as np
import scipy

FILE_A = "Results/result-Res.txt"
FILE_B = "Results/result-Res-Yes.txt"


def readFile (path):
    data = {}
    with open(path, "r") as f:
        next(f) # skip header
        for line in f:
            if "," not in line:
                return data # skip summary lines at end of file
            file = line.split(",")[0]
            nss = line.split(",")[1]
            data[file] = float(nss)
    return data

def calculateTStatisticAndPValue():
    data_a_dist = readFile(FILE_A)
    data_b_dict = readFile(FILE_B)

    data_a = []
    data_b = []
    for i in data_a_dist:
        data_a.append(data_a_dist[i])

    for i in data_b_dict:
        data_b.append(data_b_dict[i])

    data_a = np.array(data_a)
    data_b = np.array(data_b)

    t_statistic, p_value = scipy.stats.ttest_ind(data_b, data_a)
    return t_statistic, p_value


t_statistic, p_value = calculateTStatisticAndPValue()
print("t-score: " + str(t_statistic))
print("p-score: " + str(p_value))



