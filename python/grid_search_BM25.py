import atire
import os
import numpy as np


def read_topic_file(topic_file_path,topic_list):
    """Read a TREC topic file and parse it into a dictionary"""
    f = open(topic_file_path,'r')
    for line in f:
        topic_id = line.split()[0]
        original_query = " ".join(line.split()[1::])
        topic_list[topic_id] = original_query




def write_to_file(filename, information):
    """

    :param filename:
    :param v: a 2d list of training information
    :return:
    """
    with open(filename, "a") as f:
        message = ""
        for row in information:
            message += str(row[0]) + str(row[1]).strip('[]') + " "

        message += "\n"
        f.write(message)
    print("Wrote record to file")




script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

record_dir = os.path.join(script_dir,"grid_search")
if not os.path.exists(record_dir):
    os.makedirs(record_dir)


BM25_grid_search_file =  os.path.join(script_dir, record_dir,"BM25_grid_search.txt")





DATA_SET = "MSA"
METRIC = " -mMAP@40"





ASSESSMENT_FILE = os.path.join(script_dir,"../atire_dataset/MSA/full_msa.qrels")
TEST_TOPIC_FILE = os.path.join(script_dir,"../atire_dataset/MSA/msa_testing_topics.txt")
# TEST_TOPIC_FILE = os.path.join(script_dir,"../atire_dataset/MSA/msa_valid_topics.100.txt")


test_query_table={}
read_topic_file(TEST_TOPIC_FILE,test_query_table)



K1_range = np.linspace(0,3,11)
B_range = np.linspace(0,1,11)

K1_range = K1_range[1:]
B_range = B_range[1:]

for K1 in K1_range:
    for B in B_range:
        BM_25_value = " -RBM25:" + str(K1) + ":" + str(B)
        params = "atire -a " + ASSESSMENT_FILE + METRIC + BM_25_value
        atire.init(params)
        mean_average_precision = 0
        average_precision = []

        for topicID in test_query_table:
            average_precision.append(atire.lookup(int(topicID), test_query_table[topicID]))

        mean_average_precision = np.mean(average_precision)

        info = [
            ["K1 Value: ", K1],
            ["B Value: ", B],
            ["MAP@40: ", mean_average_precision]
        ]

        write_to_file(BM25_grid_search_file, info)
        atire.cleanup()



