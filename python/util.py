import matplotlib
import atire


import pandas as pd
import matplotlib.pyplot as plt
import os


def read_data_file(filename,item_pos=3):
    """
    Opens a data file and read the relevant data into a list
    :param filename:
    :return:
    """
    output = []
    with open(filename, "r") as f:
        for line in f:
            output.append(float(line.split()[item_pos]))

    return output


def draw_graph(arr):
    """
    Generates graph from list of data
    :param arr: a 2-D list of data
    :return:
    """

    for item in arr:
        plt.plot(item[0][::10],label=item[1])

    plt.xlabel('Per 20 epochs')
    plt.ylabel('MAP@40')
    plt.title("MAP@40 of validation and test data")

    plt.legend()

    plt.show()






if __name__ == "__main__":


    data_to_graph = []

    filename = "GOOGLE_W2V_FC256_Small_offset_in_loss_function_"

    script_dir = os.path.dirname(__file__)
    record_dir = os.path.join(script_dir,"record", filename)

    test_time = read_data_file(os.path.join(record_dir,"test_time MAP-Unseen query"),
                               item_pos=3)
    validation = read_data_file(os.path.join(record_dir,"validation MAP"
                                                                ),item_pos=3)

    # all_docs_random = read_data_file(os.path.join(os.path.join(record_dir,"graph_generations",
    #                                                             "test time MAP")),item_pos=3)



    df = pd.DataFrame(test_time)
    moving_ave = df.rolling(window=20).mean()




    validation_raw_value = 0.086467
    validation_roccio_PRF = 0.0924303

    test_raw_value = 0.0639842
    test_roccio_PRF = 0.0666502

    validation_raw = []
    validation_rocchio = []

    test_raw = []
    test_roccio = []


    for i in range(0,len(test_time)):
        validation_raw.append(validation_raw_value)
        validation_rocchio.append(validation_roccio_PRF)
        test_raw.append(test_raw_value)
        test_roccio.append(test_roccio_PRF)







    data_to_graph.append([test_time,"Test on unseen documents"])
    data_to_graph.append([moving_ave.values,"Moving average of test performance"])

    data_to_graph.append([validation,"validation on seen documents"])

    # data_to_graph.append([validation_raw,"Raw atire, validation batch"])
    # data_to_graph.append([validation_rocchio,"Rocchio, validation batch"])

    # data_to_graph.append([test_raw, "raw ATIRE, Test batch"])
    # data_to_graph.append([test_roccio,"Test batch Rocchio"])

    draw_graph(data_to_graph)



