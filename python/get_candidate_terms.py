
import atire
import pickle as pkl
import os
import re

import numpy as np


"""

A program which scans through a list of topics, acquire the top 10 results and their terms. Then serialize the lot.
This is done so we wouldn't have to query the search engine everytime. 

"""

# WSJ, MSA, Jeopardy?,
script_dir = os.path.dirname(__file__)
wsj_topics_path = os.path.join(script_dir,"../atire_dataset/all_WSJ_topics.txt")
# wsj_topics_path = os.path.join(script_dir, "../evaluation/topic_qrel/topics/topics.51-55.txt")



msa_topics_path = os.path.join(script_dir,"../atire_dataset/MSA/msa_all_topics.txt")


DATASETS_PATH = {
    "WSJ":wsj_topics_path
    # "MSA":msa_topics_path
}




def read_topic_file(topic_file_path,topic_list):
    """Read a TREC topic file and parse it into a dictionary"""
    f = open(topic_file_path,'r')
    for line in f:
        topic_id = line.split()[0]
        original_query = " ".join(line.split()[1::])
        topic_list[topic_id] = original_query


def retrieve_document_terms(query):
    """
    :param query: A query to pass to the search engine
    :return: a dictionary of term-Word2Vec embedding pairs, in the order the terms appear in the collection.
    """
    tokens = []
    results = atire.lookup(-1, query)

    for result in results:
        tokens.append(atire.get_ordered_tokens(result))

    return tokens


def load_pickled(path):
    return pkl.load(open(path, "rb"))

if __name__ == "__main__":

    atire.init("atire -findex wiki_index.aspt")

    for dataset in DATASETS_PATH:
        token_cache = {}
        querys_table = {}
        read_topic_file(DATASETS_PATH[dataset], querys_table)

        for topicID in querys_table:

            current_query = []
            current_query.append(re.sub(r'\W+', " ", querys_table[topicID]).lower())

            terms_in_results = retrieve_document_terms(" ".join(current_query))
            terms_in_query = current_query[0].split(" ")
            terms_in_results.insert(0, terms_in_query)
            token_cache["".join(current_query)] = tuple(terms_in_results)

        pkl.dump(token_cache, open(os.path.join(script_dir,"../atire_dataset/")+ dataset + " result_terms.p", "wb+"))

        print("Done")


