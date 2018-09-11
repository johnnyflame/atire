import atire
import numpy as np
import tensorflow as tf
import pickle as pkl
import tensorflow.contrib.slim as slim
import re
from itertools import islice
import os
from random import sample, randint, random
import random
import gensim
import datetime

"""
"First system"

A prototype of the model described in the paper "Task-Oriented Query Reformulation with Reinforcement Learning", 
Nogueira and Cho 2017


Author: Johnny Flame Lee 

"""

max_epochs=5000


STARTING_LERNING_RATE = 1e-4
K_TERMS = 200
fc_neurons = 256

SHUFFLE = True
TRAIN_BATCH_SIZE = 64

DATA_SET = "WSJ" # Avalaible option includes ["WSJ","MSA","TREC-CAR","JEOPARDY","PRE_TRAIN"]
load_model = "PRE_TRAINED" # ["CONTINUE","NONE","PRE_TRAINED"]


# How frequenty to test the network
TEST_FREQUENCY = 5


FREEZE_POLICY = True

episode = 0
epoch = 0
INDEX_FLAG = ""
DATE = str(datetime.date.today())

FILENAME = "WSJ50 queries"
METRIC = " -mMAP@40"

SUBDIR = DATA_SET + "_" + FILENAME + "_" + DATE


########################################################################################################################
script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
record_dir = os.path.join(script_dir,"record",SUBDIR)


if not os.path.exists(record_dir):
    os.makedirs(record_dir)
weights_dir = os.path.join(script_dir,"weights",SUBDIR)
if not os.path.exists(weights_dir):
    os.mkdir(weights_dir)
weight_saving_path = os.path.join(weights_dir,"model.ckpt")



reformulated_query_filepath = os.path.join(script_dir, record_dir,"test time reformulated query-Unseen query")

SL_reformulated = os.path.join(script_dir, record_dir,"SL reformulated")


average_training_MAP = os.path.join(script_dir, record_dir,"validation MAP")
training_records = os.path.join(script_dir, record_dir,"training_records.txt")
test_time_MAP = os.path.join(script_dir, record_dir,"test_time MAP-Unseen query")

epoch_update_records = os.path.join(script_dir, record_dir,"epoch_update_records.txt")

reformulated_cache = os.path.join(script_dir, "../atire_dataset/") + DATA_SET + " reformulated_query_score.p"




if DATA_SET == "MSA":
    TOPIC_FILE = os.path.join(script_dir,"../atire_dataset/MSA/train_50k.txt")
    ASSESSMENT_FILE = os.path.join(script_dir,"../atire_dataset/MSA/full_msa.qrels")
    TEST_TOPIC_FILE = os.path.join(script_dir,"../atire_dataset/MSA/rand_valid_topics_500.txt")
    TRAIN_BATCH_SIZE = 64
    # maximum number of terms in q0
    MAX_SEQUENCE_LENGTH = 70

    params = "atire " + INDEX_FLAG + " -a " + ASSESSMENT_FILE + METRIC



elif DATA_SET == "WSJ":
    TOPIC_FILE = os.path.join(script_dir, "../evaluation/topic_qrel/topics/topics.51-100.txt")
    TEST_TOPIC_FILE = os.path.join(script_dir, "../evaluation/topic_qrel/topics/topics.101-150.txt")
    # TOPIC_FILE = os.path.join(script_dir, "../evaluation/topic_qrel/topics/topics.51-55.txt")
    # TEST_TOPIC_FILE = os.path.join(script_dir, "../evaluation/topic_qrel/topics/topics.51-55.txt")
    ASSESSMENT_FILE = os.path.join(script_dir,"../evaluation/WSJ.qrels")

    INDEX_FLAG = "-findex ./index_WSJ.aspt"
    MAX_SEQUENCE_LENGTH = 15
    params = "atire " + INDEX_FLAG + " -a " + ASSESSMENT_FILE + METRIC
    pretrained_weights_save_path = os.path.join(script_dir,"/weights/pretrained_weights/WSJ_51-100/model.ckpt")


elif DATA_SET == "PRE_TRAIN":

    # METRIC = ""
    # GROUND_TRUTH = os.path.join(script_dir,"../atire_dataset/MSA/supervised/rocchio_full_270k.txt")
    # TOPIC_FILE = os.path.join(script_dir, "../atire_dataset/MSA/msa_train_topics.txt")
    # ASSESSMENT_FILE = os.path.join(script_dir, "../atire_dataset/MSA/full_msa.qrels")
    # TRAIN_BATCH_SIZE = 64
    # # maximum number of terms in q0
    # MAX_SEQUENCE_LENGTH = 70
    # TEST_TOPIC_FILE = os.path.join(script_dir,"../atire_dataset/MSA/rand_valid_topics_500.txt")
    # params = "atire " + INDEX_FLAG


    METRIC = ""
    GROUND_TRUTH = os.path.join(script_dir,"../atire_dataset/WSJ/51-100.feedback")
    TOPIC_FILE = os.path.join(script_dir, "../evaluation/topic_qrel/topics/topics.51-100.txt")
    ASSESSMENT_FILE = os.path.join(script_dir, "../evaluation/WSJ.qrels")
    TRAIN_BATCH_SIZE = 10
    # maximum number of terms in q0
    MAX_SEQUENCE_LENGTH = 15
    TEST_TOPIC_FILE = os.path.join(script_dir, "../evaluation/topic_qrel/topics/topics.101-150.txt")

    INDEX_FLAG = "-findex ./index_WSJ.aspt"
    params = "atire " + INDEX_FLAG + " -a " + ASSESSMENT_FILE + METRIC
    WORD_EMBEDDING_PATH =  os.path.join(script_dir,"../atire_dataset/WSJ/word_embeddings/wsj-collection-vectors")
    CANDIDATE_CACHE_FILE = os.path.join(script_dir,"../atire_dataset/WSJ/WSJ result_terms.p")

# Word embedding file
# WORD_EMBEDDING_PATH =  os.path.join(script_dir,"../atire_dataset/GoogleNews-vectors-negative300.bin")



########################################################################################################################

CREATE_DICTIONARY = True
CONTEXT_WINDOW = 4
# Total number of terms to go into the second network
CANDIDATE_AND_CONTEXT_LENGTH = CONTEXT_WINDOW * 2 + 1


WORD_VECTOR_DIMENSIONS = 300



PADDING = np.zeros(WORD_VECTOR_DIMENSIONS)


random.seed(500)


# HYPERPARAMETERS:
atire.init(params)



def extract_minibatch(dataset,minibatch_size):
    """Returns a randomly sampled subset from a dataset"""
    keys = list(dataset)
    n = len(keys)
    out = []

    np.random.shuffle(keys)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(keys[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(keys[minibatch_start:])

    for batch in minibatches:
        subset = {}
        for index in batch:
            subset[index] = dataset[index]
        out.append(subset)


    return out


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



def load_lookup_table(file):
    """Return a dictionary of term-vector pairs"""
    return pkl.load(open(file, "rb"))



def load_vocab(path, n_words=None):
    dic = pkl.load(open(path, "rb"),encoding='utf-8')
    vocab = {}

    if not n_words:
        n_words = len(dic.keys())

    for i, word in enumerate(dic.keys()[:n_words]):
        vocab[word] = i
    return vocab


def read_topic_file(topic_file_path,topic_list):
    """Read a TREC topic file and parse it into a dictionary"""
    f = open(topic_file_path,'r')
    for line in f:
        topic_id = line.split()[0]
        original_query = " ".join(line.split()[1::])
        topic_list[topic_id] = original_query

def get_epoch_num(file):
    with open(file, 'r') as f:
        lines = f.read().splitlines()
        last_line = lines[-1]
        return int(last_line.split()[1])




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

# gamma = 0.99
#
# def discount_rewards(r):
#     """ take 1D float array of rewards and compute discounted reward """
#     discounted_r = np.zeros_like(r)
#     running_add = 0
#     for t in reversed(range(0, r.size)):
#         running_add = running_add * gamma + r[t]
#         discounted_r[t] = running_add
#     return discounted_r




class GenerateNetwork:

    def __init__(self,number_of_terms):
        self.query_input = tf.placeholder(tf.float32, [None, WORD_VECTOR_DIMENSIONS], name="query_input")
        self.candidate_and_context_input = tf.placeholder(tf.float32, [None, WORD_VECTOR_DIMENSIONS]
                                                          , name="candidate_vectors")

        self.action_choice = tf.placeholder(tf.int32,[None,1],name="actions")

        if DATA_SET == "PRE_TRAIN":
            self.true_label = tf.placeholder(tf.int32, [None, 1], name="true_label")

        # Reshaping the query so it becomes Rank 4, the order is [batch_size, width,height, channel]
        self.reshaped_query_input = tf.reshape(self.query_input, [-1, number_of_terms, WORD_VECTOR_DIMENSIONS, 1])
        self.reshaped_candidate_and_context = tf.reshape(self.candidate_and_context_input,
                                                         [-1, CANDIDATE_AND_CONTEXT_LENGTH, WORD_VECTOR_DIMENSIONS, 1])

        # Add 2 convolutional layers with ReLu activation
        with tf.variable_scope("policy", reuse=tf.AUTO_REUSE):

            self.query_conv1 = slim.conv2d(
                self.reshaped_query_input, num_outputs=256,
                kernel_size=[3,WORD_VECTOR_DIMENSIONS], stride=[1,1], padding='VALID', biases_initializer=tf.zeros_initializer()
            )

            # Second convolution layer
            self.query_conv2 = slim.conv2d(
                self.query_conv1, num_outputs=256,
                kernel_size=[3,1], stride=[1,1], padding='VALID', biases_initializer=tf.zeros_initializer()
            )




            # Not super confident about these parameters, may need revisit
            self.query_pooled = tf.nn.max_pool(
                self.query_conv2,
                ksize=[1,MAX_SEQUENCE_LENGTH-4,1,1],
                strides=[1, 1, 1,1],
                padding='VALID',
                name="pool")

            self.candidates_conv1 =  slim.conv2d(
                self.reshaped_candidate_and_context, num_outputs=256,
                kernel_size=[5,WORD_VECTOR_DIMENSIONS], stride=[1,1], padding='VALID', biases_initializer=tf.zeros_initializer()
            )

            self.candidates_conv2 =  slim.conv2d(
                self.candidates_conv1, num_outputs=256,
                kernel_size=[3,1], stride=[1,1], padding='VALID', biases_initializer=tf.zeros_initializer()
            )

            self.candidates_pooled = tf.nn.max_pool(
                self.candidates_conv2,
                ksize=[1, 3, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")

            self.pooled_vectors_concatenated = tf.concat([self.query_pooled, self.candidates_pooled], 3)

            self.policy_fc1 = tf.contrib.layers.fully_connected(tf.reshape(self.pooled_vectors_concatenated,[-1,512]),
                                                                num_outputs=fc_neurons,activation_fn=tf.nn.tanh,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           biases_initializer=tf.constant_initializer(0.0001))

            # self.aprob = slim.fully_connected(self.policy_fc1,1,weights_initializer=tf.contrib.layers.xavier_initializer(),biases_initializer=tf.zeros_initializer(),activation_fn=tf.nn.sigmoid)

            self.aprob = slim.fully_connected(self.policy_fc1, 1,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.sigmoid)




        self.mean_context_vector = tf.reduce_mean(self.candidates_pooled,axis=0)
        self.mean_context_and_query_concatenated = tf.concat((tf.reduce_mean(self.query_pooled,axis=0),self.mean_context_vector),axis=2)


        with tf.variable_scope("value",reuse=tf.AUTO_REUSE):

            self.value_fc1 = tf.contrib.layers.fully_connected(tf.reshape(self.mean_context_and_query_concatenated,[1,512]),
                                                                num_outputs=fc_neurons,activation_fn=tf.nn.tanh,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           biases_initializer=tf.zeros_initializer())

            self.value_prediction = tf.contrib.layers.fully_connected(self.value_fc1,num_outputs=1,activation_fn=tf.nn.sigmoid,
                                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                                        biases_initializer=tf.zeros_initializer())

        self.value = tf.squeeze(self.value_prediction)


        self.learning_rate = tf.placeholder(shape=[],dtype=tf.float32)



        # Using the same training parameters from the paper
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-8)

        # Update the parameters according to the computed gradient.
        # train_step = optimizer.minimize(loss)

        # self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        self.reward = tf.placeholder(shape=[],dtype=tf.float32)
        #self.predicted_reward = tf.placeholder(shape=[],dtype=tf.float32)
        self.predicted_reward = tf.stop_gradient(self.value_prediction)

        if DATA_SET == "PRE_TRAIN":
            # Do Cross Entropy here
            self.policy_loss =  -tf.reduce_sum(
                tf.to_float(self.true_label) * tf.log((self.aprob + 1e-8)) +
                tf.to_float(1 - self.true_label) * tf.log((self.aprob + 1e-8)))
        else:
            self.policy_loss = (self.reward - self.predicted_reward) * -tf.reduce_sum(
                tf.to_float(self.action_choice) * tf.log((self.aprob + 1e-8)))

        # self.policy_loss = self.loss((self.reward-self.predicted_reward))


        vars = tf.trainable_variables()
        self.value_fc_variables = [v for v in vars if v.name.startswith("value")]
        self.policy_weights = [v for v in vars if v.name.startswith("policy")]


        self.value_loss = 0.001 * tf.square((self.reward - self.value_prediction))
        self.train_policy_network = self.optimizer.minimize(loss=self.policy_loss,var_list=[self.policy_weights])
        # self.value_loss = 0.1 * tf.nn.l2_loss(self.reward - self.value_prediction)
        self.train_value_network = self.optimizer.minimize(self.value_loss,
                                                           var_list=[self.value_fc_variables])



def lookup_term_vectors(terms):
    """
    Looks up the wordembedding for a set of terms,
    and returns a numpy array version of the vectors to be used in the model.

    :param query: the query to search for in the lookup table
    :return: a numpy ndarray, each entry corresponding to a term in the set.
    """

    query = []
    word_vector = []

    # query_terms.append([x for x in term.split(" ")])

    query.append(re.sub(r'\W+', " ", terms).lower())


    for terms in query:
        for word in terms.split(" "):
            # TODO: Retrain Word2Vec and remove this line
            if word not in word_embedding.wv.vocab:
                print ("word: " + word + " not found in pre-trained vocab list.\n")
                word_vector.append(np.zeros(shape=WORD_VECTOR_DIMENSIONS))
            else:
                word_vector.append(word_embedding.wv[word])


    if len(word_vector) < MAX_SEQUENCE_LENGTH:
        diff = MAX_SEQUENCE_LENGTH - len(word_vector)

        for i in range(0,diff):
            word_vector.append(PADDING)

    return query,np.array(word_vector)



def window(seq, n=3):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


if __name__ == "__main__":

    tf.reset_default_graph()  # Clear the Tensorflow graph.

    word_embedding = gensim.models.Word2Vec.load(WORD_EMBEDDING_PATH)
    # word_embedding = gensim.models.KeyedVectors.load_word2vec_format(WORD_EMBEDDING_PATH,binary=True)


    querys_table = {}
    test_query_table={}
    ground_truth = {}
    read_topic_file(TOPIC_FILE,querys_table)
    read_topic_file(TEST_TOPIC_FILE,test_query_table)



    result_cache = pkl.load(open(reformulated_cache),"rb")
    token_cache = pkl.load(open(CANDIDATE_CACHE_FILE, "rb"))


    if DATA_SET == "PRE_TRAIN":
        read_topic_file(GROUND_TRUTH, ground_truth)




    network = GenerateNetwork(number_of_terms=MAX_SEQUENCE_LENGTH)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # Uses 80% of GPU,
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        if load_model == "PRE_TRAINED":
            saver.restore(sess, pretrained_weights_save_path)
            epoch = 0
            print("Starting a new session using pretrained weights, episode = 0")
        elif load_model == "CONTINUE":
            saver.restore(sess, weight_saving_path)
            epoch = get_epoch_num(test_time_MAP)
            print("Model loaded successfully, resuming episode: " + str(episode))
        else:
            sess.run(init)
            epoch = 0
            print("Starting new session, episode = 0")


        # Training starts here
        while epoch < max_epochs:

            epoch += 1
            average_precisions = []
            average_policy_loss = []
            average_value_loss = []
            mean_average_precision = 0
            total_epoch_loss = []

            print("epoch: ", epoch)

            #Generate a set of n-batches for training.
            mini_batch_set = extract_minibatch(querys_table,TRAIN_BATCH_SIZE)
            for training_batch in mini_batch_set:
                for topicID in training_batch:
                    episode += 1
                    current_query_term_list = training_batch[topicID]
                    # This is the input to the left half of the neural network
                    current_query, query_vectors = lookup_term_vectors(current_query_term_list)
                    # Get a list of all the words in the top 10 documents
                    if "".join(current_query) not in token_cache.keys():
                        terms_in_results = retrieve_document_terms(" ".join(current_query))
                        terms_in_query = current_query[0].split(" ")
                        terms_in_results.insert(0, terms_in_query)
                        token_cache["".join(current_query)] = tuple(terms_in_results)
                    else:
                        terms_in_results = token_cache["".join(current_query)]

                    document_selected = []
                    # First append the original query
                    document_selected.append(terms_in_results[0])

                    if DATA_SET == "PRE_TRAIN":
                        ground_truth_label = []

                    if len(terms_in_results) > 1:
                        # Randomly sample from the other top documents
                        document_selected.append(terms_in_results[random.randint(1,len(terms_in_results)-1)][:K_TERMS])

                    # current_query = "South African"
                    reformulated_query = []
                    ep_history = []
                    candidate_terms = []
                    actions = []

                    # terms_in_results = [['60','50','sanction','bus','europe','50','bus','china','bus']]

                    # Each document in the top 10 results list
                    for doc in document_selected:
                        # For each term in one of the documents
                        for i in range(0,len(doc)):
                            candidate_and_context = []
                            candidate_term = (doc[i])
                            candidate_terms.append(candidate_term)

                            # This represents the state
                            candidate_and_context_vectors = []

                            # pad on the left
                            if i < CONTEXT_WINDOW:
                                diff = CONTEXT_WINDOW - i
                                candidate_and_context = doc[0:i + CONTEXT_WINDOW + 1]
                                for term in candidate_and_context:
                                    if term in word_embedding.wv.vocab:
                                        candidate_and_context_vectors.append(word_embedding.wv[term])
                                    else:
                                        candidate_and_context_vectors.append(PADDING)

                                for j in range(0,diff):
                                    candidate_and_context.insert(0,"$PADDING$")
                                    candidate_and_context_vectors.insert(0,PADDING)

                                if (len(candidate_and_context)) < CANDIDATE_AND_CONTEXT_LENGTH:
                                    for j in range(len(candidate_and_context),CANDIDATE_AND_CONTEXT_LENGTH):
                                        candidate_and_context.append("$PADDING$")
                                        candidate_and_context_vectors.append(PADDING)

                            # pad on the right---
                            elif (len(doc) - (i+1)) < CONTEXT_WINDOW:
                                # TODO: A known issue here: '' seperation at the end of each document is counted as a valid term, this may require fixing if it causes a problem in query reformulation.

                                diff = CONTEXT_WINDOW - (len(doc) - (i+1))
                                candidate_and_context = doc[i-CONTEXT_WINDOW:len(doc)]
                                for term in candidate_and_context:
                                    if term in word_embedding.wv.vocab:
                                        candidate_and_context_vectors.append(word_embedding.wv[term])
                                    else:
                                        candidate_and_context_vectors.append(PADDING)

                                for j in range(0, diff):
                                    candidate_and_context.append("$PADDING$")
                                    candidate_and_context_vectors.append(PADDING)

                                if (len(candidate_and_context)) < CANDIDATE_AND_CONTEXT_LENGTH:
                                    for j in range(len(candidate_and_context), CANDIDATE_AND_CONTEXT_LENGTH):
                                        candidate_and_context.insert(0, "$PADDING$")
                                        candidate_and_context_vectors.insert(0, PADDING)
                            # No padding, sliding window in normal range
                            else:
                                candidate_and_context = doc[i-CONTEXT_WINDOW:i + CONTEXT_WINDOW + 1]
                                for term in candidate_and_context:
                                    if term in word_embedding.wv.vocab:
                                        candidate_and_context_vectors.append(word_embedding.wv[term])
                                    else:
                                        candidate_and_context_vectors.append(PADDING)


                            a_prob = sess.run(network.aprob, feed_dict={network.query_input: query_vectors,
                                                                                network.candidate_and_context_input: candidate_and_context_vectors})
                            a = 0

                            sampler = random.random()

                            if a_prob > sampler:
                                a = 1
                            else:
                                a = 0


                            actions.append(a)
                            ep_history.append([query_vectors, candidate_and_context_vectors, a])

                    ep_history = np.array(ep_history)



                    for i in range(0, len(actions)):
                        if actions[i] == 1:
                            reformulated_query.append(candidate_terms[i])
                        if DATA_SET == "PRE_TRAIN":
                            if candidate_terms[i] in ground_truth[topicID]:
                                ground_truth_label.append([1])
                            else:
                                ground_truth_label.append([0])



                    reformulated_query = " ".join(reformulated_query)
                    print("reformulated: ", reformulated_query)

                    if epoch > 1000:
                        STARTING_LERNING_RATE = 1e-4
                    if epoch > 2000:
                        STARTING_LERNING_RATE = 1e-5
                    if epoch > 3000:
                        STARTING_LERNING_RATE = 1e-6


                    if DATA_SET == "PRE_TRAIN":
                        ground_truth_label = np.array(ground_truth_label)

                        policy_loss, _  = sess.run(
                            [network.policy_loss, network.train_policy_network],
                            feed_dict={network.learning_rate: STARTING_LERNING_RATE,
                                       network.query_input: np.vstack(ep_history[:, 0]),
                                       network.candidate_and_context_input: np.vstack(ep_history[:, 1]),
                                       network.true_label:np.vstack(ground_truth_label)
                                       })
                        # TODO: Sum up loss and report on it
                        total_epoch_loss.append(policy_loss)

                        if epoch % 5 == 0:
                            info = [
                                ["epoch: ", epoch],
                                ["topicID: ", topicID],
                                ["loss: ", policy_loss],
                                ["Reformulated Query: ", reformulated_query]
                            ]

                            write_to_file(SL_reformulated, info)

                    else:

                        if topicID not in result_cache.keys():
                            result_cache[topicID] = {}

                        if reformulated_query not in result_cache[topicID].keys():
                            reward = atire.lookup(int(topicID), reformulated_query)
                            result_cache[topicID][reformulated_query] = reward
                        else:
                            reward = result_cache[topicID][reformulated_query]



                        value_loss, _ = sess.run([network.value_loss, network.train_value_network],
                                                 feed_dict={network.learning_rate: STARTING_LERNING_RATE,
                                                            network.query_input: np.vstack(ep_history[:, 0]),
                                                            network.candidate_and_context_input: np.vstack(
                                                                ep_history[:, 1]),
                                                            network.reward: reward[0]
                                                            })

                        # If we freeze policy, we'll let value catch up for 10 epochs by itself first.
                        if FREEZE_POLICY:
                            if epoch > 10:
                                policy_loss, _, predicted_reward = sess.run([network.policy_loss,network.train_policy_network,
                                                                             network.predicted_reward],
                                                                            feed_dict={network.learning_rate:STARTING_LERNING_RATE,
                                                                                                                    network.query_input:np.vstack(ep_history[:,0]),
                                network.candidate_and_context_input:np.vstack(ep_history[:,1]),
                                network.reward:reward[0],
                                network.action_choice:np.vstack(ep_history[:,2])
                            })
                        else:
                            policy_loss, _, predicted_reward = sess.run(
                                [network.policy_loss, network.train_policy_network,
                                 network.predicted_reward],
                                feed_dict={network.learning_rate: STARTING_LERNING_RATE,
                                           network.query_input: np.vstack(ep_history[:, 0]),
                                           network.candidate_and_context_input: np.vstack(ep_history[:, 1]),
                                           network.reward: reward[0],
                                           network.action_choice: np.vstack(ep_history[:, 2])
                                           })


                        print("reward: " + str(reward))
                        print("predicted reward: " + str(predicted_reward))
                        print("policy loss:{}, value loss {} ".format(abs(policy_loss),value_loss))

                        average_precisions.append(reward)
                        average_policy_loss.append(abs(policy_loss))
                        average_value_loss.append(abs(value_loss[0]))



                        if episode % 50 == 0:
                            info = [
                                ["epoch: ", epoch],
                                ["episode: ", episode],
                                ["topicID: ", topicID],
                                ["predicted reward: ", predicted_reward],
                                ["average precision@40: ", reward]
                            ]
                            write_to_file(training_records, info)




            # mean_average_precision = np.mean(average_precisions)
            # print("Mean average precision: ", mean_average_precision)
            if DATA_SET == "PRE_TRAIN":
                print("Mean Epoch Loss: ", np.mean(total_epoch_loss))

                info = [
                    ["epoch: ", epoch],
                    ["Mean Epoch Loss: ", np.mean(total_epoch_loss)]
                ]

                write_to_file(training_records, info)




            save_path = saver.save(sess, weight_saving_path)
            print("Model saved in path: %s" % save_path)
            pkl.dump(result_cache,
                     open(reformulated_cache, "wb+"))

            ########################################################################################################################

            if epoch % TEST_FREQUENCY == 0 and DATA_SET != "PRE_TRAIN":

                average_precisions = np.array(average_precisions)
                average_policy_loss = np.array(average_policy_loss)
                average_value_loss = np.array(average_value_loss)

                info = [
                    ["epoch: ", epoch],
                    ["mean MAP@40: ", np.mean(average_precisions)],
                    ["average policy loss: ", np.mean(average_policy_loss)],
                    ["average value loss: ", np.mean(average_value_loss)]
                ]
                write_to_file(average_training_MAP, info)

                print("validating on unseen data...")
                average_precisions = []

                # testing_batch = extract_minibatch(test_query_table,TEST_BATCH_SIZE)
                testing_batch = test_query_table

                for topicID in testing_batch:
                    current_query_term_list = testing_batch[topicID]
                    # This is the input to the left half of the neural network
                    current_query, query_vectors = lookup_term_vectors(current_query_term_list)

                    if "".join(current_query) not in token_cache.keys():
                        terms_in_results = retrieve_document_terms(" ".join(current_query))
                        terms_in_query = current_query[0].split(" ")
                        terms_in_results.insert(0, terms_in_query)
                        token_cache["".join(current_query)] = tuple(terms_in_results)
                    else:
                        terms_in_results = token_cache["".join(current_query)]


                    actions = []
                    reformulated_query = []
                    candidate_terms = []
                    document_selected = []
                    # First append the original query
                    document_selected.append(terms_in_results[0])
                    
                    # Randomly sample from the other top documents
                    document_selected.append(terms_in_results[random.randint(1, len(terms_in_results) - 1)][:K_TERMS])

                    # terms_in_results = [['60','50','sanction','bus','europe','50','bus','china','bus']]

                    # Each document in the top 10 results list
                    for doc in document_selected:
                        # For each term in one of the documents
                        for i in range(0, len(doc)):
                            candidate_and_context = []
                            candidate_term = (doc[i])
                            candidate_terms.append(candidate_term)

                            # This represents the state
                            candidate_and_context_vectors = []

                            # pad on the left
                            if i < CONTEXT_WINDOW:
                                diff = CONTEXT_WINDOW - i
                                candidate_and_context = doc[0:i + CONTEXT_WINDOW + 1]
                                for term in candidate_and_context:
                                    if term in word_embedding.wv.vocab:
                                        candidate_and_context_vectors.append(word_embedding.wv[term])
                                    else:
                                        candidate_and_context_vectors.append(PADDING)


                                for j in range(0, diff):
                                    candidate_and_context.insert(0, "$PADDING$")
                                    candidate_and_context_vectors.insert(0, PADDING)

                                if (len(candidate_and_context)) < CANDIDATE_AND_CONTEXT_LENGTH:
                                    for j in range(len(candidate_and_context), CANDIDATE_AND_CONTEXT_LENGTH):
                                        candidate_and_context.append("$PADDING$")
                                        candidate_and_context_vectors.append(PADDING)

                            # pad on the right---
                            elif (len(doc) - (i + 1)) < CONTEXT_WINDOW:
                                # TODO: A known issue here: '' seperation at the end of each document is counted as a valid term, this may require fixing if it causes a problem in query reformulation.

                                diff = CONTEXT_WINDOW - (len(doc) - (i + 1))
                                candidate_and_context = doc[i - CONTEXT_WINDOW:len(doc)]
                                for term in candidate_and_context:
                                    if term in word_embedding.wv.vocab:
                                        candidate_and_context_vectors.append(word_embedding.wv[term])
                                    else:
                                        candidate_and_context_vectors.append(PADDING)

                                for j in range(0, diff):
                                    candidate_and_context.append("$PADDING$")
                                    candidate_and_context_vectors.append(PADDING)

                                if (len(candidate_and_context)) < CANDIDATE_AND_CONTEXT_LENGTH:
                                    for j in range(len(candidate_and_context), CANDIDATE_AND_CONTEXT_LENGTH):
                                        candidate_and_context.insert(0, "$PADDING$")
                                        candidate_and_context_vectors.insert(0, PADDING)
                            # No padding, sliding window in normal range
                            else:
                                candidate_and_context = doc[i - CONTEXT_WINDOW:i + CONTEXT_WINDOW + 1]
                                for term in candidate_and_context:
                                    if term in word_embedding.wv.vocab:
                                        candidate_and_context_vectors.append(word_embedding.wv[term])
                                    else:
                                        candidate_and_context_vectors.append(PADDING)

                            a_prob = sess.run(network.aprob, feed_dict={network.query_input: query_vectors,
                                                                        network.candidate_and_context_input: candidate_and_context_vectors})

                            if a_prob > 0.5:
                                a = 1
                            else:
                                a = 0

                            actions.append(a)

                    for i in range(0, len(actions)):
                        if actions[i] == 1:
                            reformulated_query.append(candidate_terms[i])

                    reformulated_query = " ".join(reformulated_query)
                    print("reformulated query at test time: ", reformulated_query)


                    reward = atire.lookup(int(topicID), reformulated_query)
                    print("average precision for test time reformulated query: ", reward)

                    info = [
                        ["epoch: ", epoch],
                        ["Topic: ", topicID],
                        ["reward :", reward],
                        ["reformulated query ", reformulated_query],
                        [" ", " "]
                    ]

                    write_to_file(reformulated_query_filepath, info)
                    average_precisions.append(reward)

                average_precisions = np.array(average_precisions)
                mean_average_precision = np.mean(average_precisions)
                print("Mean average precision: ", mean_average_precision)


                info = [
                    ["epoch: ", epoch],
                    ["MAP@40: ", mean_average_precision],
                ]
                write_to_file(test_time_MAP, info)










