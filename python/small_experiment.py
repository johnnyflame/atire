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

"""
"First system"

A prototype of the model described in the paper "Task-Oriented Query Reformulation with Reinforcement Learning", 
Nogueira and Cho 2017


Author: Johnny Flame Lee 

"""
DUMMYMODE = False

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
rel_path = "experiment.txt"
fitness_record = os.path.join(script_dir, rel_path)
running_reward_filepath = os.path.join(script_dir, rel_path)



CREATE_DICTIONARY = True
#TOPIC_FILE = "/Users/johnnyflame/atire/evaluation/topics.51-100.txt"

TOPIC_FILE = "/Users/johnnyflame/atire/evaluation/topics.51-61"
ASSESSMENT_FILE = "/Users/johnnyflame/atire/evaluation/WSJ.qrels"

CONTEXT_WINDOW = 4

# Total number of terms to go into the second network
CANDIDATE_AND_CONTEXT_LENGTH = CONTEXT_WINDOW * 2 + 1


WORD_VECTOR_DIMENSIONS = 300

# maximum number of terms in q0
MAX_SEQUENCE_LENGTH = 15

WORD_EMBEDDING_PATH = "wsj-collection-vectors"

METRIC = " -mMAP@40"


annealing_steps = 10000.
start_eps = 1.0
end_eps = 0.1
eps = start_eps
stepDrop = (start_eps - end_eps) / annealing_steps





PADDING = np.zeros(WORD_VECTOR_DIMENSIONS)

random.seed(500)



# HYPERPARAMETERS:
atire.init("atire -a " + ASSESSMENT_FILE + METRIC)


def write_to_file(filename, v):
    """

    :param filename:
    :param v: a list of training information
    :return:
    """
    with open(filename, "a") as f:
        message = "episode: " + str(v[0]) + ' reward: ' + str(v[1]) + "\n"
        f.write(message)
    f.closed
    print("Wrote record to file")



def load_lookup_table(file):
    """Return a dictionary of term-vector pairs"""
    return pkl.load(open(file, "rb"))

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

gamma = 0.99

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r




class GenerateNetwork:

    def __init__(self,number_of_terms):
        self.query_input = tf.placeholder(tf.float32, [None, WORD_VECTOR_DIMENSIONS], name="query_input")
        self.candidate_and_context_input = tf.placeholder(tf.float32, [None, WORD_VECTOR_DIMENSIONS]
                                                          , name="candidate_vectors")

        self.action_choice = tf.placeholder(tf.int32,[None,1],name="actions")

        # Reshaping the query so it becomes Rank 4, the order is [batch_size, width,height, channel]
        self.reshaped_query_input = tf.reshape(self.query_input, [-1, number_of_terms, WORD_VECTOR_DIMENSIONS, 1])
        self.reshaped_candidate_and_context = tf.reshape(self.candidate_and_context_input,
                                                         [-1, CANDIDATE_AND_CONTEXT_LENGTH, WORD_VECTOR_DIMENSIONS, 1])

        # Add 2 convolutional layers with ReLu activation
        with tf.variable_scope("policy", reuse=tf.AUTO_REUSE):

            self.query_conv1 = slim.conv2d(
                self.reshaped_query_input, num_outputs=256,
                kernel_size=[3,WORD_VECTOR_DIMENSIONS], stride=[1,1], padding='VALID', biases_initializer=tf.constant_initializer(0.1)
            )

            # Second convolution layer
            self.query_conv2 = slim.conv2d(
                self.query_conv1, num_outputs=256,
                kernel_size=[3,1], stride=[1,1], padding='VALID', biases_initializer=tf.constant_initializer(0.1)
            )

            # Not super confident about these parameters, may need revisit
            self.query_pooled = tf.nn.max_pool(
                self.query_conv2,
                ksize=[1,11,1,1],
                strides=[1, 1, 1,1],
                padding='VALID',
                name="pool")

            self.candidates_conv1 =  slim.conv2d(
                self.reshaped_candidate_and_context, num_outputs=256,
                kernel_size=[5,WORD_VECTOR_DIMENSIONS], stride=[1,1], padding='VALID', biases_initializer=tf.constant_initializer(0.1)
            )

            self.candidates_conv2 =  slim.conv2d(
                self.candidates_conv1, num_outputs=256,
                kernel_size=[3,1], stride=[1,1], padding='VALID', biases_initializer=tf.constant_initializer(0.1)
            )

            self.candidates_pooled = tf.nn.max_pool(
                self.candidates_conv2,
                ksize=[1, 3, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")

            self.pooled_vectors_concatenated = tf.concat([self.query_pooled, self.candidates_pooled], 3)

            self.policy_fc1 = tf.contrib.layers.fully_connected(tf.reshape(self.pooled_vectors_concatenated,[-1,512]),
                                                                num_outputs=256,activation_fn=tf.nn.tanh,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           biases_initializer=tf.constant_initializer(0.001))

            self.aprob = slim.fully_connected(self.policy_fc1,1,biases_initializer=tf.constant_initializer(0.001),activation_fn=tf.nn.sigmoid)





        self.mean_context_vector = tf.reduce_mean(self.candidates_pooled,axis=0)
        self.mean_context_and_query_concatenated = tf.concat((tf.reduce_mean(self.query_pooled,axis=0),self.mean_context_vector),axis=2)


        with tf.variable_scope("value",reuse=tf.AUTO_REUSE):

            self.value_fc1 = tf.contrib.layers.fully_connected(tf.reshape(self.mean_context_and_query_concatenated,[1,512]),
                                                                num_outputs=256,activation_fn=tf.nn.tanh,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           biases_initializer=tf.constant_initializer(0.01))

            self.value_prediction = tf.contrib.layers.fully_connected(self.value_fc1,num_outputs=1,activation_fn=tf.nn.sigmoid,
                                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                                        biases_initializer=tf.constant_initializer(0.01))

        self.value = tf.squeeze(self.value_prediction)

        # Using the same training parameters from the paper
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-6,beta1=0.9,beta2=0.999,epsilon=1e-8)

        # Update the parameters according to the computed gradient.
        # train_step = optimizer.minimize(loss)


        # self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        self.reward = tf.placeholder(shape=[],dtype=tf.float32)
        self.predicted_reward = tf.placeholder(shape=[],dtype=tf.float32)




        self.policy_loss = (self.reward - self.predicted_reward) * -tf.reduce_sum(tf.to_float(self.action_choice) * tf.log(self.aprob) +
                                                                                  (1.0-tf.to_float(self.action_choice))*tf.log(1.0-self.aprob))

        # self.policy_loss = self.loss((self.reward-self.predicted_reward))


        vars = tf.trainable_variables()
        self.value_fc_variables = [v for v in vars if v.name.startswith("value")]
        self.policy_weights = [v for v in vars if v.name.startswith("policy")]


        self.value_loss = 0.01 * tf.square((self.reward - self.value_prediction))


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
                print (word)
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



    querys_table = {}
    read_topic_file(TOPIC_FILE,querys_table)

    network = GenerateNetwork(number_of_terms=MAX_SEQUENCE_LENGTH)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)



        for episode in range(0,500000):
            for topicID in querys_table:


                current_query = querys_table[topicID]
                # current_query = "South African"
                reformulated_query = []

                ep_history = []
                candidate_terms = []
                policy_training_history = []
                aprob_list = []
                actions = []

                # This is the input to the left half of the neural network
                current_query, query_vectors = lookup_term_vectors(current_query)

                #queries_vectors = lookup_term_vectors("gobbody-gook dah baqfuafk vnvi")


                # Get a list of all the words in the top 10 documents
                terms_in_results = retrieve_document_terms(" ".join(current_query))
                terms_in_query = current_query[0].split(" ")

                terms_in_results.insert(0,terms_in_query)


                terms_in_results = terms_in_results[:2]

                # terms_in_results = [['60','50','sanction','bus','europe','50','bus','china','bus']]

                # Each document in the top 10 results list
                for doc in terms_in_results:
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
                                candidate_and_context_vectors.append(word_embedding.wv[term])


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
                                candidate_and_context_vectors.append(word_embedding.wv[term])

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
                                candidate_and_context_vectors.append(word_embedding.wv[term])


                        # a = randint(0,1)


                        # TODO: might need to add randomness here, or weights may not get updated at all.

                        if eps > end_eps:
                            eps -= stepDrop

                        r = random.random()
                        a_prob = None


                        if r < eps:
                            a_prob = randint(0,1)
                        else:
                            a_prob = sess.run(network.aprob, feed_dict={network.query_input: query_vectors,
                                                                            network.candidate_and_context_input: candidate_and_context_vectors})
                        aprob_list.append(a_prob)

                        a = 0

                        if a_prob > 0.5:
                            action_taken = True
                            a = 1
                            policy_training_history.append([query_vectors, candidate_and_context_vectors, a])
                        else:
                            a = 0


                        actions.append(a)


                        ep_history.append([query_vectors, candidate_and_context_vectors, a])

                ep_history = np.array(ep_history)
                # policy_training_history = np.array(policy_training_history)



                for i in range(0, len(actions)):
                    if actions[i] == 1:
                        reformulated_query.append(candidate_terms[i])


                reformulated_query = " ".join(reformulated_query)
                print("reformulated: ", reformulated_query)
                reward = atire.lookup(int(topicID), reformulated_query)

                predicted_reward = sess.run(network.value_prediction,feed_dict={network.query_input:np.vstack(ep_history[:,0]),
                                                                   network.candidate_and_context_input: np.vstack(
                                                                       ep_history[:, 1])})

                value_loss, _ = sess.run([network.value_loss,network.train_value_network],feed_dict={network.query_input:np.vstack(ep_history[:,0]),
                    network.candidate_and_context_input:np.vstack(ep_history[:,1]),
                    network.reward:reward[0], network.predicted_reward : np.squeeze(predicted_reward)
                })




                print("reward: " + str(reward))
                print("predicted reward: " + str(predicted_reward))

                policy_loss, _ = sess.run([network.policy_loss,network.train_policy_network],feed_dict={network.query_input:np.vstack(ep_history[:,0]),
                    network.candidate_and_context_input:np.vstack(ep_history[:,1]),
                    network.reward:reward[0], network.predicted_reward : np.squeeze(predicted_reward),
                    network.action_choice:np.vstack(ep_history[:,2])
                })

                print("policy loss:{}, value loss {} ".format(policy_loss,value_loss))





                if episode % 50 == 0:
                    write_to_file(running_reward_filepath, [episode,reward])






                """
                At this point we are ready to evaluate the reformulated query and retrieve a reward from ATIRE.
                
                Things we need here:
                
                1. A list of words representing the reformulated query
                2. Decide on a reward metric: precision@k or recall@k
                3. The original queryID
                4. Path to the evaluation file 
                
                """



