import gensim
import atire


datasets = {"MSA":480721, "WSJ":173251}


INDEX_FLAG = ""

for name in datasets:

    if name == "MSA":
        params = "atire " + INDEX_FLAG

    elif name == "WSJ":
        INDEX_FLAG = "-findex ./index_WSJ.aspt"
        params = "atire " + INDEX_FLAG

    input = []
    COLLECTION_SIZE = datasets[name]

    atire.init(params)


    for i in range(0, COLLECTION_SIZE):
        input.append(atire.get_ordered_tokens(i))

        model = gensim.models.Word2Vec(sentences=input, size=300,window=5,workers=64,min_count=1,iter=30)
        model.save(name + '-collection-vectors')

    atire.cleanup()
    # # TODO: Re-train Word2Vec model, it's not getting numbers in at the moment.Do it Friday night. Train for 30 iterations
    # print(atire.get_ordered_tokens(63615))
    #
    #
    # print ("done")
    #
    #
    # loaded_models = gensim.models.Word2Vec.load("wsj-collection-vectors")
    # #
    # #
    # print (loaded_models.most_similar('money'))