import numpy as np
import pandas as pd
import sklearn
import deap
import argparse
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
import random
import time

# Reading the training set, the validation set, and the test set of TwiBot-20 with labels from my file system.
# Each of the aforementioned set is properly formatted as a result of the execution of format_preprocess.py.
# The only difference is that each record has an extra attribute called "label" (0 human, 1 bot).
# However, instead of executing scale_transform.py on each of them, the scaling is directly performed inside this script in order to show how the file scaler.pkl was generated. 
trainDataPreProcessed = pd.read_json("trainPreProcessed.json",orient="records",typ="frame",dtype=True,convert_dates=False,precise_float=False,lines=False,encoding="utf-8",encoding_errors="replace")    
devDataPreProcessed = pd.read_json("devPreProcessed.json",orient="records",typ="frame",dtype=True,convert_dates=False,precise_float=False,lines=False,encoding="utf-8",encoding_errors="replace")
testDataPreProcessed = pd.read_json("testPreProcessed.json",orient="records",typ="frame",dtype=True,convert_dates=False,precise_float=False,lines=False,encoding="utf-8",encoding_errors="replace")

trainDataPreProcessed = trainDataPreProcessed.set_index("id")
devDataPreProcessed = devDataPreProcessed.set_index("id")
testDataPreProcessed = testDataPreProcessed.set_index("id")

trData = trainDataPreProcessed.select_dtypes(include="number")
teData = testDataPreProcessed.select_dtypes(include="number")
deData = devDataPreProcessed.select_dtypes(include="number")

chosenAttr = ["reputation","listed_growth_rate","favourites_growth_rate","friends_growth_rate",
              "followers_growth_rate","statuses_growth_rate",
              "screenNameLength","frequencyOfWords","frequencyOfHashtags","frequencyOfMentionedUsers",
              "frequencyOfRetweets","frequencyOfURLs",
              "words_raw_count_std","hashtag_raw_count_std",
              "mentioned_users_raw_count_std",
              "tweets_similarity_mean","stdDevTweetLength"
              ]

X_train = trData[chosenAttr].values
y_train = trData["label"].values

X_dev = deData[chosenAttr].values
y_dev = deData["label"].values

X_test = teData[chosenAttr].values
y_test = teData["label"].values

po_mm_scaler = Pipeline([("pow_scaler",PowerTransformer()),("min_max_scaler",MinMaxScaler(feature_range=(0,1)))])

scaler = po_mm_scaler


# Modified version of the algorithms.eaSimple of DEAP library.
# Here we perform a random sampling of the training set adopted for fitness evaluation at each generation.
# Moreover, fitness is always evaluated on the entire population at each generation.
# This means that if an individual in a specific generation has the fitness already evaluated from the previous generation,
# the evaluation is performed again on the new sample of training data.
# Original method: https://github.com/DEAP/deap/blob/master/deap/algorithms.py 
def eaSimpleWithRepeatedFitnessEvaluation(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::
        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring
    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.
    .. note::
        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.
    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.
    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    toolbox.sample()
    # Evaluate the individuals with an invalid fitness
    #invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        toolbox.sample()
        # Evaluate the individuals with an invalid fitness
        #invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
            

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(offspring), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

# Modified version of the gp.cxOnePoint of DEAP library.
# The original method does not allow for reproducibility because iterations on set objects are performed.
# This version (https://github.com/EpistasisLab/tpot/pull/412/files) enables us to write code that can be reproduced by fixing random seeds.
# Original method: https://github.com/DEAP/deap/blob/master/deap/gp.py 
def cxOnePoint(ind1, ind2):
    """Randomly select in each individual and exchange each subtree with the
    point as root between each individual.
    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :returns: A tuple of two trees.
    """
    # Define the name of type for any types.
    __type__ = object
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2
    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    if ind1.root.ret == __type__:
        # Not STGP optimization
        types1[__type__] = range(1, len(ind1))
        types2[__type__] = range(1, len(ind2))
        common_types = [__type__]
    else:
        for idx, node in enumerate(ind1[1:], 1):
            types1[node.ret].append(idx)
        common_types = []
        for idx, node in enumerate(ind2[1:], 1):
            if node.ret in types1 and not node.ret in types2:
                common_types.append(node.ret)
            types2[node.ret].append(idx)

    if len(common_types) > 0:
        type_ = np.random.choice(common_types)

        index1 = np.random.choice(types1[type_])
        index2 = np.random.choice(types2[type_])
        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]
    return ind1, ind2

# Modified version of the gp.mutUniform of DEAP library.
# With this method, it is not possible to mute an individual with a mutation applied to individual root, and thus leading to a total replace of the original individual.
# Original method: https://github.com/DEAP/deap/blob/master/deap/gp.py 
def mutUniformButNoRoot(individual, expr, pset):
    """Randomly select a point in the tree *individual*, then replace the
    subtree at that point as a root by the expression generated using method
    :func:`expr`.
    :param individual: The tree to be mutated.
    :param expr: A function object that can generate an expression when
                 called.
    :returns: A tuple of one tree.
    """
    index = random.randrange(start=1,stop=len(individual),step=1) # Original method starts from 0, here we start from 1.
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_)
    return individual,

class GAEvaluation:
    def __init__(self,X,y,train_percentage=0.6):
        self.X, self.y = X, y
        self.numOfPoints = self.X.shape[0]
        self.human_indexes = np.where(self.y==0)[0].tolist()
        self.bot_indexes = np.where(self.y==1)[0].tolist()
        self.toSample = int(len(self.human_indexes)*train_percentage)
        self.sample = None
    def sampling(self):
        ind=np.random.choice(len(self.bot_indexes),self.toSample,replace=False)
        ind1=[self.bot_indexes[i] for i in ind]
        ind=np.random.choice(len(self.human_indexes),self.toSample,replace=False)
        ind2=[self.human_indexes[i] for i in ind]
        self.sample = ind1+ind2
    def evaluation(self, individual):
        mse = [np.square(int(np.dot(self.X[i],individual)<=0) - self.y[i]) for i in self.sample]
        return sum(mse)/len(mse),

from deap import base, creator, tools, gp, algorithms
import multiprocessing
from numpy.random import default_rng

X = X_train.copy()
y = y_train.copy()

scaler.fit(X) # this is the scaler "scaler.pkl" adopted in scale_transform.py.

# The following statement makes the scaler to be persistent.
pickling_on = open("scaler.pkl","wb")
pickle.dump(scaler, pickling_on)
pickling_on.close()

X = scaler.transform(X)

gaeval = GAEvaluation(X,y)
    
def createBaseObjects(weights=(-1.0,)):
    creator.create("FitnessMulti", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.FitnessMulti)

def createToolbox(random_generator, eval_class, ind_size=10, tournsize=7):
    toolbox = base.Toolbox()
    
    toolbox.register("random", random_generator.standard_normal)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.random, n=ind_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("sample",eval_class.sampling)
    toolbox.register("evaluate", eval_class.evaluation)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0,sigma=300,indpb=0.15)

    
    return toolbox

def mainGAEASimple(toolbox):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    toolbox.register("map", pool.map)
    pop = toolbox.population(n=100000)
    hof = tools.HallOfFame(100)
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_fit.register("avg", np.mean, axis=0)
    stats_fit.register("std", np.std, axis=0)
    stats_fit.register("min", np.min, axis=0)
    stats_fit.register("max", np.max, axis=0)
    mstats = tools.MultiStatistics(fitness=stats_fit)
    pop, logbook = eaSimpleWithRepeatedFitnessEvaluation(pop, toolbox, cxpb=0.5, mutpb=0.3, ngen=220, stats=mstats, halloffame=hof, verbose=True)
    pool.close()
    return pop, logbook, hof, mstats

parser = argparse.ArgumentParser()
parser.add_argument("-i","--ind",help="Index of the experiment.",type=int)
parser.add_argument("-s","--seed",help="Seed of the experiment.",type=int)
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)
defrng = default_rng(args.seed)
createBaseObjects(weights=(-1.0,))
toolbox=createToolbox(defrng,gaeval,ind_size=X.shape[1],tournsize=7)

def check_validation_ga(best_ind,validation_x,validation_y):
    res = []
    for individual in best_ind:
        mse = [np.square(int(np.dot(validation_x[i],individual)<=0) - validation_y[i]) for i in range(len(validation_y))]
        res.append((individual,sum(mse)/len(mse)))
    res.sort(key=lambda x:x[1],reverse=False)
    return res


if __name__=="__main__":
    start_time = time.time()
    pop, logbook, hof, mstats = mainGAEASimple(toolbox)
    end_time = time.time()

    twibot20GA = {"logbook":logbook,"hof":[hof[i] for i in range(0,len(hof))],
                         "mstats_last_pop":mstats.compile(pop),"executionTimeInHours":(end_time-start_time)*(1/3600)}
    print(twibot20GA["executionTimeInHours"])
    file_name = "twibot20GA"+"_"+str(args.ind)+".pickle"
    pickling_on = open(file_name,"wb")
    pickle.dump(twibot20GA, pickling_on)
    pickling_on.close()
    
    pickle_off = open("scaler.pkl", 'rb')
    scaler_loaded = pickle.load(pickle_off)
    pickle_off.close()

    X_de=scaler_loaded.transform(X_dev)
    X_te=scaler_loaded.transform(X_test)
    
    res = check_validation_ga(hof,X_de,y_dev) # evaluating the hall of fame on the validation set
    print(str(res[0][0])) # the best individual evaluated on validation set is the discovered model
    
    # The model presented in predict.py was obtained by executing the following line:
    # python train.py -i 1 -s 52 

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
