import pandas as pd
# df = pd.read_csv('Lexicons of bias - Gender stereotypes.csv')
import gensim
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

# better to store on disk and load when needed since this takes long to decompress
# model1 = KeyedVectors.load_word2vec_format('/h/brandon/internship/Uncover_implicit_bias/data/GoogleNews-vectors-negative300.bin.gz',binary=True)
model1 = KeyedVectors.load('/h/brandon/internship/Uncover_implicit_bias/data/google_word_vectors.kv', mmap='r')


from empath import Empath
lexicon = Empath()

intellect = ["intellectual", "intuitive", "imaginative", "knowledgeable", "ambitious", "intelligent", "opinionated", "admirable", "eccentric", "crude", "likable", "empathetic", "superficial", "tolerant", "resourceful", "uneducated", "academically", "studious", "temperamental", "exceptional", "cynical", "outspoken", "destructive", "dependable", "amiable", "impulsive", "frivolous", "insightful", "overconfident", "charismatic", "prideful", "influential", "likeable", "unconventional", "educated", "flawed", "articulate", "pretentious", "perceptive", "vulgar", "easygoing", "listener", "skillful", "assertive", "philosophical", "rebellious", "selfless", "cunning", "deceptive", "artistic", "appalling", "overbearing", "temperament", "diligent", "charitable", "disposition", "quirky", "strategic", "compulsive", "benevolent", "pessimistic", "scientific", "flamboyant", "obsessive", "selective", "oriented", "humorous", "narcissistic", "reliable", "headstrong", "manipulative", "practical", "rewarding", "refined", "resilient", "desirable", "spiritual", "tendencies", "pompous", "judgmental", "respected", "inexperienced", "compassionate", "promiscuous", "argumentative", "conventional", "intellectually", "expressive", "impractical", "observant", "fickle", "hyperactive", "immoral", "straightforward", "vindictive"]

import numpy as np
import math
# def get_words(cates):
#     tmp = df.loc[:,cates].values
#     words = []
#     for i in tmp:
#         for j in i:

#             if(type(j) is not float):
#                 words.append(j)
#     return set(words)

def get_words(category, seeds, size=100):
    lexicons = lexicon.create_category(category,seeds, size=size)
    return lexicons
    

# we get lexicons for these categories
# using empath may result in biased lexicons -- i.e. feminine for appearance
# do further filtering for masculine/feminine terms?
appearence = ["beautiful","sexual"]
# filtered out terms like girly, feminine, masculine -- should be ideally gender neutral
appear = ['beautiful', 'sexual', 'sexy', 'perfect', 'attractive', 'romantic', 'irresistible', 'unique', 'sweet', 'gorgeous', 'pretty', 'touching', 'pleasing', 'amazing', 'provocative', 'enchanting', 'meaningful', 'sinful', 'intimate', 'poetic', 'desirable', 'flirtatious', 'flattering', 'unattractive', 'innocent', 'cute', 'inappropriate', 'flawless', 'sensitive', 'modest', 'captivating', 'enticing', 'daring', 'incredible', 'creative', 'vulgar', 'alluring', 'weird', 'spontaneous', 'mysterious', 'wonderful', 'cliché', 'addictive', 'dreamy', 'erotic', 'charming', 'cheesy', 'intriguing', 'repulsive', 'adorable', 'extraordinary', 'freaky', 'likeable', 'exotic', 'expressive', 'unbelievable', 'realistic', 'sappy', 'disturbing', 'distracting', 'inspirational', 'shocking', 'outrageous', 'freaky', 'endearing', 'perverted', 'flirty', 'undeniably', 'predictable', 'shy', 'boyish', 'stereotypical', 'good', 'imaginative', 'different', 'inspiring', 'special', 'exquisite', 'quirky', 'fabulous', 'classy', 'real', 'straightforward', 'bold', 'perverted', 'subtle', 'sophisticated', 'freaky', 'lovable', 'heartwarming', 'artsy', 'just_sex', 'insightful']

power = ["dominant","strong"]
power =  ['strong', 'dominant', 'powerful', 'resilient', 'fierce', 'fearless', 'weak', 'ruthless', 'brave', 'courageous', 'tough', 'stronger', 'submissive', 'vicious', 'independent', 'invincible', 'vulnerable', 'aggressive', 'fragile', 'weakest', 'dangerous', 'possessive', 'lethal', 'stubborn', 'gentle', 'unstoppable', 'formidable', 'confident', 'flexible', 'firm', 'sensitive', 'destructive', 'dominate', 'brutal', 'violent', 'strongest', 'inexperienced', 'human', 'persuasive', 'agile', 'resistant', 'fighter', 'Carpathian', 'overpower', 'potent', 'headstrong', 'experienced', 'deadly', 'skilled', 'selfless', 'delicate', 'ferocious', 'primitive', 'determined', 'daring', 'good_fighter', 'masculine', 'cunning', 'savage', 'rational', 'forceful', 'persistent', 'warrior', 'wise', 'ambitious', 'demanding', 'yet', 'stong', 'feeble', 'hard', 'unpredictable', 'loyal', 'territorial', 'controlled', 'compelling', 'passive', 'superior', 'dependable', 'unbreakable', 'admirable', 'overbearing', 'desirable', 'powerless', 'defiant', 'humble', 'humane', 'fit', 'bloodthirsty', 'intelligent', 'empowered', 'volatile', 'obedient', 'so_much_power', 'compassionate', 'untouchable', 'cruel', 'skillful', 'frail', 'resourceful', 'compulsion']

weak = ['submissive','weak','dependent','afraid']
weak = ['weak', 'afraid', 'vulnerable', 'submissive', 'strong', 'powerless', 'dependent', 'defenseless', 'invincible', 'helpless', 'foolish', 'fragile', 'ruthless', 'scared', 'defenceless', 'dominant', 'Yet', 'inexperienced', 'reckless', 'fear', 'dangerous', 'unfit', 'human', 'physically', 'ashamed', 'shameful', 'rational', 'desperate', 'powerful', 'certain', 'fearful', 'fearless', 'selfish', 'unstable', 'brave', 'greedy', 'unhappy', 'independent', 'crippled', 'dependant', 'immune', 'heartless', 'cruel', 'stubborn', 'violent', 'unworthy', 'inferior', 'nuisance', 'useless', 'yet', 'confident', 'Because', 'Though', 'frightened', 'willing', 'attached', 'feared', 'destructive', 'angry', 'hurt', 'wounded', 'mortal', 'weakest', 'careless', 'hopeless', 'irrational', 'cowardly', 'feeble', 'prone', 'harmful', 'susceptible', 'sensitive', 'unwanted', 'obedient', 'tolerant', 'own_person', 'shamed', 'flawed', 'cautious', 'unreasonable', 'aggressive', 'critical', 'mindset', 'desirable', 'determined', 'truly', 'rebellious', 'weakness', 'own_will', 'Yet', 'resilient', 'vicious', 'superior', 'pathetic', 'extreme', 'painful', 'brutal']

print("loaded")

from scipy import stats
import pickle
from scipy import spatial
import pickle


file_list = []

file_list.append("male_masked_subj.txt_at_dict")
file_list.append("female_masked_subj.txt_at_dict")

file_list.append("male_masked_subj.txt_xr_dict")
file_list.append("female_masked_subj.txt_xr_dict")

file_list.append("male_two_and_above_obj.txt_or_dict")
file_list.append("female_two_and_above_obj.txt_or_dict")

file_list.append("male_two_and_above_subj.txt_or_dict")
file_list.append("female_two_and_above_subj.txt_or_dict")


def load_file(file):
    file = open(file, 'rb')

    words = pickle.load(file)

    file.close()
    return words

# getting word vectors for the lexicons
# might need to remove the .wv
weak_vecs = [model1[i] for i in weak if i in model1]
power_vecs = [model1[i] for i in power if i in model1]
appear_vecs = [model1[i] for i in appear if i in model1]
intellect_vecs = [model1[i] for i in intellect if i in model1]


def calculateSubspace(A, B):
    """ 
    computes (2) in the paper -- the power semantic axis
    avg strong embedding subtracted by avg weak embedding
    Subspace in this case = a vector in the direction of power
    """
    A_vecs = [model1.wv[i] for i in A if i in model1]
    B_vecs = [model1.wv[i] for i in B if i in model1]

    suma = A_vecs[0].copy()

    for i in range(1, len(A_vecs)):
        suma += A_vecs[i]
    sumb = B_vecs[0].copy()
    for i in range(1, len(B_vecs)):
        suma += B_vecs[i]
    return suma / len(A) - sumb / len(B)


def compute(words_clusters, title):
    """ word_clusters = Comet inferences for a given gender
        title = save file 

        computes (1) in the paper -- association score between lexicon (i.e. intellect) and protagonist inferences (i.e. xAttr, xReact, ...)
    """
    # sum of intel scores between L and x_i
    intel_sum = []
    appear_sum = []
    power_sum = []
    
    power_subspace = calculateSubspace(power, weak)

    for x in words_clusters:
        if x not in model1:
            continue
        # weak
        intel_sims = 0
        appear_sims = 0
        # computes (1) -- pointwise between L (set of lexicons) and x (a single word)
        for j in intellect:
            if (j in model1):
                intel_sims += model1.similarity(x, j)
        for k in appear:
            if (k in model1):
                appear_sims += model1.similarity(x, k)

        power_sum.append(1 - spatial.distance.cosine(model1.wv[x], \
                                                     power_subspace))
        intel_sum.append(intel_sims / len(intellect))

        appear_sum.append(appear_sims / len(appear))
    print("dumping")
    f = open(title + "_intellect.pkl", "wb")
    pickle.dump(intel_sum, f)
    f.close()
    f = open(title + "_appear.pkl", "wb")
    pickle.dump(appear_sum, f)
    f.close()
    f = open(title + "_power.pkl", "wb")
    pickle.dump(power_sum, f)
    f.close()
    return [np.median(stats.zscore(intel_sum)), np.median(stats.zscore(appear_sum)), np.median(stats.zscore(power_sum))]


def get_stats(l):
    return (np.max(l), np.min(l), np.percentile(l, 25), np.percentile(l, 75), \
            np.median(l))


def getLexiconScore(f1, f2):
    male = load_file(f1)
    female = load_file(f2)
    if (type(female[0]) is list):
        # picking 1st beam across all inferences for diff stories. collapsing the lists.
        male = [i[0] for i in male]
        female = [i[0] for i in female]
    if (type(female[0]) is tuple):
        # not sure that this means
        male = [i[1] for i in male]
        female = [i[1] for i in female]

    # remove inferences with 0 occurrences
    male_removed = Counter(male).most_common(5)
    female_removed = Counter(female).most_common(5)

    # what does this collect?
    male_removed = [i[0] for i in male_removed]
    female_removed = [i[0] for i in female_removed]

    m = []
    for i in male:
        if i in male_removed:
            continue
        m.append(i)
    f = []
    for i in female:
        if i in female_removed:
            continue
        f.append(i)

    if "curious" in m:
        print("wrong")
    else:
        print("OK")

    m = list(filter(lambda a: a != 'none', m))
    f = list(filter(lambda a: a != 'none', f))

    # set of inferences is filtered before running inference
    # should 2nd param be female?
    return compute(m, "male"), compute(f, "female")


from collections import Counter


def getLexiconScore_b5(f1, f2):
    male = load_file(f1)
    female = load_file(f2)

    mm = []
    # iterating through each male inference (beam 5)?
    for i in male:
        mm += set(i)
    ff = []

    for i in female:
        ff += set(i)

    m = list(filter(lambda a: a != 'none', mm))
    f = list(filter(lambda a: a != 'none', ff))

    return compute(m, "../0515replotting/" + f1.split("/")[1] + "/" + f1.split("/")[3]) \
        , compute(f, "../0515replotting/" + f2.split("/")[1] + "/" + f2.split("/")[3])

# comparing object for generated story vs. human written story
# ex. different xIntent b/w the two
directs = ["../Generated/Beam_5/", "../humanWritten/Beam_5/"]

# stores male/female z-scores for each dimension (intelligence, power, ...)
# result[i][0] = male and result[i][1] = female. i = model vs. human generated story 
result = []
for direct in directs:
    if "Beam" in direct:
        print(direct)
        print("subj_at")
        result.append(
            getLexiconScore_b5(direct + "male_masked_subj.txt_at_dict", direct + "female_masked_subj.txt_at_dict"))

    else:
        print(direct)
        print("subj_at")
        result.append(
            getLexiconScore(direct + "male_masked_subj.txt_at_dict", direct + "female_masked_subj.txt_at_dict"))

print(result)