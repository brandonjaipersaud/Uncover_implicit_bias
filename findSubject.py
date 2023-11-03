
from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz",cuda_device = 0)
import nltk
import spacy
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

nlp = spacy.load('en_core_web_lg')

male = []
female = []
unresolved = []


def writeFile(file_name, content):
    a = open(file_name, 'a')
    a.write(content)
    a.close()


def getPron(para):
    """
    Given a paragraph, returns a list of coreferent pronouns in the paragraph.
    
    Args:
    para: str, the paragraph to search for coreferent pronouns
    
    Returns:
    list: a list of coreferent pronouns in the paragraph
    """

    output = predictor.predict(
        document=para
    )

    for i in output['clusters']:
        for j in i:
            print(output['document'][j[0]:j[1] + 1])
        print('\n')

    return output['clusters']


def findSubj(li):
    """
    Given a list of token dependencies, returns True if the root of the sentence has a subject that is 
    "protagonistA" or "ProtagonistA", and False otherwise.
    
    Args:
    li: list of tuples, where each tuple contains information about a token's dependency
    
    Returns:
    bool: True if the root of the sentence has a subject that is "protagonistA" or "ProtagonistA", and False otherwise.
    """
    root = ""
    for i in li:
        if (i[1] == "ROOT"):
            root = i[0]
    for i in li:
        if (i[1] == "nsubj" and i[2] == root and (i[0] == "protagonistA" \
                                                  or i[0] == "ProtagonistA")):
            return True
    return False


def process(inputf, outputf):
    """
    Processes the input file to extract and write sentences without a subject into a new output file.
    
    The function reads an input file line by line, tokenizes each sentence, tags it with POS tags,
    and uses dependency parsing to check for the presence of a subject. Sentences without a subject 
    are written into a specified output file.
    
    Parameters:
    - inputf (str): Path to the input file that contains text to be processed.
    - outputf (str): Base path for the output file where results will be written.
    
    The output file is appended with '_obj.txt' to denote the collection of object-centric sentences.
    """

    f = open(inputf, "r")
    for p in tqdm(f.readlines()):
        #         p = "i like hiking."
        p.replace("!", ".")
        p = p.strip("\n").split(".")
        p = [i + "\n" for i in p if i != '']

        for para in p:
            if (para == "\n" or para == ".\n"):
                continue
            tokens = nltk.word_tokenize(para)
            tagged_sent = nltk.pos_tag(tokens)
            doc = nlp(para)


            token_dependencies = [(token.text, token.dep_, token.head.text) for token in doc]

            flag = findSubj(token_dependencies)

            if (flag == True):
                if(len(para)>5):
                    # write sentences where ProtagonistA is the subject to the file. Useful for xAttr ...
                    writeFile(outputf + '_subj.txt', para)
                # continue
            else:
                # write sentences where ProtagonistB is the subject to the file. Useful for oAttr ...
                if(len(para)>5):
                    writeFile(outputf + '_obj.txt', para)

#process("male_two_and_above.txt","male_two_and_above")
process("female_two_and_above.txt","female_two_and_above")