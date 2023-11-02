
This repo contains code for EMNLP 2021 paper: 
**Uncovering Implicit Gender Bias Through Commonsense Inference**


## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* Prerequisites
  ```sh
  pip install -r requirements.txt
  ```
Download RocStory dataset from https://cs.rochester.edu/nlp/rocstories/

Download StanfordNERTagger

### Installation

1. COMeT2

   Install COMeT2 according to https://github.com/vered1986/comet-commonsense

<!-- USAGE EXAMPLES -->
## Usage

1. Classify stories according to protagonsit's gender
      ```sh
      python preprocess.py <story_filename.tsv>
      ```
2. Anonymization
      
      ```sh
      python replaceGender.py 
      ```      
      - taking ~1hr:30mins to run on vector cluster across 2000 lines/stories. 
3. Extract stories having more than two characters

      ```sh
      python extractTwo.py 
      ```  
      - split data into 2 subsets
        - 1 = 1 character (protagonist)
        - 2 = 2 or more characters
        - for the 2nd subset, use comet to make inferences that require 2 people. Ex. how protagonist's actions affects mental state of others.
   
4. Classify sentences according to protagonist
      ```sh
      python findSubj.py 
      ```  
   
5. Get COMeT outputs

      ```sh
      python generate_inferences.py
      ```  
      - does not generate all inferences required by the paper
   
6. Calculate Valence, arousal scores 
      ```sh
      python connotation_COMET_NRC.py
      ```  
   
7. Calculate Intellect, Appearance, Power scores
      ```sh
      python get_lexicon_score.py
      ```  
      - need to manually download Google word2vec
      - missing `Lexicons of bias - Gender stereotypes.csv`. This is used in `get_words(list[str])` to get the lexicons of a given list of words.
      - maybe the filled in lists, contain the lexicons so no need to run `get_words` 
      
      Acknowledgement:
      
      We borrowed some code from this repository: https://github.com/ddemszky/textbook-analysis
