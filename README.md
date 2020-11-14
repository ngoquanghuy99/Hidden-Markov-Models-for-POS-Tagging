## Hidden Markov Models for Part of speech Tagging

<p> An implementation of Part of Speech Tagging task for English using Hidden Markov Models. </p>
<p> Created by Ngo Quang Huy @ngoquanghuy99 </p>
<p> Email: ngoquanghuy1999lp@gmail.com </p>

### Overview
In this repo, i implemented Part-of-speech Tagging task using Hidden Markov Model and decoded by a dynamic programming algorithm named Viterbi.
There are 2 tagged datasets collected from the Wall Street Journal (WSJ).
Take a look at the Penn Treebank II tag set [here](http://relearn.be/2015/training-common-sense/sources/software/pattern-2.6-critical-fork/docs/html/mbsp-tags.html).
* One dataset (WSJ-2_21.pos) will be used for training.
* The other (WSJ-24.pos) for testing.
* The vocabulary is formed of the training data.
* The vocabulary is augmented with a set of 'unknown word tokens' described in `utils.py`
To improve accuracy, words that are not in the vocabulary are further analyzed to extract available hints as to their appropriate tag.
For example, the suffix 'ize' is a hint that the word is a verb, as in 'final-ize' or 'character-ize'.
They will all replace unknown words in both training set, testing set and vocabulary.
### Results on testing set
*accuracy*: 0.95 
## Getting started
### Install dependencies
#### Requirements
- python>=3.6
- numpy>=1.18.2
- nltk>=3.4.5
### Training

    $ python main.py
    
### Testing

    $ python test.py --sent "My heart is always breaking for the ghosts that haunt this room."
    
Output:

    $ [('My', 'PRP$'), ('heart', 'NN'), ('is', 'VBZ'), ('always', 'RB'), ('breaking', 'VBG'), ('for', 'IN'), ('the', 'DT'), ('ghosts', 'NNS'), ('that', 'WDT'), ('haunt', 'VBP'), ('this', 'DT'), ('room', 'NN'), ('.', '.')]
