# POS tagging of non-standard text

In language processing, non-standard languages such as Swiss German dialects typically have to be normalized in order to 
achieve satisfactory results in downstream tasks such as POS-tagging. 

**baseline.py** performs a simple normalisation algorithm on tokenized Swiss German text which is based on the most 
frequent normalisation in another text. The text is then POS-tagged in three different normalisation settings and the results are evaluated in **evaluation.py**.

## Utilisation
- baseline.py
```sh
$ python3 baseline.py WUS_POS_data/train.txt WUS_POS_data/dev.txt WUS_POS_data/test.txt
```
- evaluation.py
```sh
$ python3 evaluation.py WUS_POS_data/dev_norm_out.txt WUS_POS_data/test_norm_out.txt
```

## Data
For this script, I worked with the [WUS corpus](https://whatsup.linguistik.uzh.ch).
In the corpus, there are three tab-seperated files **dev.txt**, **test.txt** and **train.txt** which are each structured 
in the following way:
- first column: original non-standard word
- second column: manual normalisation (gold standard)
- third column: manually written POS-tag (gold standard)

Messages are seperated by empty lines. 
```sh
nena	nena	NE
böge	Böge	VVFIN

wie	wie	KOUS
gohts	geht es	VVFIN+PPER
?	?	$.

tip top	tipptopp	ADJD
..	..	$.
hett	hätte	VAFIN
kli	klein	ADJD
sushi	Sushi	NN
dihai	daheim	ADV
emojiQcatFaceWithWrySmile	emojiQcatFaceWithWrySmile	EMOJI
```

## Normalisation
In **baseline.py**, the normalisation gold standard column of **train.txt** is used in order to generate new
normalisations for **dev.txt** and **test.txt**. We can distinguish between three normalization methods: 
- Unique: Words that occur in the training set exactly once get normalized in the way they were normalized in the training set
- Ambiguous: Words that have more than one normalization in the training set get the most frequent normalization or a random one in the case of a tie. 
- New: Words that don't occur in the training set don't get normalized (the original non-standard word is copied.)

The output files **dev_norm_out.txt** and **test_norm_out.txt** are formated in the following way: 
- 1st column: normalization strategy 
- 2nd column: original non-standard word
- 3rd column: predicted normalisation (baseline)
- 4th column: manual normalisation (gold standard)
- 5th column: manually written POS-tag (gold standard)

## POS tagging
In **evaluation.py**, the predicted normalisation (3rd column in **dev_norm_out.txt** and **test_norm_out.txt**) gets 
Part-of-speech tagged by a pre-trained tagger. 
The tagger is part of a [spaCy model for german](https://spacy.io/models/de).

POS tags are created in three different normalisation settings: 
- **upper bound**: the POS-tagger is applied to the manually normalized data (gold standard)
- **lower bound**: the POS-tagger is applied to the original non-standard data 
- **baseline**: the POS-tagger is applied to the data that was normalised in **baseline.py**


The output files **dev_norm_POS_out.txt** and **test_norm_POS_out.txt** are structured in the following way: 
- 1st column: normalization strategy 
- 2nd column: original non-standard word
- 3rd column: predicted normalisation (baseline)
- 4th column: manual normalisation (gold standard)
- 5th column: manually written POS-tag (gold standard)
- 6th column: POS tags (lower bound)
- 7th column: POS tags (upper bound)
- 8th column: POS tags (baseline)

## Evaluation
Finally, in **evaluation.py** the POS tagger gets evaluated in the different normalization settings. 
The newly generated POS tags that were created in the three settings are compared to the manually written POS tags 
(gold standard). The results of the evaluation can be found in **eval_report.txt**.


## Requirements
- Python version 3.6 or newer
- Install tabulate and spaCy and download spaCy's small model for German: 
```sh
# $ pip install -U pip setuptools wheel
# $ pip install -U spacy
# $ python -m spacy download de_core_news_sm
# $ pip install tabulate
```

## Folder Structure
```
project
│   README.md
│   baseline.py    
│   evaluation.py  
│   eval_report.txt  
│
└───WUS_POS_data
        dev.txt
        dev_norm_out.txt
        dev_norm_POS_out.txt
        test.txt
        test_norm_out.txt
        test_norm_POS_out.txt
        train.txt

```

## License
This script is licensed under the term of the MIT License, see the file LICENSE for more details. 