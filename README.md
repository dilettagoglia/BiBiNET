# BiBiNET: BiLSTM for bipolarity sentiment analysis.
_A Gated Recurrent Neural Network for Supervised Text
Classification: detecting hate speech from different online
textual genres._

<a href="https://github.com/dilettagoglia/BiBiNET/blob/main/LICENSE"><img src="https://img.shields.io/github/license/dilettagoglia/BiBiNET" alt="License" /></a>
 <a href="https://github.com/dilettagoglia/BiBiNET/stargazers"><img src="https://img.shields.io/github/stars/dilettagoglia/BiBiNET" alt="GitHub stars" /></a>
 <a href="https://github.com/dilettagoglia/BiBiNET/network/members"><img alt="GitHub forks" src="https://img.shields.io/github/forks/dilettagoglia/BiBiNET" /></a>

## Description
This project was developed for the ”Human Language Technologies” course of Professor Giuseppe Attardi.

## Directory structure (main elements)
```
BiBiNET
  │── src
  │    │── data_import.py
  │    │── data_prep.py
  │    │── preproc.py
  │    │── transform.py
  │    │── classifiers.py
  │    │── test.py
  │    │── utilities.py
  │    └── main.py
  └── data
  │    └── forum_data
  │    │   │── all_files.csv                # text
  │    │   └── annotations_metadata.csv     # labels
  │    └── twitter_1
  │    │   └── twitter_dataset.csv       
  │    └── twitter_2
  │    │    │── train.csv                
  │    │    └── test.csv     
  │    └── wikipedia_data
  │         │── train.csv                
  │         └── test.csv    
  └── glove
  │   │── glove.twitter.27B.100d      
  │   └── glove.twitter.27B.200d        
  └── model          
  │   └── model.h5                          # final model   
  └── requirements.txt
  └── report.pdf                            # project guide
```

## Quick start
Install Python:<br>
`sudo apt install python3`

Install pip:<br>
`sudo apt install --upgrade python3-pip`

Install requirements:<br>
`python -m pip install --requirement requirements.txt`

Execute [main](main.py)
```
cd src/
python main.py
```

## Corresponding author
**Dr. Diletta Goglia** <a href="https://orcid.org/0000-0002-2622-7495"><img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a> <br/>
**Postgraduate Student in MSc in Artificial Intelligence** <br/>
**Computer Science department, University of Pisa, Italy** <br/>
[d.goglia@studenti.unipi.it](mailto:d.goglia@studenti.unipi.it) <br/>
[dilettagoglia.netlify.app](www.dilettagoglia.netlify.app) 