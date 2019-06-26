# ComputerPoetry
"Can a computer write poetry?" 

This project proposes a Romanian based application developed by using both probabilistic and recurrent neural network models. The goal is to generate a text with metrical constraints (rhyme, number of syllables), that can be read as poetry. The idea of the system came from [GhostWriter](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP221.pdf). Many of the functions used were adapted from [here](https://www.kaggle.com/paultimothymooney/poetry-generator-rnn-markov/notebook) and [here](https://github.com/mary-octavia/Syllabification.git). I also made available the data set that we used for training, the current rhyme dictionary and a pre-trained model.


## Getting started

The environment used to develop and test the app is [Colab](https://colab.research.google.com/) from Google. Colab provides GPU and it’s totally free. If you need more information on how to use Google Colab notebooks, you cand find good information [here](https://towardsdatascience.com/getting-started-with-google-colab-f2fff97f594c).

1. Click on New Notebook. 
2. Go to Edit -> Notebook Settings and choose Python3 and Hardware Accelerator, GPU.

## Prerequisites

Using the new created Notebook, you need to install [Markovify](https://github.com/jsvine/markovify):

```python
! pip install markovify
```

