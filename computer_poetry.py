import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
def plotWordFrequency(input):
    f = open(artist_file,'r')
    words = [x for y in [l.split() for l in f.readlines()] for x in y]
    data = sorted([(w, words.count(w)) for w in set(words)], key = lambda x:x[1], reverse=True)[:40] 
    most_words = [x[0] for x in data]
    times_used = [int(x[1]) for x in data]
    plt.figure(figsize=(20,10))
    plt.bar(x=sorted(most_words), height=times_used, color = 'grey', edgecolor = 'black',  width=.5)
    plt.xticks(rotation=45, fontsize=18)
    plt.yticks(rotation=0, fontsize=18)
    plt.xlabel('Cele mai des întâlnite cuvinte:', fontsize=18)
    plt.ylabel('Numărul de apariții:', fontsize=18)
    plt.title('Cele mai folosite cuvinte: %s' % (artist_file), fontsize=24)
    plt.show()
artist_file = 'poezii_compus.txt'
plotWordFrequency(artist_file)

#############################################################################################################
import markovify
import re
import random
import numpy as np
import os
import keras                          
from keras.models import Sequential
from keras.layers import LSTM 
from keras.layers.core import Dense

#############################################################################################################
def create_network(depth):           
  model = Sequential()
  model.add(LSTM(4, input_shape=(2, 2), return_sequences=True))   
  for i in range(depth):
    model.add(LSTM(8, return_sequences=True))   
  model.add(LSTM(2, return_sequences=True))     
  model.summary() 
  model.compile(optimizer='rmsprop',           
            loss='mse')                      
  if artist + ".poezie" in os.listdir(".") and train_mode == False:
    model.load_weights(str(artist + ".poezie"))
    print("loading saved network: " + str(artist) + ".poezie") 
  return model
  
#############################################################################################################
def markov(text_file):
colectie_de_poezii = open(text_file, "r", encoding='utf-8').read()
text_model = markovify.NewlineText(colectie_de_poezii, state_size = 2)     
return text_model
  
#############################################################################################################  
#voc="aeiouy"
#cons="bcdfghjklmnprstvxz"
#cc_o=["ch", "gh", "sp", "sc", "st", "sf", "zb", "zg", "zd", "zv", "jg", "jd", "sm", "sn", "sl", "zm", "zl", "jn", "tr", "cl", "cr", "pl", "pr", "dr", "gl", "gr", "br", "bl", "fl", "fr", "vl", "vr", "hr", "hl", "ml", "mr"]
#ccc_o=["spl", "spr", "str", "jgh", "zdr", "scl", "scr", "zgl", "zgr", "sfr"]
#dif=["ea", "oa", "ia", "ua", "iu", "uu", "ie", "ii"]
#trif=["eoa", "eai", "eau", "iau"]
#hiat=["aa","au", "ae", "ie", "ai", "ee", "oe", "oo", "yu"]

voc="aeiouăîâ"
cons="bcdfghjklmnprsștțvxz"
cc_o=["ch", "gh", "sp", "sc", "st", "sf", "zb", "zg", "zd", "zv", "șk", "șp", "șt", "șf", "șv", "jg", "jd", "sm", "sn", "sl", "șm", "șn", "șl", "zm", "zl", "jn", "tr", "cl", "cr", "pl", "pr", "dr", "gl", "gr", "br", "bl", "fl", "fr", "vl", "vr", "hr", "hl", "ml", "mr"]
ccc_o=["spl", "spr", "șpl", "șpr", "str", "ștr", "jgh", "zdr", "scl", "scr", "zgl", "zgr", "sfr"]
dif=["ea", "oa", "ia", "ua", "ua" "iu", "uu", "ie"]
trif=["eoa", "eai", "eau", "iau"]
hiat=["aa", "au", "ae","ie", "ai", "ee", "oe"]
def ccv(cuv, i):
    if cuv[i:i+2] in cc_o: 
        cuv=cuv[:i] + '-' + cuv[i:] 
        i=i+3  
    else: 
        cuv=cuv[:i+1] + '-' + cuv[i+1:]  
        i=i+3
    return cuv,i

def cccv(cuv, i):
    if cuv[i:i+3] in ccc_o:  
        cuv=cuv[:i] + '-' + cuv[i:]  
        #print cuv
        i=i+4  
    elif cuv[i+1:i+3] in cc_o:  
        cuv=cuv[:i+1] + '-' + cuv[i+1:] 
        #print cuv
        i=i+4  
    else: 
        cuv=cuv[:i+2] + '-' + cuv[i+2:] 
        #print cuv
        i=i+4  
    return cuv,i    

def ccccv(cuv,i):
    if cuv[i+1:i+4] in ccc_o:  
        cuv=cuv[:i+1] + '-' + cuv[i+1:]
        i=i+5
    elif cuv[i+2:i+4] in cc_o:  
        cuv=cuv[:i+2] + '-' + cuv[i+2:]
        i=i+5
    else: 
        cuv=cuv[:i+3] + '-' + cuv[i+3:]
        i=i+5
    return cuv, i   

def vvv(cuv, i):
    if cuv[i:i+3] in trif: 
        i=i+3  
    elif cuv[i+1:i+3] in dif:
        cuv=cuv[:i+1] + '-' + cuv[i+1:]  
        i=i+4  
        #print cuv
    else:  
        i=i+3  
    return cuv,i


def syll(prop):
  numarSilabe=0
  for word in prop.split():
    temp = word
    for c in word:
      if c in '-!.,?;:':
        temp = temp.replace(c,"")
    numarSilabe += syllCuv(temp)
  return numarSilabe
  
  
def syllCuv(cuv):
    i=0
    nucleu=0
    while i<len(cuv)-1:
        #print i, cuv[i]
        if cuv[i] in voc:  
            nucleu=1  
            if cuv[i+1] in voc:  
                if (i+2)<=len(cuv)-1:  
                    if cuv[i+2] in voc: 
                        if (i+3)<=len(cuv)-1: 
                            if cuv[i+3] in voc: 
                                cuv=cuv[:i+2] + '-' + cuv[i+2:]  
                                i=i+5
                            else:  
                                cuv,i=vvv(cuv,i)
                        else: 
                            cuv,i=vvv(cuv,i)
                    else: 
                        if cuv[i:i+2] in dif:  
                            i=i+2  
                        elif cuv[i:i+2] in hiat: 
                            cuv=cuv[:i+1] + '-' + cuv[i+1:] 
                            #print cuv
                            i=i+3  
                        else: 
                            i=i+2 
                else:  
                    i=i+2
            else: #avem VC
                i=i+1 
        else:  
            if nucleu == 0: 
                i=i+1 
            else:  
                if cuv[i+1] in cons: 
                    if (i+2) < (len(cuv)-1): 
                        if cuv[i+2] in cons:  
                            if (i+3) < (len(cuv)-1):
                                if cuv[i+3] in cons:  
                                    if (i+4) < (len(cuv)-1):
                                        if cuv[i+4] in cons:  
                                            cuv=cuv[:i+2] + '-' + cuv[i+2:]
                                            i=i+6
                                        else:  
                                            cuv,i=ccccv(cuv,i)
                                    else: 
                                        if cuv[i+4] in cons:  
                                            i=i+5  
                                        else: 
                                            cuv,i=ccccv(cuv,i)
                                else: 
                                    cuv,i=cccv(cuv,i)
                            else:  
                                if cuv[i+3] in cons: 
                                    i=i+4
                                else:  
                                    cuv,i=cccv(cuv,i)    
                        else:  
                            cuv,i=ccv(cuv,i)  
                    elif (i+2)==(len(cuv)-1):  
                        if cuv[i+2] in voc: 
                            cuv,i=ccv(cuv,i) 
                        else: 
                            i=i+3
                    else: 
                        break
                else:  
                    cuv=cuv[:i] + '-' + cuv[i:] 
                    #print cuv
                    i=i+2
    return len(cuv.split('-'))
  
############################################################################################################# 
import json

qRime = json.load(open("Rimeq.txt"))
wRime = json.load(open("Rimew.txt"))
eRime = json.load(open("Rimee.txt"))
rRime = json.load(open("Rimer.txt"))
tRime = json.load(open("Rimet.txt"))
yRime = json.load(open("Rimey.txt"))
uRime = json.load(open("Rimeu.txt"))
iRime = json.load(open("Rimei.txt"))
oRime = json.load(open("Rimeo.txt"))
pRime = json.load(open("Rimep.txt"))
aRime = json.load(open("Rimea.txt"))
sRime = json.load(open("Rimes.txt"))
dRime = json.load(open("Rimed.txt"))
fRime = json.load(open("Rimef.txt"))
gRime = json.load(open("Rimeg.txt"))
hRime = json.load(open("Rimeh.txt"))
jRime = json.load(open("Rimej.txt"))
kRime = json.load(open("Rimek.txt"))
lRime = json.load(open("Rimel.txt"))
zRime = json.load(open("Rimez.txt"))
xRime = json.load(open("Rimex.txt"))
cRime = json.load(open("Rimec.txt"))
vRime = json.load(open("Rimev.txt"))
bRime = json.load(open("Rimeb.txt"))
nRime = json.load(open("Rimen.txt"))
mRime = json.load(open("Rimem.txt"))
aqRime = json.load(open("Rimeă.txt"))
iqRime = json.load(open("Rimeî.txt"))
tqRime = json.load(open("Rimeț.txt"))
sqRime = json.load(open("Rimeș.txt"))
aaqRime = json.load(open("Rimeâ.txt"))
def lista_rime(cuvant):
  temp = cuvant.lower()
  cuvantAdaptat = temp
  for c in temp:
    if c in '-!.,?;:':
      cuvantAdaptat = cuvantAdaptat.replace(c,"")
  if len(cuvantAdaptat) >0:
    if cuvantAdaptat[0] in 'qwertyuiopasdfghjklzxcvbnmâșțăî':
      if cuvantAdaptat[0] == 'q':
        if cuvantAdaptat in qRime:
          return qRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'w':
        if cuvantAdaptat in wRime:
          return wRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'e':
        if cuvantAdaptat in eRime:
          return eRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'r':
        if cuvantAdaptat in rRime:
          return rRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 't':
        if cuvantAdaptat in tRime:
          return tRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'y':
        if cuvantAdaptat in yRime:
          return yRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'u':
        if cuvantAdaptat in uRime:
          return uRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'i':
        if cuvantAdaptat in iRime:
          return iRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'o':
        if cuvantAdaptat in oRime:
          return oRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'p':
        if cuvantAdaptat in pRime:
          return pRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'a':
        if cuvantAdaptat in aRime:
          return aRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 's':  
        if cuvantAdaptat in sRime:
          return sRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'd':
        if cuvantAdaptat in dRime:
          return dRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'f':
        if cuvantAdaptat in fRime:
          return fRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'g':
        if cuvantAdaptat in gRime:
          return gRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'h':
        if cuvantAdaptat in hRime:
          return hRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'j':
        if cuvantAdaptat in jRime:
          return jRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'k':
        if cuvantAdaptat in kRime:
          return kRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'l':
        if cuvantAdaptat in lRime:
          return lRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'z':
        if cuvantAdaptat in zRime:
          return zRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'x':
        if cuvantAdaptat in xRime:
          return xRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'c':
        if cuvantAdaptat in cRime:
          return cRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'v':
        if cuvantAdaptat in vRime:
          return vRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'b':
        if cuvantAdaptat in bRime:
          return bRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'n':
        if cuvantAdaptat in nRime:
          return nRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'm':
        if cuvantAdaptat in mRime:
          return mRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'ă':
        if cuvantAdaptat in aqRime:
          return aqRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'ș':
        if cuvantAdaptat in sqRime:
          return sqRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'ț':
        if cuvantAdaptat in tqRime:
          return tqRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'î':
        if cuvantAdaptat in iqRime:
          return iqRime[cuvantAdaptat]

      if cuvantAdaptat[0] == 'â':
        if cuvantAdaptat in aaqRime:
          return aaqRime[cuvantAdaptat]
  return []

#############################################################################################################
def split_lyrics_file(text_file):
  text = open(text_file, encoding='utf-8').read()
  text = text.split("\n")
  while "" in text:
    text.remove("")
  print(type(text[0]))
  return text

#############################################################################################################
def build_dataset(lines, rhyme_list):
  print(lines)
  dataset = []
  line_list = []
  for line in lines:                        
    line_list = [line, syll(line), rhyme(line, rhyme_list)]  
    dataset.append(line_list) 
  x_data = []
  y_data = []
  for i in range(len(dataset) - 3):
    line1 = dataset[i    ][1:]    
    line2 = dataset[i + 1][1:]
    line3 = dataset[i + 2][1:]
    line4 = dataset[i + 3][1:]
    x = [line1[0], line1[1], line2[0], line2[1]]
    x = np.array(x)
    x = x.reshape(2,2)
    x_data.append(x)
    y = [line3[0], line3[1], line4[0], line4[1]]
    y = np.array(y)
    y = y.reshape(2,2)
    y_data.append(y)
  x_data = np.array(x_data)
  y_data = np.array(y_data)
  return x_data, y_data

#############################################################################################################
def rhyme(line, rhyme_list):
  word = re.sub(r"\W+", '', line.split(" ")[-1]).lower() 
  rhymeslist = lista_rime(word)
  rhymeslistends = []
  for i in rhymeslist:               
    rhymeslistends.append(i[-2:])    
  try:
    rhymescheme = max(set(rhymeslistends), key=rhymeslistends.count) 
  except Exception:
    rhymescheme = word[-2:]
  try:
    float_rhyme = rhyme_list.index(rhymescheme)
    float_rhyme = float_rhyme / float(len(rhyme_list))
    return float_rhyme
  except Exception:
    float_rhyme = 0
    return float_rhyme

#############################################################################################################
def rhymeindex(lyrics):
  if str(artist) + ".rime" in os.listdir(".") and train_mode == False:
    return open(str(artist) + ".rime", "r",encoding='utf-8').read().split("\n")
  else:
      rhyme_master_list = []
      for i in lyrics:
        word = re.sub(r"\W+", '', i.split(" ")[-1]).lower() 
        if len(word) > 1:
          rhymeslist = lista_rime(word)   
          rhymeslistends = []      
          for i in rhymeslist:         
            rhymeslistends.append(i[-2:])    
          try:
            rhymescheme = max(set(rhymeslistends), key=rhymeslistends.count)
          except Exception:           
            rhymescheme = word[-2:]
          rhyme_master_list.append(rhymescheme)
      rhyme_master_list = list(set(rhyme_master_list))  
      reverselist = [x[::-1] for x in rhyme_master_list]
      reverselist = sorted(reverselist)
      rhymelist = [x[::-1] for x in reverselist]
      print("Lista de rime sortate după ultimele 2 litere:")
      print(rhymelist)
      f = open(str(artist) + ".rime", "w", encoding='utf-8')
      f.write("\n".join(rhymelist))
      f.close()
      return rhymelist

#############################################################################################################
def train(x_data, y_data, model):  
	model.fit(np.array(x_data), np.array(y_data),
			  batch_size=2,
			  epochs=7,
			  verbose=1, validation_split = 0.2)
	model.save_weights(artist + ".poezie")

#############################################################################################################
def vectors_into_poetry(vectors, generated_lyrics, rhyme_list):
  print ("\n\n")	
  print ("Poezie:")
  print ("\n\n")
  def last_word_compare(poetry, line2):
    penalty = 0 
    for line1 in poetry:
      word1 = line1.split(" ")[-1]
      word2 = line2.split(" ")[-1]
      if(len(word1)>1):
        while word1[-1] in "?!,. ":
          word1 = word1[:-1]
      if(len(word2)>1):
        while word2[-1] in "?!,. ":
          word2 = word2[:-1]
      if word1 == word2:
        penalty += 0.2
    return penalty
  def calculate_score(vector_half, syllables, rhyme, penalty):
    desired_syllables = vector_half[0]
    desired_rhyme = vector_half[1]
    desired_syllables = desired_syllables * maxsyllables
    desired_rhyme = desired_rhyme * len(rhyme_list)
    score = 1.0 - abs(float(desired_syllables) - float(syllables)) + abs(float(desired_rhyme) - float(rhyme)) - penalty
    return score
  dataset = []
  for line in generated_lyrics:
    line_list = [line, syll(line), rhyme(line, rhyme_list)]
    dataset.append(line_list)
  poetry = []
  vector_halves = []
  for vector in vectors:
    vector_halves.append(list(vector[0][0])) 
    vector_halves.append(list(vector[0][1]))
  for vector in vector_halves:
    scorelist = []
    for item in dataset:
      line = item[0]
      if len(poetry) != 0:
        penalty = last_word_compare(poetry, line)
      else:
        penalty = 0
      total_score = calculate_score(vector, item[1], item[2], penalty)
      score_entry = [line, total_score]
      scorelist.append(score_entry)
    fixed_score_list = [0]
    for score in scorelist:
      fixed_score_list.append(float(score[1]))
    max_score = max(fixed_score_list)
    for item in scorelist:
      if item[1] == max_score:
        poetry.append(item[0])
        print (str(item[0]))
        for i in dataset:
          if item[0] == i[0]:
            dataset.remove(i)
            break
        break     
  return poetry

#############################################################################################################
def main(depth, train_mode):
  model = create_network(depth)  
  text_model = markov(text_file)  
  if train_mode == True:
    bars = split_lyrics_file(text_file)  
  if train_mode == False:
    bars = generate_lyrics(text_model, text_file) 
  rhyme_list = rhymeindex(bars)
  if train_mode == True: 
    x_data, y_data = build_dataset(bars, rhyme_list)
    train(x_data, y_data, model)
  if train_mode == False:
    vectors = compose_poetry(bars, rhyme_list, text_file, model)
    poetry = vectors_into_poetry(vectors, bars, rhyme_list)
    f = open(poetry_file, "w", encoding='utf-8')
    count_rhymes(poetry)
    count_syll(poetry)

#############################################################################################################
def count_rhymes(poetry):
  rime=0
  cuvinte_rime = []
  for i in range(len(poetry)-1):
    vers1 = poetry[i]   
    vers2 = poetry[i+1]
    word1 = ""
    word2 = ""
    word3 = ""
    word4 = ""
    
    word1good = False
    word2good = False
    word3good = False
    word4good = False
    
    word1 = re.sub(r"\W+", '', vers1.split(" ")[-1]).lower() 
    if(len(word1)>2):
      for c in word1:
        if c in '-!.,?;:':
          word1 = word1.replace(c,"")
      word1good= True

    word2 = re.sub(r"\W+", '', vers2.split(" ")[-1]).lower() 
    if(len(word2)>2):
      for c in word2:
        if c in '-!.,?;:':
          word2 = word2.replace(c,"")
      word2good= True

    if (i+2) < len(poetry):
      vers3 = poetry[i+2]
      word3 = re.sub(r"\W+", '', vers3.split(" ")[-1]).lower() 
      if(len(word3)>2):
        for c in word3:
          if c in '-!.,?;:':
            word3 = word3.replace(c,"")
        word3good= True
    if (i+3) < len(poetry):
      vers4 = poetry[i+3]
      word4 = re.sub(r"\W+", '', vers4.split(" ")[-1]).lower() 
      if(len(word4)>2):
        for c in word4:
          if c in '-!.,?;\n\r':
            word4 = word4.replace(c,"")
        word4good= True
    
    if word1good:
      if word2good:
        if word1[-2:] == word2[-2:]:
          
          rime +=1
          cuvinte_rime.append(i)
          cuvinte_rime.append(i+1)
          continue
      if word3good:
        if word1[-2:] == word3[-2:]:
          rime +=1
          cuvinte_rime.append(i)
          cuvinte_rime.append(i+2)          
          continue
      if word4good:
        if word1[-2:] == word4[-2:]:
          rime +=1
          cuvinte_rime.append(i)
          cuvinte_rime.append(i+3)          
          continue

  cuvinte_rime_master = list(set(cuvinte_rime))
  procent = len(cuvinte_rime_master)/len(poetry) * 100
  print(str(procent)+"% " + "rime") 


def count_syll(poetry):
  count = 0;
  for bar in poetry:
    count += syll(bar)
  media = count/len(poetry)
  print(str(media) + " silabe/vers")

#############################################################################################################
depth = 4 
artist = "artist"
model_file = "artist.poezie"
text_file = "corpus_poezii.txt"
poetry_file = "out.txt"
maxsyllables = 10
maxverses =  10

#############################################################################################################
#### use this if you want to train the network
train_mode = True       
main(depth, train_mode)

#############################################################################################################
#### use this if you have the pre-trained model uploaded
train_mode = False
main(depth, train_mode)





