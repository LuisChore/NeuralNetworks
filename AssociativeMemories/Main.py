import numpy as np
import matplotlib.pyplot as plt
from vowels import vowels,vowels_s
from Hopfield import Hopfield
from Steinbuch import Steinbuch
from Misc import show_letters,get_accurracy,create_patterns,create_patterns_with_noise,add_noise







###############################################################################
###############################################################################
############################     HOPFIELD     #################################
###############################################################################
###############################################################################



def testing_hopfield(list_patterns,hopfield_net):
    inputs_outpus = []
    for i in list_patterns:
        retrieve_i = hopfield_net.retrieve(i)
        i = i.reshape(-1,5)
        retrieve_i = retrieve_i.reshape(-1,5)
        inputs_outpus.append(i)
        inputs_outpus.append(retrieve_i)
    show_letters(5,10,inputs_outpus,5,2)




def process_noise_patterns(patterns,hopfield_net):
    percents = [0,5,10,15,20,25,30,35,40,45,50]
    for i in percents:
        patterns_with_noise = create_patterns_with_noise(patterns,i,1,-1)
        testing_hopfield(patterns_with_noise,hopfield_net)




def process_graphic_error_hopfield(vowel,hopfield_net,title,its = 100):
    x = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
    y = [0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(0,its):
        for j in range(0,len(y)):
            new_p = add_noise(vowel,x[j],1,-1)
            retrive_p = hopfield_net.retrieve(new_p)
            y[j] += get_accurracy(vowel,retrive_p)

    y = [i/100 for i in y]
    plt.plot(x,y)
    plt.title(title)
    plt.xlabel("Noise")
    plt.ylabel("Acurracy %")
    plt.show()


def graphic_error_hopfield(patterns,hopfield_net):
    labels = ["A","E","I","O","U"]
    for vowel,label in zip (patterns,labels):
        process_graphic_error_hopfield(vowel,hopfield_net,label)




'''
show_letters(10,5,vowels)
patterns = create_patterns(vowels)
hop = Hopfield(patterns) ## constructor
testing_hopfield(patterns,hop) #testing original patterns
process_noise_patterns(patterns,hop) # testing patterns with noise
graphic_error_hopfield(patterns,hop) # graphic error
'''












###############################################################################
###############################################################################
############################     Steinbuch     #################################
###############################################################################
###############################################################################




def testing_steinbuch(list_patterns,steinbuch_net):
    inputs_outpus = []
    for i in list_patterns:
        retrieve_i = steinbuch_net.retrieve(i)
        i = i.reshape(-1,5)
        retrieve_i = retrieve_i.reshape(-1,5)
        inputs_outpus.append(i)
        inputs_outpus.append(retrieve_i)
    show_letters(5,10,inputs_outpus,5,2)




def process_noise_patterns_s(patterns,steinbuch_net):
    percents = [0,5,10,15,20,25,30,35,40,45,50]
    for i in percents:
        patterns_with_noise = create_patterns_with_noise(patterns,i,1,0)
        testing_steinbuch(patterns_with_noise,steinbuch_net)




def process_graphic_error_steinbuch(vowel,steinbuch_net,title,target,its = 100):
    x = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
    y = [0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    t = np.array([0,0,0,0,0])
    t[target] = 1
    t = t.reshape(-1,1)
    for i in range(0,its):
        for j in range(0,len(y)):
            new_p = add_noise(vowel,x[j],1,0)
            retrieve_p = steinbuch_net.retrieve(new_p)
            y[j] += get_accurracy(t,retrieve_p)

    y = [i/100 for i in y]
    plt.plot(x,y)
    plt.title(title)
    plt.xlabel("Noise")
    plt.ylabel("Acurracy %")
    plt.show()


def graphic_error_steinbuch(patterns,steinbuch_net):
    labels = ["A","E","I","O","U"]
    targets = [0,1,2,3,4]
    for vowel,label,target in zip (patterns,labels,targets):
        process_graphic_error_steinbuch(vowel,steinbuch_net,label,target)





show_letters(10,5,vowels_s)
patterns = create_patterns(vowels_s)
steinbuch = Steinbuch(patterns) ## constructor
testing_steinbuch(patterns,steinbuch) #testing original patterns
process_noise_patterns_s(patterns,steinbuch) # testing patterns with noise
graphic_error_steinbuch(patterns,steinbuch) # graphic error
