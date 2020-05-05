import numpy as np
import matplotlib.pyplot as plt

def show_letters(w,h,arr,R = 1, C = None):
    if C == None:
        C = len(arr)

    fig = plt.figure(figsize = (w,h)) # size plot (weight,height)
    counter = 1

    rows = R
    columns = C

    for value in arr:
        fig.add_subplot(rows,columns,counter)
        plt.imshow(value,cmap = 'gray')
        counter += 1
    plt.show()





def create_patterns(vowels):
    patterns = []
    for vi in vowels:
        vi = vi.reshape(-1,1)
        patterns.append(vi)
    return patterns





def add_noise(p,percent,a,b):
    new_pattern = p.copy()
    for i in range(0,len(new_pattern)):
        temp = rng = np.random.rand() * 100.00
        if temp <= percent:
            if np.array_equal(new_pattern[i],a):
                new_pattern[i] = b
            else:
                new_pattern[i] = a
    return new_pattern

def create_patterns_with_noise(patterns,percent,a,b):
    a = np.array([a])
    b = np.array([b])
    new_patterns = []
    for pi in patterns:
        noise_pi = add_noise(pi,percent,a,b)
        new_patterns.append(noise_pi)
    return new_patterns


def get_accurracy(A,B):
    counter = 0
    goods = 0
    for i,j in zip(A,B):
        counter += 1
        if np.array_equal(i,j):
            goods += 1
    return goods * 100 / counter
