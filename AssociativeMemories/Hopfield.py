import numpy as np



class Hopfield():

    '''
        patterns
            list of p patterns of length n
            every pattern is a numpy array
            here the operator H is created
    '''


    def __init__(self,patterns):
        self.patterns = patterns.copy()
        self.p = len(patterns)
        self.n = len(patterns[0])
        self.H = np.zeros((self.n,self.n))
        I = np.identity(self.n)
        for pi in patterns:
            pi = pi.reshape(-1,1)
            tpi = pi.transpose()
            temp = pi.dot(tpi) - I
            self.H = np.add(self.H,temp)

    

    '''
        retrive process
        returns a pattern (local minimum)
        pattern
            pattern of length n

    '''
    def retrieve(self,pattern):
        return self.retrieve_process(pattern)

    def retrieve_process(self,pattern):
        x0 = pattern.reshape(-1,1)
        limit = 1000
        while True:
            limit -= 1
            xi = self.H.dot(x0)
            xi[xi>0] = 1
            xi[xi<0] = -1
            if np.array_equal(x0,xi) or limit == 0:
                break
            x0 = xi.copy()

        return x0
