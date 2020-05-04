import numpy as np

class Steinbuch():

    '''
        patterns
            list of p patterns of length n
            every pattern is a numpy array
            here the operator H is created
        Note
            given the order of the patterns
            the pattern i is gonna be associated to
            yi = vector of size p with all elements
            are 0's except the position i, it is 1

    '''
    def __init__(self,patterns):
        self.eps = 1
        self.patterns = patterns.copy()
        self.p = len(patterns)
        self.n = len(patterns[0])

        self.H = np.zeros((self.p,self.n))
        for i in range(0,self.p):
            for j in range(0,self.n):
                if patterns[i][j] == 1:
                    self.H[i][j] = self.eps
                else:
                    self.H[i][j] = -self.eps

    def retrieve(self,pattern):
        p = pattern.copy()
        p = p.reshape(-1,1)
        ans = self.H.dot(p)
        max_val = np.amax(ans)
        ans[ans < max_val] = 0
        ans[ans == max_val] = 1
        return ans
