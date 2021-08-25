#!/bin/python3 -u
import numpy as np
import pickle

idx = 'EEEE'

class tranche:
    def __init__(self, folder, subtranches = None):
        self.folder = folder

        if subtranches is None:
            self.read()

        else:
            self.combine(subtranches)
            self.write()

    def combine(self, subtranches):
        print("combining...")

        self.pred = np.concatenate([st.pred for st in subtranches])
        print(self.pred.shape)

    def read(self):
        print("reading from %s..."%self.folder)

        with open("%s/%s_Mee.pickle"%(self.folder,idx), 'rb') as f:
             self.pred = pickle.load(f)

    def write(self):
        print("writing to %s..."%self.folder)

        with open("%s/%s_Mee.pickle"%(self.folder,idx), 'wb') as f:
            pickle.dump(self.pred, f)

T0 = tranche('T0/EE')
T1 = tranche('T1/EE')
T2 = tranche('T2/EE')
T3 = tranche('T3/EE')
T4 = tranche('T4/EE')
T5 = tranche('T5/EE')

big = tranche('EE', [T0,T1,T2,T3,T4,T5])
