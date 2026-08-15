#!/usr/bin/env python
# coding: utf-8

# # NAME: Prabhkirat Singh Marwah
# # SEC: B
# # BATCH: B4
# # ROLL NO: 47
# # ML LAB : Practical 3

# In[6]:


import pandas as pd 
import numpy as np 
df=pd.read_csv('enjoy.csv')


# In[7]:


X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


# In[10]:


def find_s(X,y):
    hypothesis=None
    print('initial hypothesis: ',hypothesis)

    for i in range (len(X)):
        if y[i]=='Yes':
            if hypothesis is None:
                hypothesis =X[i].copy()
            else:
                for j in range(len(hypothesis)):
                    if hypothesis[j] !=X[i][j]:
                        hypothesis[j]="?"
        print(f"\after training example {i+1}")
        print(hypothesis)
    return hypothesis
final_hypothesis = find_s(X,y)
print("\n final hypothesis ")
print(final_hypothesis)


# In[18]:


def candidate_elimination(concepts, target):
    specific = concepts[0].copy()
    general = [["?" for _ in range(len(specific))]]

    print("\n=== Candidate Elimination Algorithm ===\n")
    print("Initial S:", specific)
    print("Initial G:", general)
    print("\n---------------------------------------\n")

    for i, h in enumerate(concepts):
        if target[i] == 'Yes':  # Positive example
            for x in range(len(specific)):
                if h[x] != specific[x]:
                    specific[x] = "?"
        else:  # Negative example
            for x in range(len(specific)):
                if h[x] != specific[x]:
                    general.append(["?" if j != x else specific[x] for j in range(len(specific))])

        print(f"Example {i+1}:")
        print("S =", specific)
        print("G =", general)
        print("\n---------------------------------------\n")

    print("\n=== Final Hypotheses ===")
    print("Specific (S):", specific)
    print("General (G):", general)
    print("\n=========================\n")

    return specific, general


S, G = candidate_elimination(X, y)


# In[ ]:




