import numpy as np

def get(g1, g2, shift,nb=2):
    g2[:,0] = g2[:,0] + 128

    g1a = g1.copy()
    g2a = g2.copy()
    
    g1a = g1a[g1a[:,3] <= shift*2]
    g2a = g2a[g2a[:,3] <= shift*3]
    
    g2a[:,3] += shift
    
    if nb == 2:
        g1b = g1.copy()
        g2b = g2.copy()

        g1b = g1b[g1b[:,3] > shift*2]
        g1b = g1b[g1b[:,3] <= shift*5]

        g2b = g2b[g2b[:,3] > shift*3]
        g2b = g2b[g2b[:,3] <= shift*5]

        g1b[:,3] += shift

        g2b[:,3] += shift*2

        g = np.vstack((g1a, g1b, g2a, g2b))
    
    else : 
        g = np.vstack((g1a,g2a))

    return g
