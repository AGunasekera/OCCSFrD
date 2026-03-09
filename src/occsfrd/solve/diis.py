import numpy as np

def overlapMatrix(oldErrorVecs):
    mat = np.zeros((len(oldErrorVecs), len(oldErrorVecs)))
    for i in range(len(oldErrorVecs)):
        for j in range(len(oldErrorVecs)):
            for t, aTensUp in enumerate(oldErrorVecs[0]):
                for d, aDiagUp in enumerate(aTensUp):
                    mat[i,j] += np.sum(oldErrorVecs[i][t][d] * oldErrorVecs[j][t][d])
    return mat

def LagrangianMatrix(oldErrorVecs):
    mat = np.zeros((len(oldErrorVecs) + 1, len(oldErrorVecs) + 1))
    mat[:-1, :-1] = overlapMatrix(oldErrorVecs)
    mat[-1, :-1] = 1
    mat[:-1, -1] = 1
    return mat

def getDIISWeights(oldErrorVecs):
    augmentedLagrangianVector = np.zeros(len(oldErrorVecs) + 1)
    augmentedLagrangianVector[-1] = 1
    LMatrix = LagrangianMatrix(oldErrorVecs)
    # print("LMatrix", LMatrix)
    augmentedWeightsVector = np.matmul(np.linalg.inv(LMatrix), augmentedLagrangianVector)
    return augmentedWeightsVector[:-1]

def updateAmpsDIIS(weights, oldAmplitudes, oldErrorVecs):
    newDIISAmplitudes = sum(weight * (oldAmplitudes[w] + oldErrorVecs[w]) for w, weight in enumerate(weights))
    return newDIISAmplitudes