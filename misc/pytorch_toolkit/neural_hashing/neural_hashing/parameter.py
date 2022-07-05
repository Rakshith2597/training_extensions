import os

zSize = 48
batchSize = 256
iteration = 100
learningRate = 0.0001
alpha = 0.53  # hyperparameter Discriminator
gamma = 1  # hyperparameter cauchy function
gamma2 = 1.5
beta = 0.5  # hyperparameter cauchy loss 2
delta = 0.6  # hyperparameter cauchy loss 1
numClasses = 19

# change your training path
trainingDataPath = "/storage/asim/Hashing_MedMNISTV2/train1"
testDataPath = "/storage/asim/Hashing_MedMNISTV2/test1"  # change your test path

dataStorePath = "/storage/asim/Hashing_MedMNISTV2/Abl_study/disc_cauchyLoss"
if not os.path.exists(dataStorePath):
    os.makedirs(dataStorePath)
