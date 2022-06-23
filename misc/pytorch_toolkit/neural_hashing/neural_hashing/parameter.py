import os

zSize = 48
#batchSize = 82
batchSize = 256
iteration = 100
learningRate = 0.0001
#learningRate = 0.00001
alpha = 0.53  # hyperparameter Discriminator
#alpha = 0.5
gamma = 1  # hyperparameter cauchy function
#gamma= 3
#GAMMA2 = GAMMA
gamma2 = 1.5
# beta1 = 2  # hyperparameter cauchy loss 1
beta = 0.5  # hyperparameter cauchy loss 2
delta = 0.6  # hyperparameter cauchy loss 1
numClasses = 19

#print("ZSIZE: %d" % zSize)
#print("BATCH_SIZE: %d" % batchSize)
#print("ITERATIONS: %d" % iteration)
#print("LEARNING_RATE: %f" % learningRate)
#print("GAMMA: %d" % gamma)
#print("GAMMA2: %d" % gamma2)
#print("BETA: %d" % beta1)
#print("BETA1: %d" % beta)
#print("DELTA: %d" % delta)
#print("NUM_CLASSES: %d" % numClasses)
# print("a")

#model_urls = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',}
#PreTrainedmodelPath = "/home/asim/DeepNeuralHashing/preTrained/alexnet-owt-4df8aa71.pth"
#PreTrainedmodelPath = alexnet


# change your training path
trainingDataPath = "/storage/asim/Hashing_MedMNISTV2/train1"
testDataPath = "/storage/asim/Hashing_MedMNISTV2/test1"  # change your test path

dataStorePath = "/storage/asim/Hashing_MedMNISTV2/Abl_study/disc_cauchyLoss"
if not os.path.exists(dataStorePath):
    os.makedirs(dataStorePath)

#print("Classes : \n")
# print(sorted(os.listdir(trainingDataPath)))
# print(dataStorePath.split('/')[-1])
# print(PreTrainedmodelPath)
