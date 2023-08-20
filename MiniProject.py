# #Question 1
# # That part helps me to importing
# import h5py
# import numpy as np
# from matplotlib import pyplot as plt
# import seaborn as sn
# import sys
# class Q1Answer(object):
# # For question 1, we need to create autoencoder. Therefore, on that class, I worked to create autoencoder.
#     def spesification(self, inL, hidL):
#     # To determine our autoencoder, we need to assign our value spesification.    
#         Lout = inL
#         baseW = np.sqrt(6 / (inL + hidL))
#         W1 = np.random.uniform(-baseW, baseW, size=(inL, hidL))
#         b1 = np.random.uniform(-baseW, baseW, size=(1, hidL))
#         W2 = np.random.uniform(-baseW, baseW, size=(hidL, Lout))
#         b2 = np.random.uniform(-baseW, baseW, size=(1, Lout))
#         We = (W1, W2, b1, b2)
#         return We

#     def dataTraining(self, data, pmeters,totalBatch, eta=0.1, alpha=0.9, epoch=10): 
#         # As we can understand, this is our one of the main function which trains our data.
#         listJ = []
#         inL = pmeters["inL"]
#         hidL = pmeters["hidL"]
#         We = self.spesification(inL,  hidL)
#         mWe = (0, 0, 0, 0)
        
#         itenum = int(data.shape[0] / totalBatch)

#         for i in range(epoch):
#             totJ = 0
#             beg = 0
#             end = totalBatch
#             perm = np.random.permutation(data.shape[0])
#             data = data[perm]
#             mWe = (0, 0, 0, 0)

#             for j in range(itenum):
#                 batchData = data[beg:end]
#                 J, Jgrad, cache = self.aeCost(We, batchData, pmeters)
#                 We, mWe = self.uptWeight(Jgrad, cache, We, mWe, eta, alpha)
#                 totJ = J + totJ
#                 beg = end
#                 end = totalBatch + end

#             totJ = totJ/itenum
#             print("Our loss is: {:.2f} [Total Epoch: {} Current Epoch: {}]".format(totJ, i+1, epoch))
#             listJ.append(totJ)

#         print("\n")
#         return We, listJ


#     def aeCost(self, We, data, pmeters):
#         # This definition defines our gradients' first error. Then, I did forward pass thanks to detection of gradients.
#         listData = data.shape[0]
#         W1, W2, b1, b2 = We
#         p = pmeters["p"]
#         beta = pmeters["beta"]
#         lambd = pmeters["lambd"]
#         inL = pmeters["inL"]
#         hidL = pmeters["hidL"]
        
#         #That side operates forward pass.
#         lin1 = data @ W1 + b1
#         hid = 1 / (1 + np.exp(-lin1))
#         lin2 = hid @ W2 + b2
#         hidd = 1 / (1 + np.exp(-lin2))
#         # That part operates sigmoid variables.       
#         derivHid = hid * (1 - hid)
#         derivHidd = hidd * (1 - hidd)
               
#         base_p = hid.mean(axis=0, keepdims=True)
        
#         totalLoss = 0.5/listData * (np.linalg.norm(data - hidd, axis=1) ** 2).sum()
#         Tyk = 0.5 * lambd * (np.sum(W1 ** 2) + np.sum(W2 ** 2)) #Tyk equation is to apply regularization and cope overfitting
#         KulLeiDiv = p * np.log(p/base_p) + (1 - p) * np.log((1 - p)/(1 - base_p)) #KulLeiDiv makes sparcity tuning
#         KulLeiDiv = beta * KulLeiDiv.sum()

#         backPropJ = totalLoss + Tyk + KulLeiDiv
        
#         derivTotalLoss = -(data - hidd)/listData
#         derivTykk = lambd * W2
#         derivTyk = lambd * W1
#         derivKulLeiDiv = beta * (- p/base_p + (1-p)/(1 - base_p))/listData

#         cache = (data, hid, derivHid, derivHidd)
#         gradbackPropJ = (derivTotalLoss, derivTykk, derivTyk, derivKulLeiDiv)


#         return backPropJ, gradbackPropJ, cache


#     def uptWeight(self, Jgrad, cache, We, We_m, LearRate, alpha):

#         W1, W2, b1, b2 = We
#         derivdW1 = 0
#         derivdW2 = 0
#         derivdB1 = 0
#         derivdB2 = 0

#         data, hid, derivHid, derivHidd = cache
#         totalLoss, derivTykk, derivTyk, derivKulLei = Jgrad
#         delta = totalLoss * derivHidd
#         derivdW2 = hid.T @ delta + derivTykk
#         derivdB2 = delta.sum(axis=0, keepdims=True)
#         delta = derivHid * (delta @ W2.T + derivKulLei)
#         derivdW1 = data.T @ delta + derivTyk
#         derivdB1 = delta.sum(axis=0, keepdims=True)
#         derivWe = (derivdW1, derivdW2, derivdB1, derivdB2)

# # We need to update our weights based on momentum
#         W1, W2, b1, b2 = We
#         derivdW1, derivdW2, derivdB1, derivdB2 = derivWe
#         W1_m, W2_m, B1_m, B2_m = We_m

#         W1_m = LearRate * derivdW1 + alpha * W1_m
#         W2_m = LearRate * derivdW2 + alpha * W2_m
#         B1_m = LearRate * derivdB1 + alpha * B1_m
#         B2_m = LearRate * derivdB2 + alpha * B2_m

#         W1 -= W1_m
#         W2 -= W2_m
#         b1 -= B1_m
#         b2 -= B2_m

#         We = (W1, W2, b1, b2)
#         We_m = (W1_m, W2_m, B1_m, B2_m)

#         return We, We_m


#     def predict(self, data, We):
#         W1, W2, b1, b2 = We

#         lin1 = data @ W1 + b1
#         hid = 1 / (1 + np.exp(-lin1))
#         lin2 = hid @ W2 + b2
#         hidd = 1 / (1 + np.exp(-lin2))
#         return hidd
        
# # This is our last file to call data and run our code.        
# dataFile = h5py.File("data1.h5", 'r')

# dataList = list(dataFile.keys())
# derivT = dataFile.get('data')
# derivT = np.array(derivT)

# dataZeros = np.zeros((10240,16,16))
# for i in range(10240):
#     img = derivT[i]
#     img = img.transpose((1, 2, 0))
#     R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
#     dataZeros[i] = 0.2126 * R + 0.7152 * G + 0.0722 * B
    
# # This part is for normalization.
# dataZeros = np.reshape(dataZeros, (10240, 256))  # We transform our data to 2 dimensions which is 10240 and 256.
# dataZeros = dataZeros - dataZeros.mean(axis=1, keepdims=True)  # We wanted to differentiate image.
# dataStandart = np.std(dataZeros)  # We found standart of data.
# dataZeros = np.clip(dataZeros, - 3 * dataStandart, 3 * dataStandart)  # We made our standart interval by increasing and decreasing by 3
# dataZeros = (dataZeros - dataZeros.min())/(dataZeros.max() - dataZeros.min()) # We apply normalization
# dataZeros = 0.1 + dataZeros * 0.8  # We map our data between 0.1 to 0.9
# trainData = dataZeros
# dataZeros = np.reshape(dataZeros, (10240, 16, 16))  # We reshape our data to check results better.
# dataa = derivT.transpose((0, 2, 3, 1))
# dataa = np.clip(dataa,0,1)  # Clipping original data
# figure, axis = plt.subplots(15, 25, figsize=(25, 15))
# figuree, axiss = plt.subplots(15, 25, figsize=(25, 15))

# for i in range(15):
#     for j in range(25):
#         k = np.random.randint(0, dataa.shape[0])
#         axis[i, j].imshow(dataa[k].astype('float'))
#         axis[i, j].axis("off")
#         axiss[i, j].imshow(dataZeros[k], cmap='gray')
#         axiss[i, j].axis("off")
# figuree.savefig("grayful.png")
# figure.savefig("colorful.png")

# lambd = 0
# LearRate = 0.1
# totalBatch = 16
# p = 0.05
# alpha = 0.85
# epoch = 100
# beta = 0.1
# inL = trainData.shape[1]
# hidL = 100

# pmeters = {"p": p, "beta": beta, "lambd": lambd, "inL": inL, "hidL": hidL}
# func = Q1Answer()
# w = func.dataTraining(trainData, pmeters, totalBatch, LearRate, alpha, epoch)[0] 
# W = ((w[0] - w[0].min())/(w[0].max() - w[0].min())).T
# W = W.reshape((W.shape[0], 16, 16))
# name = "result"
# W_Dimension = int(np.sqrt(W.shape[0]))

# # Our plot function
# figure, axis = plt.subplots(W_Dimension,W_Dimension, figsize=(W_Dimension, W_Dimension), dpi=160)
# k = 0
# for i in range(W_Dimension):
#     for j in range(W_Dimension):
#         axis[i, j].imshow(W[k], cmap='gray')
#         axis[i, j].axis("off")
#         k += 1
        
# figure.savefig(name + ".png")

# #Question 2
# import h5py
# import numpy as np
# import seaborn as sn
# import sys

# class Q2Answer(object):
#     def init(self, tot, totalLayer = 3, stanDeriv = 0.01):
#         # On that function, I initialize parameters on that function.
#         randSeed = 1907  # This random seed helps us to create random sentences from data
#         W = []
#         b = []

#         # On that for loop, I aimed to initialize weights and biases. As it was given on interim project,
#         # our mean is 0 and our derivation is 0.01.
#         for i in range(totalLayer):

#             # On our weight embedded matrix, it is 1x3 and all of its elements are same. 
#             # Since I made "b" zero, our updated matrix's column elements will reamain same.
#             if  i == 0:
#                 np.random.seed(randSeed)
#                 col = np.random.normal(0, stanDeriv, size=(int(tot[i]/3), tot[i + 1]))
#                 W.append(np.vstack((col, col, col)))
#                 assert W[i].shape == (tot[i], tot[i + 1])
#                 np.random.seed(randSeed)
#                 b.append(np.zeros((1, tot[i + 1])))
#                 continue
#             np.random.seed(randSeed)
#             W.append(np.random.normal(0, stanDeriv, size=(tot[i], tot[i + 1])))
#             np.random.seed(randSeed)
#             b.append(np.random.normal(0, stanDeriv, size=(1, tot[i + 1])))
#             assert W[i].shape == (tot[i], tot[i + 1])
#             assert b[i].shape == (1, tot[i + 1])

#         # This side is for loading class parameters.
#         fifa = {"W": [None] * totalLayer, "b": [None] * totalLayer} # FIFA is one of the sports game which uses momentum a lot.
#         tot = tot
#         pmeters = {"W": W, "b": b}
#         neur = ("sigmoid", "sigmoid", "softmax")
        
#         return fifa,tot,pmeters,neur,randSeed

#     def train(self, inp, lab, valInp, valLab, tot, ratLearn = 0.5, epoch = 100, totalBatch = 100, alpha = 0):

#         fifa,tot,pmeters,neur,seed = self.init(tot,3,0.01)
        
#         totalLayer = 3
#         listTRloss = []
#         listVAloss = []
#         listTRacc = []
#         listVAacc = []

#         epochIter = int(inp.shape[0] / totalBatch)

#         decodedLab = lab
#         inp = self.encodeOne(inp)
#         lab = self.encodeOne(lab)
#         valInp = self.encodeOne(valInp)
#         encodedLabVal = self.encodeOne(valLab)

#         # After that loop, our training will be done.
#         for i in range(epoch):       
#             np.random.seed(seed)
#             perm = np.random.permutation(inp.shape[0])
            
            
#             inp = inp[perm]
#             lab = lab[perm]
#             decodedLab = decodedLab[perm]

#             # To preserve our epoch, we need to make our momentum zero
#             for l in range(totalLayer):
#                 fifa["W"][l] = np.zeros((tot[l], tot[l+1]))
#                 fifa["b"][l] = np.zeros((1, tot[l+1]))

#             # This part for our mini batches interval
#             beg = 0
#             end = totalBatch
#             lossTrain = 0

#             # When our mini batch operation is done, this is the code which training is over
#             for j in range(epochIter):

#                 # We choose on that shuffled data which is mini batch. 
#                 smallInp = inp[beg:end]
#                 smallLab = lab[beg:end]

                
#                 SmalldecodedLab = decodedLab[beg:end]

#                 ls, gradients = self.trainLoss(smallInp, smallLab,pmeters,neur)
#                 lossTrain += ls

#                 # On that loop, gradient descents done for its update
#                 for k in range(totalLayer):

#                     if k == 0: 
#                         fifa["W"][k] = ratLearn * gradients["W"][k] + alpha * fifa["W"][k]
#                         derivCol1, derivCol2, derivCol3 = np.array_split(fifa["W"][k], 3, axis=0)
#                         derivColAvg = (derivCol1 + derivCol2 + derivCol3)/3
#                         pmeters["W"][k] -= np.vstack((derivColAvg, derivColAvg, derivColAvg))
#                         assert pmeters["W"][k].shape == (tot[0], tot[1])
#                         continue

#                     fifa["W"][k] = ratLearn * gradients["W"][k] + alpha * fifa["W"][k]
#                     fifa["b"][k] = ratLearn * gradients["b"][k] + alpha * fifa["b"][k]
#                     pmeters["W"][k] -= fifa["W"][k]
#                     pmeters["b"][k] -= fifa["b"][k]

#                 beg = end
#                 end += totalBatch

#             predictio = self.pred(inp,pmeters,neur)
#             assert predictio.shape == decodedLab.shape
#             accTrain = (predictio == decodedLab).mean() * 100

#             predictio = self.pred(valInp,pmeters,neur)
#             assert predictio.shape == valLab.shape
#             accValue = (predictio == valLab).mean() * 100
#             predictVal = self.pred(valInp,pmeters,neur, False)
#             assert encodedLabVal.shape == predictVal.shape
#             lossValue = np.sum(- encodedLabVal * np.log(predictVal)) / encodedLabVal.shape[0]  # This is cross entropy loss
            
#             print('\r(D, P) = (%d, %d) Train Loss: %f, Value Loss: %f, Train Acc: %f, Value Acc: %f [Current Epoch: %d Total Epoch: %d].'
#             % (tot[1], tot[2], lossTrain/(j+1), lossValue, accTrain, accValue, i + 1, epoch), end='')

#             listTRloss.append(lossTrain/epochIter)
#             listVAloss.append(lossValue)
#             listTRacc.append(accTrain)
#             listVAacc.append(accValue)


#             if i > 5 :
#                 convo = listVAloss[-5:]
#                 convo = sum(convo) / len(convo)

#                 lmt = 0.01
#                 # This place is for making a warning of cross entropy difference
#                 if (convo - lmt) < lossValue < (convo + lmt) and lossValue < 3.35: 
#                     print(" ! Warning ! Training stopped ! Validation Cross Entropy difference with convergence is around 0.01 "
#                           "(Last points' average is +- 0.01).\n")
#                     return {"listTRloss": listTRloss, "listVAloss": listVAloss,
#                             "listTRacc": listTRacc, "listVAacc": listVAacc, "pmeters": pmeters, "neur": neur, }


#         return {"listTRloss": listTRloss, "listVAloss": listVAloss,
#                 "listTRacc": listTRacc, "listVAacc": listVAacc,"pmeters":pmeters,"neur":neur}

#     def trainLoss(self, inp, lab, pmeters,neur):
#         totalLayer = 3
#         W = pmeters["W"]
#         b = pmeters["b"]
#         neur = neur
#         listOut = [inp]
#         deriv = [1]
#         totalBatch = inp.shape[0]

#         # This code makes forward propagation.
#         variab = inp @ W[0] + b[0]
#         variab1, deriv1 = self.activation(neur[0], variab)
#         variab2 = variab1 @ W[1] + b[1]
#         variab21, deriv21 = self.activation(neur[1], variab2)
#         variab3 = variab21 @ W[2] + b[2]
#         variab32, deriv32 = self.activation(neur[2], variab3)
#         listOut.append(variab1)
#         listOut.append(variab21)
#         listOut.append(variab32)
#         deriv.append(deriv1)
#         deriv.append(deriv21)
#         deriv.append(deriv32)

#         # This code makes loss calculation
#         predictio = listOut[-1]
#         ls = np.sum(-lab * np.log(predictio)) / lab.shape[0]  # This code is here for to calculate cross entropy loss.
#         dlt = predictio
#         dlt[lab == 1] -= 1
#         dlt = dlt / totalBatch

#         # Lets compute grad functions.
#         derivW = []
#         derivb = []
#         ones = np.ones((1, totalBatch))

#         # We added delta to calculate backward propagation.
#         for i in reversed(range(totalLayer)):
#             derivW.append(listOut[i].T @ dlt)
#             derivb.append(ones @ dlt)
#             dlt = deriv[i] * (dlt @ W[i].T)

#         gradients = {'W': derivW[::-1], 'b': derivb[::-1]}

#         return ls, gradients

#     def pred(self, inp,pmeters,neur, classify=True):
#         W = pmeters["W"]
#         b = pmeters["b"]
#         neur = neur
#         listOut = [inp]
#         totalLayer = 3

#         # On that side, our prediction function happens. 
#         variab = inp @ W[0] + b[0]
#         variab1 = self.activation(neur[0], variab)[0]
#         variab2 = variab1 @ W[1] + b[1]
#         variab21 = self.activation(neur[1], variab2)[0]
#         variab3 = variab21 @ W[2] + b[2]
#         variab32 = self.activation(neur[2], variab3)[0]
#         variabb = (np.argmax(variab32, axis=1) + 1).T
#         variabb = np.reshape(variabb, (variabb.shape[0], 1))
#         return variabb if classify is True else variab32

#     def encodeOne(self, inp, tot=250):
#         inpEncode = np.zeros((inp.shape[0], 0))

#         for i in range(inp.shape[1]):
#             temp = np.zeros((inp.shape[0], tot))
#             tempp = np.arange(inp.shape[0])
#             temp[tempp, (inp-1)[:, i]] = 1
#             inpEncode = np.hstack((inpEncode, temp))

#         return inpEncode

#     def activation(self, neur, inp):
#         if neur == "sigmoid":
#             act = 1 / (1 + np.exp(-inp))
#             der = act * (1 - act)
#             return act, der
#         if neur == "softmax":
#             act = np.exp(inp) / np.sum(np.exp(inp), axis= 1, keepdims=True)
#             der = None
#             return act, der
#         return None

# h5data = "data2.h5"
# h5 = h5py.File(h5data, 'r')
# words = h5['words'][()]
# trainx = h5['trainx'][()]
# traind = h5['traind'][()]
# valx = h5['valx'][()]
# vald = h5['vald'][()]
# testx = h5['testx'][()]
# testd = h5['testd'][()]
# h5.close()
    
# traind = np.reshape(traind, (traind.shape[0], 1))
# vald = np.reshape(vald, (vald.shape[0], 1))
# testd = np.reshape(testd, (testd.shape[0], 1))
# words = np.reshape(words, (words.shape[0], 1))

# ratLearn = 0.15
# alpha = 0.85
# epoch = 50
# totalBatch = 200

# D = 8
# P = 64
# layerHid = [D, P]

# tot = [750]
# tot += layerHid
# tot.append(250)

# totalLayer = len(tot) - 1

# network = Q2Answer()
# info = network.train(trainx, traind, valx, vald,tot, ratLearn, epoch, totalBatch, alpha)
# listTRloss, listVAloss, listTRacc, listVAacc,pmeters,neur = info.values()
# print()
# classTestPred = network.pred(network.encodeOne(testx),pmeters,neur)
# print("\n\nTest accuracy for (D, P) = (8, 64): ", (classTestPred == testd).mean() * 100)

# # 16, 128
# D = 16
# P = 128
# layerHid = [D, P]

# tot = [750]
# tot += layerHid
# tot.append(250)

# network = Q2Answer()
# info = network.train(trainx, traind, valx, vald,tot, ratLearn, epoch, totalBatch, alpha)
# listTRloss, listVAloss, listTRacc, listVAacc,pmeters,neur = info.values()
# print()
# classTestPred = network.pred(network.encodeOne(testx),pmeters,neur)
# print("\n\nTest accuracy for (D, P) = (16, 128): ", (classTestPred == testd).mean() * 100)

# # 32, 256
# D = 32
# P = 256

# layerHid = [D, P]

# tot = [750]
# tot += layerHid
# tot.append(250)

# network = Q2Answer()
# info = network.train(trainx, traind, valx, vald,tot, ratLearn, epoch, totalBatch, alpha)
# listTRloss, listVAloss, listTRacc, listVAacc,pmeters,neur = info.values()
# print()

# classTestPred = network.pred(network.encodeOne(testx),pmeters,neur)
# print("\n\nTest accuracy for (D, P) = (32, 256): ", (classTestPred == testd).mean() * 100)

# w = 5  # Total Prediction

# np.random.seed(1907)
# p = np.random.permutation(testx.shape[0])
# testx = testx[p][:w]
# testd = testd[p][:w]

# testx_e = network.encodeOne(testx)
# test_pred = network.pred(testx_e,pmeters,neur, False)

# n = 10
# s = (np.argsort(-test_pred, axis=1) + 1)[:, :n]

# for i in range(w):
#         print("\n--------------")
#         print("Sentence:", words[testx[i][0] - 1], words[testx[i][1] - 1], words[testx[i][2] - 1])
#         print("Label:", words[testd[i] - 1])
#         for j in range(n):
#             print(str(j + 1) + ". ", words[s[i][j] - 1])
