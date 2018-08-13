import numpy as np
from pybrain.datasets           import SupervisedDataSet
from pybrain.tools.shortcuts    import buildNetwork
from pybrain.supervised         import BackpropTrainer

entrada = np.genfromtxt("DataSets/irisdataset.data", delimiter=",",usecols=(0,1,2,3))
saida = np.genfromtxt("DataSets/irisdataset.data", delimiter=",",usecols=(4))
'''
Iris-setosa = 0
iris-versicolor = 1
iris-virginica = 2
'''

entrada_treino = np.concatenate((entrada[:35], entrada[50:85], entrada[100:135]))
saida_treino = np.concatenate((saida[:35], saida[50:85], saida[100:135]))

entrada_teste = np.concatenate((entrada[35:50], entrada[85:100], entrada[135:]))
saida_teste = np.concatenate((saida[35:50], saida[85:100], saida[135:]))

treinamento = SupervisedDataSet(4,1)
for i in range(len(entrada_treino)):
    treinamento.addSample(entrada_treino[i], saida_treino[i])
# print(len(treinamento))
# print(treinamento.indim)
# print(treinamento.outdim)

# Construindo rede
rede    = buildNetwork(treinamento.indim, 2, treinamento.outdim, bias=True)
trainer = BackpropTrainer(rede, treinamento, learningrate=0.03, momentum=0.3)

# Treinando a rede
for epoch in range(1000):
    trainer.train()

# Testando a rede
teste = SupervisedDataSet(4,1)
for i in range(len(entrada_teste)):
    teste.addSample(entrada_teste[i], saida_teste[i])
trainer.testOnData(teste, verbose=True)
