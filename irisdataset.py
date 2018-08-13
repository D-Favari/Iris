from sklearn                         import datasets
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.supervised.trainers     import BackpropTrainer
from pybrain.tools.shortcuts         import buildNetwork
import matplotlib.pyplot             as plt

iris = datasets.load_iris()
x, y = iris.data, iris.target

dataset = ClassificationDataSet(4, 1, nb_classes=3)

# add amostra
for i in range(len(x)):
    dataset.addSample(x[i], y[i])

# dados para treinamento
train_data, part_data = dataset.splitWithProportion(0.6)
print("Quantidade para treino: %d" %len(train_data))

# teste e validação
test_data, val_data = part_data.splitWithProportion(0.5)
print("Quantidade para teste: %d" %len(test_data))
print("Quantidade para validação: %d" %len(val_data))

net     = buildNetwork(dataset.indim, 3, dataset.outdim)
trainer = BackpropTrainer(net, dataset=train_data, learningrate=0.01, momentum=0.1, verbose=True)

train_errors, val_errors = trainer.trainUntilConvergence(dataset=train_data,maxEpochs=1000)

plt.plot(train_errors, 'b', val_errors, 'r')
plt.show()