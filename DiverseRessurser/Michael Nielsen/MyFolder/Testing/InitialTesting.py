import numpy as np
import mnist_loader
import gzip
import pickle

klubb = np.random.randn(3, 1)
print(klubb)

jabbert = np.random.randn(3, 2)
print(jabbert)


W = np.array([[1, 2], [3, 4], [5, 6]])
a = np.array([0.5, 0.6])
b = np.array([0.1, 0.2, 0.3])

print(np.dot(W, a))
print(np.dot(W, a) + b)

mnist_loader.load_data()


fre = [[2, 3, 4], [1, 2, 3]]
gre = [np.reshape(el, (3, 1)) for el in fre]
print(gre)
e = np.zeros((10, 1))
print(e)
e[4] = 1.0
print(e)


glogg = [[0], [1], [2], [3], [4]]
print(glogg)
glogg[4] = 5
print(glogg)

huldra = np.array([2, 3, 4, 5, 6]).reshape(5, 1)
harald = np.array([2, 3, 4, 5, 6])[:, None]
hratvar = np.array([2, 3, 4, 5, 6])[None, :]

 
print(huldra)
print(harald)
print(hratvar)

f = gzip.open('data/mnist.pkl.gz', 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding = 'latin1')
f.close()

# with open ("flubb.txt", 'w') as f:
#     for i in range(len(training_data[0])):
#         f.write( str(training_data[0][i]) + ' ' + str(training_data[1][i]) + '\n')


### Where on earth are we getting the validation results from?
### I mean the targets for the test- and validation-data, where is that info?

frobbo = (np.array([[1, 2, 5], [1, 7, 9]]), np.array([5, 9]))
print(frobbo)

for frob in frobbo[0]:
    frob[:, None]

gorgo = ([frob[:, None] for frob in frobbo[0]], frobbo[1][None, :])
gragg = np.dot(gorgo[0], gorgo[1])
print("Frobbo")
print(gorgo[0])
print(gragg)

gogn = np.random.randn(3, 2)
gango = np.array([3, 1])[:, None]
print(gogn)
gaggy = np.dot(gogn, gango)
print(gaggy)


glogg = ([2, 3, 4, 5], [[2, 3], [1, 2]])

for grob, gand in zip(glogg[0], glogg[1]):
    print(grob)
    print(gand)


test_results = [(1, 2), (1, 1), (2, 2), (8, 1), (9, 2)]
polk = sum(x == y for x, y in test_results)
dolk = sum(i[0] == i[1] for i in test_results)
print(polk)
print(dolk)





