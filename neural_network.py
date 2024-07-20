import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def draw_num(example):
    ex1 = []
    for i in range(28):
        ex1.append([])
        for j in range(28):
            ex1[i].append(example[i*28 + j])

    for i in range(len(ex1)):
        for j in range(len(ex1[0])):
            print(str(ex1[i][j]).rjust(5),end=" ")
        print()

data = pd.read_csv('NN\mnist_train.csv')
data = np.array(data)

m, n = data.shape

#Transpose matrix - every column is new number example, every row is individual pixel of example
data = data.T
labels = data[0]
images = data[1:n]
images = images/255

correct_guesses = 0

def init_values():
    # w = weight, b = bias, i = input layer, h = hidden layer, o = output layer
    wih = np.random.uniform(-0.5, 0.5, (20, 784))
    who = np.random.uniform(-0.5, 0.5, (10, 20))
    bih = np.zeros((20, 1))
    bho = np.zeros((10, 1))
    return wih, who, bih, bho

def ReLU(neuron):
    # x; if x > 0
    # 0; if x < 0
    return np.maximum(neuron,0)

def ReLU_derivative(neuron):
    return neuron > 0

def softmax(neuron):
    #funky math probability distribution function: makes all values range from 0 - 1
    activated = np.exp(neuron) / sum(np.exp(neuron))
    return activated

def label_to_matrix(labels):
    #makes matrix of length 10 containing nines 0s and one 1, cooresponding to the labels index
    #ex. if label = 3,
    #       matrix = [0,0,1,0,0,0,0,0,0,0]
    matrix = np.zeros((labels.size, 10))
    matrix[np.arange(labels.size),labels] = 1
    return matrix.T

def forward_prop(wih, who, bih, bho, images):
    #forward prop: input --> hidden
    hidden = wih.dot(images) + bih
    hidden_activated = ReLU(hidden)

    #forward prop: hidden --> output
    output = who.dot(hidden_activated) + bho
    output_activated = softmax(output)
    return hidden, hidden_activated, output, output_activated

#not needed
def error(output, labels, correct_guesses):
    #finds the average error for each output node by averaging the squares between the differences between outputed and expected node values
    cost = np.sum((output-labels)**2, axis = 0)/10

    matrixed_labels = label_to_matrix(labels)
    #*argmax returns index of highest valued element*
    correct_guesses += int(np.argmax(output) == np.argmax(matrixed_labels))

def back_prop(hidden, hidden_activated, output, output_activated, wih, who, bih, bho, images, labels, learn_rate):
    matrixed_labels = label_to_matrix(labels)

    #backward prop: output --> hidden
    delta_output = output_activated - matrixed_labels
    who += -learn_rate * (1 / m * delta_output.dot(hidden_activated.T))
    bho += -learn_rate * (1 / m * np.sum(delta_output))

    #backward prop: hidden --> input
    delta_hidden = who.T.dot(delta_output) * ReLU_derivative(hidden)
    wih += -learn_rate * (1 / m * delta_hidden.dot(images.T))
    bih += -learn_rate * (1 / m * np.sum(delta_hidden))

    return who, bho, wih, bih

def get_predictions(output_activated):
    return np.argmax(output_activated, 0)

def get_accuracy(predictions, labels):
    print(predictions, labels)
    return np.sum(predictions == labels) / labels.size

def run_network(images, labels, learn_rate, iterations):
    wih, who, bih, bho = init_values()
    for i in range(iterations+1):
        hidden, hidden_activated, output, output_activated = forward_prop(wih, who, bih, bho, images)
        who, bho, wih, bih = back_prop(hidden, hidden_activated, output, output_activated, wih, who, bih, bho, images, labels, learn_rate)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(output_activated)
            print(get_accuracy(predictions, labels)*100)
    return wih, who, bih, bho

wih, who, bih, bho = run_network(images, labels, 0.1, 500)

# while True:
#     idx = int(input("Enter a number (0 - 59999): "))
#     imagesT = images.T
#     img = imagesT[idx]
#     plt.imshow(img.reshape(28, 28), cmap="Greys")

#     # Forward propagation input -> hidden
#     h_pre = bih + wih @ img.reshape(784, 1)
#     h = ReLU(h_pre)
#     # Forward propagation hidden -> output
#     o_pre = bho + who @ h
#     o = softmax(o_pre)

#     plt.title(f"Prediction: {o.argmax()}")
#     plt.show()
