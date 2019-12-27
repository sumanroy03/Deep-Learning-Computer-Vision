# import the necessary packages
import comet_ml
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from computer_vision.nn.conv import shallow_net
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
# from comet_ml import Experiment

# Hyper parameters
epochs = 40
learning_rate = 0.001
training_batch_size = 32
predict_batch_size = 32
decay_rate = learning_rate / epochs
momentum = 0.8
filters = 32
# load the training and testing data, then scale it into the
# range [0, 1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
# Create an experiment
experiment = comet_ml.Experiment(api_key="ISHj0sNayO56qDKxxgFPFAsJK", project_name="general", workspace="sumanroy03")
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
"dog", "frog", "horse", "ship", "truck"]

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=learning_rate)
# opt = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
model = shallow_net.ShallowNet.build(width=32, height=32, depth=3, classes=10, filters=filters)
model.compile(loss="categorical_crossentropy", optimizer=opt,
metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
batch_size=training_batch_size, epochs=epochs, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=predict_batch_size)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=labelNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()