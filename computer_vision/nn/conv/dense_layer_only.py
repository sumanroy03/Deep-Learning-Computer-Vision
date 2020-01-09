from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import backend as K


from keras import optimizers

class DenseLayerOnly:
    @staticmethod
    def build(width, height, depth, classes, vgg_model):
        input_shape = vgg_model.output_shape[1]
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model = Sequential()
        model.add(InputLayer(input_shape=(input_shape,)))
        model.add(Dense(512, activation='relu', input_dim=input_shape))
        model.add(Dropout(0.3))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(classes, activation='sigmoid'))
        # return the constructed network architecture
        return model