import yaml
import theano, lasagne
import theano.tensor as T
from alexNet import alexNet
import numpy as np
from lasagne.regularization import regularize_layer_params_weighted, l2

def initNetwork(X, Y, config):
    alexNetModel = alexNet2(config, X)
    #network = lasagne.layers.FlattenLayer(alexNetModel.outLayer)
    network = lasagne.layers.DropoutLayer(alexNetModel.outLayer, p=config['prob_drop'], rescale=False)  #dropout
    wtFileName = config['weightsDir'] + 'W_5.npy'; bFileName = config['weightsDir'] + 'b_5.npy'
    network = lasagne.layers.DenseLayer(network, num_units=31, W=getClassifierParam(wtFileName, False), b=getClassifierParam(bFileName, True), nonlinearity=lasagne.nonlinearities.softmax)  #if classifier weights are not present, init with random weights

    regMult = [float(i) for i in config['regularize'].split(',')]    #read off a line like :regularize: 0.1,0.1,0.1,0.1,0.1,0.1 from the config.yaml file
    layersRegMultiplier = {alexNetModel.layers[layerId]:regMult[layerId] for layerId in range(len(alexNetModel.layers))}
    layersRegMultiplier[network] = regMult[-1]
    l2_penalty = regularize_layer_params_weighted(layersRegMultiplier, l2)

    prediction = lasagne.layers.get_output(network, deterministic=True)
    lossAll = lasagne.objectives.categorical_crossentropy(prediction, Y)  #loss function
    loss = lossAll.mean()
    loss = loss + l2_penalty

    accuracy = T.mean(T.eq(T.argmax(prediction, axis=1), Y), dtype=theano.config.floatX)
    match = T.eq(T.argmax(prediction, axis=1), Y)
    params = lasagne.layers.get_all_params(network, trainable=True)
    return [loss, params, accuracy, match]
    
    
    
configFile = 'someConfigFile.yaml'
with open(configFile, 'r') as f:
    config = yaml.load(f)
X = T.ftensor4('X')  #batch, channel, width/height
Y = T.vector('Y', dtype='int64')  #batch   (categoryId for each image)
loss, params, accuracy, match = initNetwork(X, Y, config)
lrMultipliers = [float(i) for i in config['lrMultipliers'].split(',')]  #learning rate multiplier for each layer

#test function
test_fn = theano.function([X, Y], [loss, accuracy, match], allow_input_downcast=True)

#training function
grads = theano.grad(loss, params)
for idx, param in enumerate(params):
  gradScale = lrMultipliers[idx] #obtain multiplier for that param
  if gradScale != 1:
      grads[idx] *= gradScale
updates = lasagne.updates.nesterov_momentum(grads, params, lr, momentum)
train_fn = theano.function([X, Y], [loss, acc], updates=updates, allow_input_downcast=True)
