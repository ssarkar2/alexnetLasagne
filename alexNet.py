#alexnet with lasagne
import lasagne
import os
import numpy as np
import yaml
import theano.tensor as T
import theano

class alexNet():
    def __init__(self, config, inputVar):
        #This class is designed only for the conv layers of alexnet (ie to extract alexnt features)
        #the 4d blob is: batch, channels, rows, cols (bc01?)
        self.config = config
        batch_size = config['batch_size']
        self.numLayers = config['numLayers']
        initWeights = config['initWeights']
        if initWeights:
            self.weightsDir = config['weightsDir']
            self.weightFileTag = config['weightFileTag']
        imgRow = config['imgRow']
        imgCol = config['imgCol']
        self.layers = []

        # parameters describing structure of alexnet
        self.numGroups = [1,2,1,2,2]
        self.numFeatureMaps = [96,256,384,384,256]
        self.convKernelSize = [11,5,3,3,3]
        self.convStride = [8,1,1,1,1]  #note its 8 instead of 4
        self.poolKernelSize = [3,3,-1,-1,3]
        self.poolStride = [2,2,-1,-1,2]
        self.useLRN = [True,True,False,False,False]

        meanVal = np.load(config['mean_file']).astype('float32')
        inp = lasagne.layers.InputLayer(shape=(None,3,imgRow,imgCol), input_var=inputVar)
        #using code from standardize layer of lasagne
        inp = lasagne.layers.BiasLayer(inp, theano.shared(-meanVal), shared_axes=0)
        inp.params[inp.b].remove('trainable') # Do not optimize the offset parameter

        layer = inp
        for layerNum in range(self.numLayers):
            layer = self.createConvLayer(layer, layerNum)
            #print lasagne.layers.get_output_shape(layer), 'ddd'
            self.layers.append(layer)
        self.outLayer = self.layers[-1]  #the last layer is the output layer


    def createConvLayer(self, inputLayer, layerNum):
        def createConvLayerForSingleGroup(inp, numFeatureMaps, convKernelSize, convStride, weights, useLRN, poolKernelSize, poolStride):
            layerOut = lasagne.layers.Conv2DLayer(incoming=inp, num_filters=numFeatureMaps, filter_size=(convKernelSize,)*2, stride=convStride, W=weights[0], b=weights[1], nonlinearity=lasagne.nonlinearities.rectify, pad='same')
            if useLRN:
                layerOut = lasagne.layers.LocalResponseNormalization2DLayer(layerOut, alpha=0.0001, k=2, beta=0.75, n=5)
            if poolKernelSize > 0:
                layerOut = lasagne.layers.MaxPool2DLayer(layerOut, pool_size=(poolKernelSize,)*2, stride=poolStride)
            return layerOut

        weights = self.getParams(layerNum)
        groups = self.numGroups[layerNum]
        numFeatureMaps = self.numFeatureMaps[layerNum]; convKernelSize = self.convKernelSize[layerNum]; convStride = self.convStride[layerNum]; useLRN = self.useLRN[layerNum]; poolKernelSize = self.poolKernelSize[layerNum]; poolStride = self.poolStride[layerNum]
        if groups == 1:
            layerOut = createConvLayerForSingleGroup(inputLayer, numFeatureMaps, convKernelSize, convStride, weights, useLRN, poolKernelSize, poolStride)
        else:
            splitPoint = self.numFeatureMaps[layerNum-1]/groups
            slice0 = lasagne.layers.SliceLayer(inputLayer, indices = slice(0, splitPoint), axis=1)
            slice1 = lasagne.layers.SliceLayer(inputLayer, indices = slice(splitPoint, None), axis=1)
            layerOut0 = createConvLayerForSingleGroup(slice0, numFeatureMaps/2, convKernelSize, convStride, weights[0:2], useLRN, poolKernelSize, poolStride)
            layerOut1 = createConvLayerForSingleGroup(slice1, numFeatureMaps/2, convKernelSize, convStride, weights[2:], useLRN, poolKernelSize, poolStride)
            layerOut = lasagne.layers.ConcatLayer([layerOut0, layerOut1], axis=1)
        return layerOut


    def getParams(self, layerNum):
        retVals = []
        groups = self.numGroups[layerNum]
        for group in range(groups):
            fileName = self.weightsDir + 'W' + ('',str(group))[groups > 1] + '_' + str(layerNum)  + self.weightFileTag + '.npy'
            if os.path.exists(fileName):
                W = np.cast['float32'](np.load(fileName))
                #print W.shape, 'ddd'
                #W is in shape: i01o (inp, row,col, output maps)
                W = np.rollaxis(W, 3)  #converts it to oi01
                #print W.shape, 'ccc'
                retVals += [lasagne.utils.create_param(W, W.shape, name='W_'+str(layerNum) + '_' + str(group))]
            else:
                print 'init weight ( '+fileName+ ' )not found. init-ing randomly'; retVals += [lasagne.init.GlorotUniform()]    #randomly initialized params do not have names (unlike the read-from-file weights in the if case above). can they be given names?

            fileName = self.weightsDir + 'b' + ('',str(group))[groups > 1] + '_' + str(layerNum) + self.weightFileTag + '.npy'
            if os.path.exists(fileName):
                b = np.cast['float32'](np.load(fileName))
                retVals += [lasagne.utils.create_param(b, b.shape, name='b_'+str(layerNum) + '_' + str(group))]
            else:
                print 'init weight ( '+fileName+ ' )not found. init-ing randomly'; retVals += [lasagne.init.Constant(0.)]
        return retVals

'''
#usage
x = T.tensor4('x')
an = alexNet(yaml.load(open('tempConfig.yaml', 'r')), x)
print lasagne.layers.get_all_params(an.outLayer)
'''
