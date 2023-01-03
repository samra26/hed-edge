import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os


class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]

def hed_edge(input_image,prototxt,caffemodel):
    # Load the model.
    #prototxt='deploy.prototxt'
    #caffemodel='hed_pretrained_bsds.caffemodel'
    net = cv.dnn.readNetFromCaffe(prototxt, caffemodel)
    
    scale_percent = 85
    name=input_image.split()[0]
    
    image=cv.imread(input_image)
    
    '''width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image=cv.resize(image,(width,height))'''
    width=image.shape[1]
    height=image.shape[0] 
    inp = cv.dnn.blobFromImage(image, scalefactor=1.0, size=(width, height),
                           mean=(104.00698793, 116.66876762, 122.67891434),
                           swapRB=False, crop=False)
    net.setInput(inp)
    # edges = cv.Canny(image,image.shape[1],image.shape[0])
    out = net.forward()

    out = out[0, 0]
    out = cv.resize(out, (image.shape[1], image.shape[0]))

    print(out.shape)
    #out=cv.cvtColor(out,cv.COLOR_GRAY2BGR)
    out = 255 * out
    out = out.astype(np.uint8)
    print(input_image)
    #print(time.time())
    test_folder='edge'
    #filename = os.path.join(input_image, str(time.time()) + '_edge.png')
    #cv.imwrite(filename,out)
    filename=os.path.join(test_folder, name[:-4] + '_edge.png')
    out=cv.imwrite(filename,out)
    print(out)

    #cv.dnn_registerLayer('Crop', CropLayer)
    return out
