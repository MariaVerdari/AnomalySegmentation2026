# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib
import time

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable          
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from erfnet import ERFNet
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry

NUM_CHANNELS = 3
NUM_CLASSES = 20

# blocco aggiunto per le etichette sfasate 

'''
class MapToTrainIds(object): # vecchio metodo per mappare, sostituito da quello più efficiente
    def __init__(self):
        mapping = {
            0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255,
            10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255,
            19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
            29: 255, 30: 255, 31: 16, 32: 17, 33: 18, -1: 255
        }
        self.lut = np.zeros(256, dtype=np.uint8)
        for k, v in mapping.items():
            if k >= 0:
                self.lut[k] = v

    def __call__(self, img):
        img_np = np.array(img)
        mapped_np = self.lut[img_np]
        return Image.fromarray(mapped_np)
'''
    
class MapToTrainIds(object): 
    def __init__(self):
        # Inizializziamo tutto a 19 (il valore void dell'ERFNet)
        self.lut = np.full(256, 19, dtype=np.uint8) 
        
        # Mappiamo SOLO i labelId originali di Cityscapes nei rispettivi trainId
        train_mapping = {
            7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
            21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
            28: 15, 31: 16, 32: 17, 33: 18
        }
        for lid, tid in train_mapping.items():
            self.lut[lid] = tid

    def __call__(self, img):
        img_np = np.array(img)
        mapped_np = self.lut[img_np]
        return Image.fromarray(mapped_np)
    


image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor()
    #Normalize([.485, .456, .406], [.229, .224, .225])
])
target_transform_cityscapes = Compose([
    Resize(512, Image.NEAREST),
    MapToTrainIds(),
    ToLabel(),
    #Relabel(255, 19),   #ignore label to 19
])

def main(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    #model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")


    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")


    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)


    iouEvalVal = iouEval(NUM_CLASSES)
    iouEvalVal_05 = iouEval(NUM_CLASSES)
    iouEvalVal_075 = iouEval(NUM_CLASSES)
    iouEvalVal_11 = iouEval(NUM_CLASSES)

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):  #ciclo di elaborazione dei batch
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)
        with torch.no_grad():
            outputs = model(inputs) #logits, forward pass, non serve softmax
            #temperature scaling
            temperature = [0.5,0.75,1.1]
            scaled_result_05 = outputs/ temperature[0]
            scaled_result_075 = outputs / temperature[1]
            scaled_result_11 = outputs / temperature[2]

        iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, labels) # aggiunge previsione finale e soluzione vera per aggiornare matrice di confusione
        iouEvalVal_05.addBatch(scaled_result_05.max(1)[1].unsqueeze(1).data, labels)
        iouEvalVal_075.addBatch(scaled_result_075.max(1)[1].unsqueeze(1).data, labels)
        iouEvalVal_11.addBatch(scaled_result_11.max(1)[1].unsqueeze(1).data, labels)
   

        filenameSave = filename[0].split("leftImg8bit/")[1]

        print (step, filenameSave)


    iouVal, iou_classes = iouEvalVal.getIoU() # calcola iou da matrice di confusione
    iouVal_05, iou_classes_05 = iouEvalVal_05.getIoU()
    iouVal_075, iou_classes_075 = iouEvalVal_075.getIoU()
    iouVal_11, iou_classes_11 = iouEvalVal_11.getIoU()

    #formatta iou per stampa
    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    iou_classes_str_05 = []
    for i in range(iou_classes_05.size(0)):
        iouStr = getColorEntry(iou_classes_05[i])+'{:0.2f}'.format(iou_classes_05[i]*100) + '\033[0m'
        iou_classes_str_05.append(iouStr)

    iou_classes_str_075 = []
    for i in range(iou_classes_075.size(0)):
        iouStr = getColorEntry(iou_classes_075[i])+'{:0.2f}'.format(iou_classes_075[i]*100) + '\033[0m'
        iou_classes_str_075.append(iouStr)

    iou_classes_str_11 = []
    for i in range(iou_classes_11.size(0)):
        iouStr = getColorEntry(iou_classes_11[i])+'{:0.2f}'.format(iou_classes_11[i]*100) + '\033[0m'
        iou_classes_str_11.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")
    #print("TOTAL IOU: ", iou * 100, "%")
    '''
    print("Per-Class IoU:")
    print(iou_classes_str[0], "Road")
    print(iou_classes_str[1], "sidewalk")
    print(iou_classes_str[2], "building")
    print(iou_classes_str[3], "wall")
    print(iou_classes_str[4], "fence")
    print(iou_classes_str[5], "pole")
    print(iou_classes_str[6], "traffic light")
    print(iou_classes_str[7], "traffic sign")
    print(iou_classes_str[8], "vegetation")
    print(iou_classes_str[9], "terrain")
    print(iou_classes_str[10], "sky")
    print(iou_classes_str[11], "person")
    print(iou_classes_str[12], "rider")
    print(iou_classes_str[13], "car")
    print(iou_classes_str[14], "truck")
    print(iou_classes_str[15], "bus")
    print(iou_classes_str[16], "train")
    print(iou_classes_str[17], "motorcycle")
    print(iou_classes_str[18], "bicycle")
    print("=======================================")
    '''
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")

    iouStr_05 = getColorEntry(iouVal_05)+'{:0.2f}'.format(iouVal_05*100) + '\033[0m'
    print ("MEAN IoU (0.5): ", iouStr_05, "%")

    iouStr_075 = getColorEntry(iouVal_075)+'{:0.2f}'.format(iouVal_075*100) + '\033[0m'
    print ("MEAN IoU (0.75): ", iouStr_075, "%")

    iouStr_11 = getColorEntry(iouVal_11)+'{:0.2f}'.format(iouVal_11*100) + '\033[0m'
    print ("MEAN IoU (1.1): ", iouStr_11, "%")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())


