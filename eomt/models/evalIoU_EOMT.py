# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

### abbiamo usato ioueval

from eomt.models import vit
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
#from models.vit import ViT
#from models.eomt import EoMT
from vit import ViT
from eomt import EoMT   
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry

NUM_CHANNELS = 3
NUM_CLASSES = 20 # 19 classi valide + 1 classe ignore (void)

image_transform = ToPILImage()

# Impostiamo la risoluzione fissa 512x1024 richiesta dall'encoder
input_transform_cityscapes = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),
])
target_transform_cityscapes = Compose([
    Resize((512, 1024), Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   # Mappa il void standard di Cityscapes (255) a 19
])

def main(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    # 1. COSTRUISCO L'ENCODER E IL MODELLO EOMT
    print("Inizializzazione del modello EoMT...")
    encoder = ViT(
        img_size=(512, 1024),
        patch_size=16,
        backbone_name="vit_base_patch14_reg4_dinov2"
    )
    
    model = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES, 
        num_q=100,
        num_blocks=3
    )

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    # 2. CARICO I PESI (Con la funzione per interpolare il pos_embed)
    weightspath = args.loadDir + args.loadWeights
    print(f"Caricamento pesi da: {weightspath}")
    
    def load_my_state_dict(model, state_dict):  
        own_state = model.state_dict()
        for name, param in state_dict.items(): 
            clean_name = name
            if clean_name.startswith("network."):
                clean_name = clean_name.replace("network.", "")
            if clean_name.startswith("module."):
                clean_name = clean_name.replace("module.", "")
                
            if clean_name not in own_state:
                continue
            else:
                if own_state[clean_name].shape == param.shape:
                    own_state[clean_name].copy_(param)
                elif "pos_embed" in clean_name:
                    # Interpolazione dinamica del pos_embed
                    dim = param.shape[-1]
                    orig_size = int(param.shape[1] ** 0.5) 
                    
                    H_new = 512 // 16 # Usa 14 se hai impostato patch_size=14 nell'encoder
                    W_new = 1024 // 16 
                    
                    param_reshaped = param.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
                    param_interpolated = F.interpolate(param_reshaped, size=(H_new, W_new), mode='bilinear', align_corners=False)
                    param_final = param_interpolated.permute(0, 2, 3, 1).reshape(1, -1, dim)
                    
                    own_state[clean_name].copy_(param_final)
                    print(f"✅ pos_embed interpolato con successo!")
        return model

    # Carica i pesi originali dal file
    state_dict = torch.load(weightspath, map_location="cpu", weights_only=True)
    
    # Applica la nostra funzione di pulizia e interpolazione
    model = load_my_state_dict(model, state_dict)
    print("Model and weights LOADED successfully")

    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")


    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)


    iouEvalVal = iouEval(NUM_CLASSES)
    iouEvalVal_05 = iouEval(NUM_CLASSES)
    iouEvalVal_075 = iouEval(NUM_CLASSES)
    iouEvalVal_11 = iouEval(NUM_CLASSES)


    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)
        
        with torch.no_grad():
            # 1. Output grezzo dell'EoMT
            mask_logits_list, class_logits_list = model(inputs)
            
            mask_logits = mask_logits_list[-1]
            class_logits = class_logits_list[-1]

            # 2. Trasformo in probabilità
            mask_probs = torch.sigmoid(mask_logits)
            class_probs = torch.softmax(class_logits, dim=-1)[:, :, :-1] # Tolgo il Void delle queries

            # 3. Ricostruisco la mappa di segmentazione classica
            result_probs = torch.einsum("bqc, bqhw -> bchw", class_probs, mask_probs)
            result_probs = F.interpolate(result_probs, size=(512, 1024), mode="bilinear", align_corners=False)
            
            # 4. Trovo la classe vincente per ogni pixel!
            # .max(1)[1] significa: guarda l'asse delle classi (1) e prendi l'indice [1] del valore massimo
            predicted_classes = result_probs.max(1)[1].unsqueeze(1).data
            
            #temperature sclaing
            temperature = [0.5,0.75,1.1]
            scaled_result = result_probs / temperature[0]
            predicted_classes_05 = scaled_result.max(1)[1].unsqueeze(1).data
            scaled_result = result_probs / temperature[1]
            predicted_classes_075 = scaled_result.max(1)[1].unsqueeze(1).data
            scaled_result = result_probs / temperature[2]
            predicted_classes_11 = scaled_result.max(1)[1].unsqueeze(1).data 

        # Passo le predizioni e le maschere ground truth al valutatore IoU
        iouEvalVal.addBatch(predicted_classes, labels)
        iouEvalVal_05.addBatch(predicted_classes_05, labels)
        iouEvalVal_075.addBatch(predicted_classes_075, labels)
        iouEvalVal_11.addBatch(predicted_classes_11, labels)

        filenameSave = filename[0].split("leftImg8bit/")[1] 
        print (step, filenameSave)
        
        


    iouVal, iou_classes = iouEvalVal.getIoU()
    iouVal_05, iou_classes_05 = iouEvalVal_05.getIoU()
    iouVal_075, iou_classes_075 = iouEvalVal_075.getIoU()
    iouVal_11, iou_classes_11 = iouEvalVal_11.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    iou_classes_05_str = []
    for i in range(iou_classes_05.size(0)):
        iouStr = getColorEntry(iou_classes_05[i])+'{:0.2f}'.format(iou_classes_05[i]*100) + '\033[0m'
        iou_classes_05_str.append(iouStr)

    iou_classes_075_str = []
    for i in range(iou_classes_075.size(0)):
        iouStr = getColorEntry(iou_classes_075[i])+'{:0.2f}'.format(iou_classes_075[i]*100) + '\033[0m'
        iou_classes_075_str.append(iouStr)

    iou_classes_11_str = []
    for i in range(iou_classes_11.size(0)):
        iouStr = getColorEntry(iou_classes_11[i])+'{:0.2f}'.format(iou_classes_11[i]*100) + '\033[0m'
        iou_classes_11_str.append(iouStr)

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
    # calcolo la miou
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")

    # calcolo la miou con temperature scaling
    iouStr_05 = getColorEntry(iouVal_05)+'{:0.2f}'.format(iouVal_05*100) + '\033[0m'
    iouStr_075 = getColorEntry(iouVal_075)+'{:0.2f}'.format(iouVal_075*100) + '\033[0m'
    iouStr_11 = getColorEntry(iouVal_11)+'{:0.2f}'.format(iouVal_11*100) + '\033[0m'
    print ("MEAN IoU (Temp 0.5): ", iouStr_05, "%")
    print ("MEAN IoU (Temp 0.75): ", iouStr_075, "%")
    print ("MEAN IoU (Temp 1.1): ", iouStr_11, "%")     

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
