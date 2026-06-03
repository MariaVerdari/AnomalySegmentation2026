# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

### abbiamo usato ioueval

#from eomt.models import vit
import vit

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
NUM_CLASSES = 19 # 19 classi valide + 1 classe ignore (void)


# blocco aggiunto per le etichette sfasate 

class MapToTrainIds(object):
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
    


image_transform = ToPILImage()


'''
# Impostiamo la risoluzione fissa 512x1024 richiesta dall'encoder


input_transform_cityscapes = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225])
])

'''

from torchvision import transforms


# 1. Nel transform di input
input_transform_cityscapes = Compose([
    Resize((1024, 2048)), # Cambialo in 1024x2048: è la risoluzione nativa di Cityscapes!
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])



'''

input_transform_cityscapes = transforms.Compose([
    transforms.Resize((896, 896)), # Risoluzione nativa esatta del checkpoint (64 patch * 14 pxl)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])
'''


target_transform_cityscapes = Compose([
    #Resize((512, 1024), Image.NEAREST),
    MapToTrainIds(),
    ToLabel(),
    Relabel(255, 19),   # Mappa il void standard di Cityscapes (255) a 19
])


def main(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)


    '''
    # 1. COSTRUISCO L'ENCODER E IL MODELLO EOMT
    print("Inizializzazione del modello EoMT...")
    encoder = ViT(
        #img_size=(512, 1024),
        img_size=(1024, 1024),  
        patch_size=16,
        backbone_name="vit_base_patch14_reg4_dinov2"
    )

    '''


    # 1. COSTRUISCO L'ENCODER E IL MODELLO EOMT
    print("Inizializzazione del modello EoMT...")
    
    encoder = ViT(
    img_size=(1024, 2048), # Usa la risoluzione piena
    patch_size=16,         # Coerente con i tuoi pesi .bin
    backbone_name="vit_base_patch14_reg4_dinov2" 
    )

    
    model = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES, 
        num_q=100,
        num_blocks=3
    )


    '''
    
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
    '''

    #NUOVA VERSIONE CON NOMI GIUSTI 
    def load_my_state_dict(model, state_dict):  
            own_state = model.state_dict()
            for name, param in state_dict.items(): 
                
                # 1. Puliamo il nome della chiave dai prefissi extra

                
                # 1. Puliamo il nome della chiave usando lo slicing sicuro
                clean_name = name
                if clean_name.startswith("network."):
                    clean_name = clean_name[len("network."):]
                if clean_name.startswith("module."):
                    clean_name = clean_name[len("module."):]
                    
                # 2. Controlliamo se la chiave pulita esiste nel modello
                if clean_name not in own_state:
                    print(name, " non caricato (chiave pulita cercata:", clean_name, ")")
                    continue
                else:
                    # 3. Se le dimensioni combaciano, copia i pesi
                    if own_state[clean_name].shape == param.shape:
                        own_state[clean_name].copy_(param)


                                # --- SEZIONE INTERPOLAZIONE POS_EMBED ---
                    elif "pos_embed" in clean_name:
                        # 1. Trova automaticamente il numero di token extra
                        total_tokens = param.shape[1]
                        num_extra_tokens = 0
                        patch_side = 0
                        
                        # Cerchiamo un numero di "extra tokens" (0-10) che lasci un numero di patch quadrato
                        for extra in range(0, 10):
                            num_patches = total_tokens - extra
                            side = int(num_patches**0.5)
                            if side * side == num_patches:
                                num_extra_tokens = extra
                                patch_side = side
                                break
                        
                        if patch_side == 0:
                            print(f"❌ Impossibile determinare la griglia per {clean_name}. Shape: {param.shape}")
                            return model
                        
                        print(f"✅ Auto-detect pos_embed: {num_extra_tokens} extra tokens, griglia {patch_side}x{patch_side}")

                        # 2. Isola i token
                        extra_tokens = param[:, :num_extra_tokens, :]
                        patch_embed = param[:, num_extra_tokens:, :] 
                        
                        # 3. Trasformazione
                        dim = patch_embed.shape[-1]
                        # Reshape basato sul patch_side trovato
                        patch_embed_2d = patch_embed.reshape(1, patch_side, patch_side, dim).permute(0, 3, 1, 2)
                        
                        # 4. Interpolazione per la nuova risoluzione (1024x2048)
                        new_h = 1024 // 16 # 64
                        new_w = 2048 // 16 # 128
                        
                        patch_embed_interp = F.interpolate(
                            patch_embed_2d, 
                            size=(new_h, new_w), 
                            mode='bilinear', 
                            align_corners=False
                        )
                        
                        # 5. Riassemblaggio
                        patch_embed_final = patch_embed_interp.permute(0, 2, 3, 1).reshape(1, -1, dim)
                        pos_embed_final = torch.cat([extra_tokens, patch_embed_final], dim=1)
                        
                        # 6. Copia nel modello
                        if pos_embed_final.shape == own_state[clean_name].shape:
                            own_state[clean_name].copy_(pos_embed_final)
                            print(f"✅ Interpolazione completata: {pos_embed_final.shape}")
                        else:
                            print(f"❌ Errore shape finale: atteso {own_state[clean_name].shape}, ottenuto {pos_embed_final.shape}")
                     
                    else:
                        print(f"Dimension mismatch per {clean_name}: modello {own_state[clean_name].shape} vs pesi {param.shape}")
                        
            return model


      
    # 2. CARICO I PESI PRIMA SUL MODELLO GREGGIO
    weightspath = args.loadDir + args.loadWeights
    print(f"Caricamento pesi da: {weightspath}")
                 
    # Carica i pesi originali dal file
    state_dict = torch.load(weightspath, map_location="cpu", weights_only=True)
    
    # Applica la nostra funzione di pulizia e interpolazione
    model = load_my_state_dict(model, state_dict)
    print("Model and weights LOADED successfully")

    # 3. SOLO ADESSO PARALLELIZZI E MANDI IN GPU
    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()



    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")


    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)


    iouEvalVal = iouEval(NUM_CLASSES+1)
    iouEvalVal_05 = iouEval(NUM_CLASSES+1)
    iouEvalVal_075 = iouEval(NUM_CLASSES+1)
    iouEvalVal_11 = iouEval(NUM_CLASSES+1)

    


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


            result_probs = torch.nn.functional.interpolate(
            result_probs, 
            size= labels.shape[-2:], # Estrae altezza e larghezza reali dal tensore del Ground Truth
            mode="bilinear", 
            align_corners=False
            )


           
            # 5. Argmax finale
            predicted_classes = result_probs.argmax(dim=1)

            '''
            result_probs = F.interpolate(result_probs, size=(512, 1024), mode="bilinear", align_corners=False)
            


            # 4. Trovo la classe vincente per ogni pixel!
            # .max(1)[1] significa: guarda l'asse delle classi (1) e prendi l'indice [1] del valore massimo

            
            predicted_classes = result_probs.max(1)[1].unsqueeze(1).data

            '''
            # 5. Argmax finale (Aggiunto .unsqueeze(1) per evitare il crash)
            predicted_classes = result_probs.argmax(dim=1).unsqueeze(1).data



            #temperature sclaing

            temperature = [0.5, 0.75, 1.1]
            
            scaled_result = result_probs / temperature[0]
            predicted_classes_05 = scaled_result.argmax(dim=1).unsqueeze(1).data
            
            scaled_result = result_probs / temperature[1]
            predicted_classes_075 = scaled_result.argmax(dim=1).unsqueeze(1).data
            
            scaled_result = result_probs / temperature[2]
            predicted_classes_11 = scaled_result.argmax(dim=1).unsqueeze(1).data
            

        # --- 3. MASCHERAMENTO DEI PIXEL VOID (Previene falsi positivi) ---
        ignore_mask = (labels == 19)
        predicted_classes[ignore_mask] = 19
        predicted_classes_05[ignore_mask] = 19
        predicted_classes_075[ignore_mask] = 19
        predicted_classes_11[ignore_mask] = 19


        # Controlla che non ci siano valori fuori range prima di mandare in crash la GPU
        assert labels.max() < 20, f"Errore: trovato valore etichetta {labels.max()} > 19"
        assert predicted_classes.max() < 20, f"Errore: trovato valore predizione {predicted_classes.max()} > 19"

        # Passo le predizioni e le maschere ground truth al valutatore IoU
        iouEvalVal.addBatch(predicted_classes, labels)
        iouEvalVal_05.addBatch(predicted_classes_05, labels)
        iouEvalVal_075.addBatch(predicted_classes_075, labels)
        iouEvalVal_11.addBatch(predicted_classes_11, labels)
                

       

        filenameSave = filename[0].split("leftImg8bit/")[1] 
        print (step, filenameSave)
    
        
   



    # Recuperiamo solo i vettori IoU per singola classe
    _, iou_classes = iouEvalVal.getIoU()
    _, iou_classes_05 = iouEvalVal_05.getIoU()
    _, iou_classes_075 = iouEvalVal_075.getIoU()
    _, iou_classes_11 = iouEvalVal_11.getIoU()

    # Calcoliamo la media dividendo rigorosamente per 19 (NUM_CLASSES)
    iouVal = iou_classes[:NUM_CLASSES].mean().item()
    iouVal_05 = iou_classes_05[:NUM_CLASSES].mean().item()
    iouVal_075 = iou_classes_075[:NUM_CLASSES].mean().item()
    iouVal_11 = iou_classes_11[:NUM_CLASSES].mean().item()

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
