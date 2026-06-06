# da immagini di dataset con anomalie estraggo le queries e calcolo la distanza con medeie e matrice di covarianze calcolate su cityscapes

import os
import glob
import cv2
import torch
import random
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor
import argparse 


from sklearn.metrics import average_precision_score
from ood_metrics import fpr_at_95_tpr 

from eomt_estensione import EoMT_estensione
from vit import ViT


SEED = 42
NUM_CLASSES = 19

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

'''
input_transform = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor()
])



input_transform = Compose([
    Resize((1024, 1024), Image.BILINEAR),
    ToTensor()
])

target_transform = Compose([
    Resize((512, 1024), Image.NEAREST),
])

'''

input_transform = Compose(
    [
        Resize((1024, 1024), Image.BILINEAR), # bilinear è metodo di interpolazione per nuovi pixel quando faccio resize
        ToTensor() # trasforma in tenosore e valori diventano intervallo 0 1, e mette (Canali, Altezza, Larghezza)
        #Normalize([.485, .456, .406], [.229, .224, .225]), # normalizzazione per non so quale dataset
    ]
)

target_transform = Compose( #trasforamzioni per l'etichetta
    [
        Resize((1024, 1024), Image.NEAREST),
    ]
)

def load_my_state_dict(model, state_dict):  
    own_state = model.state_dict()
    for name, param in state_dict.items(): 
        clean_name = name
        if clean_name.startswith("network."):
            clean_name = clean_name[len("network."):]
        if clean_name.startswith("module."):
            clean_name = clean_name[len("module."):]
        if clean_name not in own_state:
            continue
        if own_state[clean_name].shape == param.shape:
            own_state[clean_name].copy_(param)
        else:
            print(f"❌ Shape mismatch per {clean_name}")
    return model

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--loadDir', type=str, required=True)
    parser.add_argument('--loadWeights', type=str, required=True)
    parser.add_argument('--prototypes_file', type=str, required=True)
    parser.add_argument('--cpu', action='store_true')
    
    args = parser.parse_args()


    weightspath = os.path.join(args.loadDir, args.loadWeights)
    print("Caricamento pesi modello da:", weightspath)

    encoder = ViT(img_size=(1024, 1024), patch_size=16, backbone_name="vit_base_patch14_reg4_dinov2")
    model = EoMT_estensione(encoder=encoder, num_classes=NUM_CLASSES, num_q=100, num_blocks=3) 
    model = load_my_state_dict(model, torch.load(weightspath, map_location='cpu'))
    
    device = torch.device("cpu" if args.cpu else "cuda")
    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda() 
    model.eval() 

    #### CARICAMENTO PROTOTIPI E CALCOLO MATRICI INVERSE ####

    statistiche = torch.load(args.prototypes_file, map_location=device) #leggere e caricare file con i prototipi
    prototipi = statistiche["prototipi"] #dizionario
    covarianze = statistiche["covarianze"] #dizionario
     
    matrici_inverse = {} #dizionario

    for cls_id in range(NUM_CLASSES):

        #  pinv (Pseudo-inversa) per evitare crash per matrici singolari
        matrici_inverse[cls_id] = torch.linalg.pinv(covarianze[cls_id].to(device))
        prototipi[cls_id] = prototipi[cls_id].to(device)

    ood_gts_list = []  # MEMORIZZA "Ground Truth" ovvero la verità assoluta delle anomalie
    anomaly_score_mahalanobis_list = [] # lista con score di anomalie 

    image_paths = glob.glob(os.path.expanduser(args.input))


    #### CICLO sulle immagini ####
    with torch.no_grad():
        for path in image_paths: 
            images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().to(device)

            # Estrazione feature dal modello
            if isinstance(model, torch.nn.DataParallel):  
                updated_queries, mask_logits_list, class_logits_list = model.module(images)
            else:
                updated_queries, mask_logits_list, class_logits_list = model(images)

            mask_logits = mask_logits_list[-1]   #  [Batch, Queries, H, W]
            class_logits = class_logits_list[-1] #  [Batch, Queries, Num_Classes + 1]
            
            mask_probs = torch.sigmoid(mask_logits).squeeze(0) # le maschere diventano valori tra 0 e 1 [Queries, H, W]
            class_probs = torch.softmax(class_logits, dim=-1).squeeze(0) # diventano probabilità [Queries, Num_Classes + 1]
            
            max_probs, pred_classes = torch.max(class_probs, dim=-1) # #max_probs è la probabilità massima, pred_classes è la classe predetta [Queries] 
            updated_queries = updated_queries.squeeze(0) # da [Batch, Queries, lunghezza queries] a [Queries, lunghezza queries] 


            #### CALCOLO DISTANZA DI MAHALANOBIS PER OGNI QUERY ####
            query_distances = torch.zeros(100, device=device) #vettore lungo quanto in numero di query
            
            for i in range(100):
                c = pred_classes[i].item() #classe predetta per quella query
                q_vec = updated_queries[i] #la query in questione

                

                #calcolo tutte le distanze e prendo come punteggio la distamza minima
                min_dist = float('inf')
                for cls_id in range(NUM_CLASSES):
                    delta = q_vec - prototipi[cls_id]
                    #d_sq = torch.dot(delta, torch.matmul(matrici_inverse[cls_id], delta))
                    d_sq = torch.dot(delta, delta) # DISTANZA EUCLIDEA
                    if d_sq < min_dist:
                        min_dist = d_sq
                query_distances[i] = torch.sqrt(torch.clamp(min_dist, min=0))
            


                '''
                if c < NUM_CLASSES: # se la classe della query non è void

                    

                    # calcolo distanza verso la CLASSE PREDETTA
                    delta = q_vec - prototipi[c] # sottrazione tra la query e la media della classe
                    dist_sq = torch.dot(delta, torch.matmul(matrici_inverse[c], delta)) # argomento della radice quadrata
                    query_distances[i] = torch.sqrt(torch.clamp(dist_sq, min=0)) # fa radice ma se radicando è negativo prende 0

                else:  # se ha predetto "void" (19)

                    # calcoliamo la distanza minima verso TUTTE le classi note
                    min_dist = float('inf')
                    for cls_id in range(NUM_CLASSES):
                        delta = q_vec - prototipi[cls_id]
                        d_sq = torch.dot(delta, torch.matmul(matrici_inverse[cls_id], delta)) #fa la comparazione senza la radice
                        if d_sq < min_dist:
                            min_dist = d_sq
                    query_distances[i] = torch.sqrt(torch.clamp(min_dist, min=0)) #dist minima
                '''

            #### DALLE QUERY ALLA MAPPA PIXEL (BROADCAST) ####


            '''
            # Moltiplichiamo ogni mask_probs [Queries, H, W] per il suo punteggio di anomalia
            query_distances_view = query_distances.view(100, 1, 1) #aggiungo dimensioni
            weighted_masks = mask_probs * query_distances_view # broadcast di distanza per ogni pixel [Queries, H_patch, W_patch] nei singoli patch
            
            # Per ogni pixel vince la query con l'anomalia più alta 
            anomaly_map, _ = torch.max(weighted_masks, dim=0) # [H_pacth, W_patch]

            '''

            # per ogni pixel l'indice della query con la probabilità di maschera più alta
            _, winning_query_indices = torch.max(mask_probs, dim=0) # [H_patch, W_patch]

            # Assegna a ogni pixel la distanza della sua query vincente
            anomaly_map = query_distances[winning_query_indices] #[H_patch, W_patch]


            
            # Upsampling alla risoluzione originale
            anomaly_map = anomaly_map.unsqueeze(0).unsqueeze(0)
            anomaly_map = F.interpolate(anomaly_map, size=(1024, 1024), mode="bilinear", align_corners=False)
            anomaly_result = anomaly_map.squeeze().cpu().numpy() # [512, 1024] in formato numpy



            # procedimento per i target uguale a evalAnomalyEOMT
            
            pathGT = path.replace("images", "labels_masks")             
            if "RoadObsticle21" in pathGT:    
               pathGT = pathGT.replace("webp", "png")
            if "fs_static" in pathGT:
               pathGT = pathGT.replace("jpg", "png")                
            if "RoadAnomaly" in pathGT:
               pathGT = pathGT.replace("jpg", "png")  
            
            if not os.path.exists(pathGT):
                continue
                
            mask = Image.open(pathGT)
            mask = target_transform(mask) 
            ood_gts = np.array(mask) 
            
            if "RoadAnomaly" in pathGT:
                ood_gts = np.where((ood_gts==2), 1, ood_gts)
            if "LostAndFound" in pathGT:
                ood_gts = np.where((ood_gts==0), 255, ood_gts)
                ood_gts = np.where((ood_gts==1), 0, ood_gts)
                ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts) 
            if "Streethazard" in pathGT:
                ood_gts = np.where((ood_gts==14), 255, ood_gts)
                ood_gts = np.where((ood_gts<20), 0, ood_gts)
                ood_gts = np.where((ood_gts==255), 1, ood_gts)

            if 1 not in np.unique(ood_gts):  # se non ci sono anomalie passa alla foto dopo
                continue              
            else: 
                 ood_gts_list.append(ood_gts) 
                 anomaly_score_mahalanobis_list.append(anomaly_result) 



            # SALVATAGGIO HEATMAP SU DRIVE (Prime 10 immagini)

            #ROSSO anomalia (distanza MAHALANOBIS) PIù ALTA, BLU PIù BASSO 


            if len(ood_gts_list) <= 10:
                debug_stem = os.path.splitext(os.path.basename(path))[0]
                output_drive_dir = "/content/drive/MyDrive/Validation_Dataset/heatmaps_mahalanobis"
                os.makedirs(output_drive_dir, exist_ok=True)

                cmap_min, cmap_max = anomaly_result.min(), anomaly_result.max()
                if cmap_max - cmap_min > 1e-8:
                    normalized_map = (anomaly_result - cmap_min) / (cmap_max - cmap_min)
                else:
                    normalized_map = np.zeros_like(anomaly_result)
                    
                map_u8 = (normalized_map * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(map_u8, cv2.COLORMAP_JET)

                img_bgr = cv2.imread(path)
                if img_bgr is not None:
                    h, w = anomaly_result.shape
                    img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
                    overlay = cv2.addWeighted(heatmap, 0.5, img_bgr, 0.5, 0)
                    debug_name = os.path.join(output_drive_dir, f"heatmap_{debug_stem}.jpg")
                    cv2.imwrite(debug_name, overlay)

            torch.cuda.empty_cache()



    #### CALCOLO METRICHE FINALI ####


    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_mahalanobis_list)
    
    # crea maschere per fare distinzione tra pixel anomali e normali, e per escludere quelli off topic (255)
    ood_mask = (ood_gts == 1)  
    ind_mask = (ood_gts == 0) 

    # divide quindi in due arrays1D in base a queste maschere, quello che era una matrice diventa due arrays
    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))  # array di 1 per i pixel anomali
    ind_label = np.zeros(len(ind_out))  # array di 0 per i pixel normali

    val_out = np.concatenate((ind_out, ood_out))  # unisce i due arrays dei punteggi di anomalia, prima quelli normali poi quelli anomali (le nostre predizioni)
    val_label = np.concatenate((ind_label, ood_label))  # unisce i due arrays delle label, prima 0 poi 1 (la verità)


    # METRICHE AUPRC E FPR95
    prc_auc = average_precision_score(val_label, val_out) 
    fpr = fpr_at_95_tpr(val_out, val_label) 

    print(f'AUPRC score: {prc_auc*100.0:.2f} %')
    print(f'FPR@TPR95:   {fpr*100.0:.2f} %')

    with open('results_mahalanobis.txt', 'a') as file:
        file.write(f'AUPRC: {prc_auc*100.0:.2f} | FPR@TPR95: {fpr*100.0:.2f}\n')

if __name__ == '__main__':
    main()