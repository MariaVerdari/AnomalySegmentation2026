# Copyright (c) OpenMMLab. All rights reserved.
from logging import config
import os
from xml.parsers.expat import model
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from eomt import EoMT # importo classe del modello eomt dal file eomt.py
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from huggingface_hub import hf_hub_download   #per scaricare i pesi da Hugging Face
import warnings
from huggingface_hub.utils import RepositoryNotFoundError
import torch.nn.functional as F
#from models.vit import ViT
from vit import ViT
import math



seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


NUM_CHANNELS = 3 # rgb
NUM_CLASSES = 19 # EOMT NE AGGIUNGE 1?

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR), # bilinear è metodo di interpolazione per nuovi pixel quando faccio resize
        ToTensor(), # trasforma in tenosore e valori diventano intervallo 0 1, e mette (Canali, Altezza, Larghezza)
        Normalize([.485, .456, .406], [.229, .224, .225]), # normalizzazione per non so quale dataset
    ]
)

target_transform = Compose( #trasforamzioni per l'etichetta
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)


def main():

    # per passare immagine dal terminale
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")  #cerca cartella
    parser.add_argument('--loadWeights', default="eomt_pretrained.pth") #pesi
    parser.add_argument('--loadModel', default="eomt.py") #modello EoMT
    parser.add_argument('--subset', default="val")  # che dataset prende #can be val or train (must have labels)
   
    #parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--datadir', default="/content/drive/MyDrive/Validation_Dataset/")
   
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args() #legge dal terminale (o quando lancio su colab)

    #creo tutte le liste per salvare i risultati dei vari metodi
    anomaly_score_msp_list = [] # punteggi di anomalia con msp
    anomaly_score_maxentropy_list = [] # punteggi di anomalia calcolati con l'entropia
    anomaly_score_maxlogit_list = [] # punteggi di anomalia calcolati con maxlogit
    anomaly_score_rba_list = [] # punteggi di anomalia calcolati con rba
    anomaly_score_msp_temp_05_list = [] # punteggi di anomalia con msp temp 0.5
    anomaly_score_msp_temp_075_list = [] # punteggi di anomalia con msp temp 0.75
    anomaly_score_msp_temp_11_list = [] # punteggi di anomalia con msp temp 1.1


    ood_gts_list = [] # MEMORIZZA "Ground Truth" ovvero la verità assoluta delle anomalie

    # forse questo blocco spostiamolo a quando siamo pronti a stampare i risultati
    if not os.path.exists('results.txt'): # file dei risultati
        open('results.txt', 'w').close() # crea se non esiste
    file = open('results.txt', 'a') # se esiste scrive in coda

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    #AGGIUNTA ENCODER
    encoder = ViT(
        img_size=(512, 1024),   # <-- IL PEZZO OBBLIGATORIO CHE MANCAVA!
        patch_size=16,          # <-- CAMBIATO DA 14 A 16 PERCHE NON COMBACIAVA
        backbone_name="vit_base_patch14_reg4_dinov2"    )


    model = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=100,
        num_blocks=3
    ) # creo l'istanza della classe (prende in input il numero delle classi da distinguere e il numero di queries e l'encoder)
    

    '''
    name = config.get("trainer", {}).get("logger", {}).get("init_args", {}).get("name") # cerca il nome in configs in cui ci sono dei files

    if name is None:
        warnings.warn("No logger name found in the config. Please specify a model name.")
    else:
        try:
            state_dict_path = hf_hub_download(   #cerca la repo su internet e scarica il file con i pesi, la varibile conterrà il percorso locale
                repo_id=f"tue-mps/{name}",
                filename="pytorch_model.bin",
            )

            is_dinov3 = "dinov3" in name

            # come gestire se il nome del modello contiene la parola "dinov3" che è particolare e il modello stesso si carica i pesi da solo
            if is_dinov3:
                model_kwargs["ckpt_path"] = state_dict_path
                model_kwargs["delta_weights"] = True


            if not is_dinov3:
                state_dict = torch.load(  #carica i pesi se non è dinov3
                    state_dict_path, map_location=f"cuda:{device}", weights_only=True
                )
                model.load_state_dict(state_dict, strict=False) # prende i numeri e li mette nella rete neurale appena creata, caricando solo quello che combacia a livello di layers

        except RepositoryNotFoundError:  #gestisce la situazione in cui si trova la repo su internet
            warnings.warn(
                f"Pre-trained model not found for `{name}`. Please load your own checkpoint."
            )

    '''
    
    '''
    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items(): #state dict associa a ogni layer della rete i suoi pesi
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    
'''
#NUOVA VERSIONE CON NOMI GIUSTI 
    def load_my_state_dict(model, state_dict):  
            own_state = model.state_dict()
            for name, param in state_dict.items(): 
                
                # 1. Puliamo il nome della chiave dai prefissi extra
                clean_name = name
                if clean_name.startswith("network."):
                    clean_name = clean_name.replace("network.", "")
                if clean_name.startswith("module."):
                    clean_name = clean_name.replace("module.", "")
                    
                # 2. Controlliamo se la chiave pulita esiste nel modello
                if clean_name not in own_state:
                    print(name, " non caricato (chiave pulita cercata:", clean_name, ")")
                    continue
                else:
                    # 3. Se le dimensioni combaciano, copia i pesi
                    if own_state[clean_name].shape == param.shape:
                        own_state[clean_name].copy_(param)
                    
                    # 2. SEZIONE NUOVA: Se è il pos_embed, lo interpoliamo dinamicamente!
                    elif "pos_embed" in clean_name:
                        
                        # I pesi sono [1, 4096, 768] (griglia 64x64). Vogliamo [1, 2048, 768] (griglia 32x64).
                        dim = param.shape[-1]
                        
                        # Calcoliamo la griglia originale (radice quadrata di 4096 = 64)
                        orig_size = int(param.shape[1] ** 0.5) 
                        
                        # Sappiamo che le nostre immagini sono 512x1024 e patch è 16
                        H_new = 512 // 16  # 32
                        W_new = 1024 // 16 # 64
                        
                        # Trasformiamo la sequenza 1D in un'immagine 2D per poterla ridimensionare
                        param_reshaped = param.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
                        
                        # Ridimensioniamo (interpolazione bilineare)
                        param_interpolated = F.interpolate(param_reshaped, size=(H_new, W_new), mode='bilinear', align_corners=False)
                        
                        # La riportiamo alla forma di sequenza 1D [1, 2048, 768]
                        param_final = param_interpolated.permute(0, 2, 3, 1).reshape(1, -1, dim)
                        
                        # Copiamo i pesi adattati nel modello
                        own_state[clean_name].copy_(param_final)
                        print(f"✅ pos_embed interpolato con successo da 4096 a 2048!")
                    else:
                        print(f"Dimension mismatch per {clean_name}: modello {own_state[clean_name].shape} vs pesi {param.shape}")
                        
            return model



                 
        


    #DEBUG
    pesi_salvati = torch.load(weightspath, map_location='cpu')
    
    print("\n" + "="*40)
    print("🔍 DEBUG: PRIME 10 CHIAVI NEL FILE .BIN:")
    print("="*40)
    for k in list(pesi_salvati.keys())[:10]:
        print(k)

    print("\n" + "="*40)
    print("🔍 DEBUG: PRIME 10 CHIAVI NEL TUO MODELLO:")
    print("="*40)
    for k in list(model.state_dict().keys())[:10]:
        print(k)
    print("="*40 + "\n")
    # --- FINE BLOCCO DI DEBUG ---

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage)) #carica i pesi del file dei pesi dentro all'istanza model
    print ("Model and weights LOADED successfully")


     #PARALLELIZZARE SOLO ORA 
    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda() # Se non hai forzato l'uso della CPU, il programma assume GPU e se possibile parallelizza




    
    model.eval() #modalità evaluation
   

    for path in glob.glob(os.path.expanduser(str(args.input[0]))): #ciclo su tutti i percorsi  delle immagini del dataset
        print(path)
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda() # forza RGB (3 channels), applico trasformazioni dell'input, aggiungo dimensione del batch
        # images = images.permute(0,3,1,2) #[Batch, Canali, Altezza, Larghezza]
        # NON SERVE DAVVERO INVERTIRE
       

        '''
        with torch.no_grad(): # senza calcolare i gradienti
            result = model(images) # è un tensore, contiene i logits per ogni classe per ogni pixel (Batch, Classi, Altezza, Larghezza)
        '''


        with torch.no_grad():

            mask_logits_list, class_logits_list = model(images) #output dal modello EoMT
            
            # previsioni dell'ultimo layer [-1]
            mask_logits = mask_logits_list[-1]   #  [Batch, Queries, H, W]
            class_logits = class_logits_list[-1] #  [Batch, Queries, Num_Classes + 1]

            mask_probs = torch.sigmoid(mask_logits) # le maschere diventano valori tra 0 e 1
            
            class_probs = torch.softmax(class_logits, dim=-1)[:, :, :-1] # diventano probabilità ed escludo void

            result = torch.einsum("bqc, bqhw -> bchw", class_probs, mask_probs) # combino le maschere e le classi facendo moltiplicazione tra matrici

            # result sarà [1, 20, 512, 1024] 
            result = F.interpolate(result, size=(512, 1024), mode="bilinear", align_corners=False) # si fa in modo che le misure siano quelle che abbiamo messo in input_transform e target_trasform
            
        

                



        # MAXLOGIT
        anomaly_maxlogit_result = - np.max(result.squeeze(0).data.cpu().numpy(), axis=0)  # per ogni pixel prendo il massimo tra i logit delle classi, LO METTO NEGATIVO (SENZA SOTTRARRE da 1) per avere un punteggio di anomalia (maxlogit)    
       
        #Ora facciamo il softmax, IMPLEMENTATO DA NOI
        soft_result = torch.softmax(result, dim=1) # trasforma i logit in probabilità, non c'entra con i tre result, serve per msp e maxentropy
       
        #MSP per ogni pixel prendo il massimo tra le probabilità delle classi, sottraggo da 1 per avere un punteggio di anomalia (maxsoftmax)
        anomaly_msp_result = 1.0 - np.max(soft_result.squeeze(0).data.cpu().numpy(), axis=0)

        #calcolo il MAXENTROPY su soft_result
        anomaly_entropy_result = - np.sum(soft_result.squeeze(0).data.cpu().numpy() * np.log(soft_result.squeeze(0).data.cpu().numpy() + 1e-10), axis=0) # entropia calcolata sui softmax


        #calcolo RbA
        #CONTROLLARE IL MENO E IL LOGIT/PROBABILITà

        rba_anomaly =- torch.sum(torch.tanh(result), dim=1) # somma su tutte le classi e viene [Batch, Altezza, Larghezza]
            
        anomaly_rba_result = rba_anomaly.squeeze(0).cpu().numpy() # si traforma in numpy e si toglie la dim del batch per metterlo nella lista
        # quindi per ogni pixel ho un punteggio di anomalia

        #temperature scaling per msp
        temperature = [0.5,0.75,1.1]
        scaled_soft_result = torch.softmax(result / temperature[0], dim=1)
        anomaly_msp_result_temp_05 = 1.0 - np.max(scaled_soft_result.squeeze(0).data.cpu().numpy(), axis=0)  
        scaled_soft_result = torch.softmax(result / temperature[1], dim=1)
        anomaly_msp_result_temp_075 = 1.0 - np.max(scaled_soft_result.squeeze(0).data.cpu().numpy(), axis=0)  
        scaled_soft_result = torch.softmax(result / temperature[2], dim=1)
        anomaly_msp_result_temp_11 = 1.0 - np.max(scaled_soft_result.squeeze(0).data.cpu().numpy(), axis=0)  


        # DOBBIAMO FARE IN MODO CHE FUNZIONI CON TUTTI I DATASET
        pathGT = path.replace("images", "labels_masks")  # percorso del file che contiene la label              
        if "RoadObsticle21" in pathGT:    # estensione giusta
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  
        print(pathGT)
        mask = Image.open(pathGT) #apre le labels
        mask = target_transform(mask) # trasforma con resize
        ood_gts = np.array(mask) # converte in un array NumPy, diventa matrice di 0 e 1 e numeri off topic
       
        #questo pezzo è per creare uno standard a prescindere dai singoli dataset: 0 per normale, 1 anomalia, 255 per off topic (non considerato)
        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts) #tutti gli oggetti diventano 1

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts): # se non ci sono anomalie passa alla foto dopo
            continue              
        else:  #altrimenti salva i risultati
             ood_gts_list.append(ood_gts)  # aggiunge alla lista ground truth
             anomaly_score_msp_list.append(anomaly_msp_result) # aggunge alla lista dei punteggi di anomalia
             anomaly_score_maxentropy_list.append(anomaly_entropy_result) # aggiunge alla lista dei punteggi di anomalia con entropia
             anomaly_score_maxlogit_list.append(anomaly_maxlogit_result) # aggiunge alla lista dei punteggi di anomalia con maxlogit
             anomaly_score_rba_list.append(anomaly_rba_result) # aggiunge alla lista dei punteggi di anomalia con maxlogit
             anomaly_score_msp_temp_05_list.append(anomaly_msp_result_temp_05) # aggiunge alla lista dei punteggi di anomalia con msp temp 0.5
             anomaly_score_msp_temp_075_list.append(anomaly_msp_result_temp_075) # aggiunge alla lista dei punteggi di anomalia con msp temp 0.75
             anomaly_score_msp_temp_11_list.append(anomaly_msp_result_temp_11) # aggiunge alla lista dei punteggi di anomalia con msp temp 1.1

   
        del result, anomaly_msp_result, anomaly_entropy_result, anomaly_maxlogit_result, anomaly_rba_result, anomaly_msp_result_temp_05, anomaly_msp_result_temp_075, anomaly_msp_result_temp_11, ood_gts, mask  # libera memoria una volta salvate le info
        torch.cuda.empty_cache()

    file.write( "\n")

    ood_gts = np.array(ood_gts_list)
    anomaly_scores_msp = np.array(anomaly_score_msp_list)
    anomaly_scores_maxentropy = np.array(anomaly_score_maxentropy_list)
    anomaly_scores_maxlogit = np.array(anomaly_score_maxlogit_list)
    anomaly_scores_rba = np.array(anomaly_score_rba_list)
    anomaly_scores_msp_temp_05 = np.array(anomaly_score_msp_temp_05_list)
    anomaly_scores_msp_temp_075 = np.array(anomaly_score_msp_temp_075_list)
    anomaly_scores_msp_temp_11 = np.array(anomaly_score_msp_temp_11_list)   

    # crea maschere per fare distinzione tra pixel anomali e normali, e per escludere quelli off topic (255)
    ood_mask = (ood_gts == 1)  
    ind_mask = (ood_gts == 0) # anche 255 viene 0

    # divide quindi in due arrays1D in base a queste maschere, quello che era una matrice diventa due arrays, per tutti e tre i metodi
    ood_out_msp = anomaly_scores_msp[ood_mask]
    ood_out_maxentropy = anomaly_scores_maxentropy[ood_mask]
    ood_out_maxlogit = anomaly_scores_maxlogit[ood_mask]
    ood_out_rba = anomaly_scores_rba[ood_mask]
    ood_out_msp_temp_05 = anomaly_scores_msp_temp_05[ood_mask]
    ood_out_msp_temp_075 = anomaly_scores_msp_temp_075[ood_mask]
    ood_out_msp_temp_11 = anomaly_scores_msp_temp_11[ood_mask]  

    ind_out_msp = anomaly_scores_msp[ind_mask]
    ind_out_maxentropy = anomaly_scores_maxentropy[ind_mask]
    ind_out_maxlogit = anomaly_scores_maxlogit[ind_mask]
    ind_out_rba = anomaly_scores_rba[ind_mask]
    ind_out_msp_temp_05 = anomaly_scores_msp_temp_05[ind_mask]
    ind_out_msp_temp_075 = anomaly_scores_msp_temp_075[ind_mask]
    ind_out_msp_temp_11 = anomaly_scores_msp_temp_11[ind_mask]

    ood_label_msp = np.ones(len(ood_out_msp)) # arrays di 1 per i pixel anomali
    ood_label_maxentropy = np.ones(len(ood_out_maxentropy))
    ood_label_maxlogit = np.ones(len(ood_out_maxlogit))
    ood_label_rba = np.ones(len(ood_out_rba))
    ood_label_msp_temp_05 = np.ones(len(ood_out_msp_temp_05))
    ood_label_msp_temp_075 = np.ones(len(ood_out_msp_temp_075))
    ood_label_msp_temp_11 = np.ones(len(ood_out_msp_temp_11))

    ind_label_msp = np.zeros(len(ind_out_msp)) # arrays di 0 per i pixel normali
    ind_label_maxentropy = np.zeros(len(ind_out_maxentropy))
    ind_label_maxlogit = np.zeros(len(ind_out_maxlogit))
    ind_label_rba = np.zeros(len(ind_out_rba))
    ind_label_msp_temp_05 = np.zeros(len(ind_out_msp_temp_05))
    ind_label_msp_temp_075 = np.zeros(len(ind_out_msp_temp_075))
    ind_label_msp_temp_11 = np.zeros(len(ind_out_msp_temp_11))

    val_out_msp = np.concatenate((ind_out_msp, ood_out_msp)) # unisce i due arrays dei punteggi di anomalia, prima quelli normali poi quelli anomali (le nostre predizioni)
    val_label_msp = np.concatenate((ind_label_msp, ood_label_msp)) # unisce i due arrays delle label, prima 0 poi 1 (la verità)
    val_out_maxentropy = np.concatenate((ind_out_maxentropy, ood_out_maxentropy))
    val_label_maxentropy = np.concatenate((ind_label_maxentropy, ood_label_maxentropy))
    val_out_maxlogit = np.concatenate((ind_out_maxlogit, ood_out_maxlogit))
    val_label_maxlogit = np.concatenate((ind_label_maxlogit, ood_label_maxlogit))
    val_out_rba = np.concatenate((ind_out_rba, ood_out_rba))
    val_label_rba = np.concatenate((ind_label_rba, ood_label_rba))
    val_out_msp_temp_05 = np.concatenate((ind_out_msp_temp_05, ood_out_msp_temp_05))
    val_label_msp_temp_05 = np.concatenate((ind_label_msp_temp_05, ood_label_msp_temp_05))
    val_out_msp_temp_075 = np.concatenate((ind_out_msp_temp_075, ood_out_msp_temp_075))
    val_label_msp_temp_075 = np.concatenate((ind_label_msp_temp_075, ood_label_msp_temp_075))
    val_out_msp_temp_11 = np.concatenate((ind_out_msp_temp_11, ood_out_msp_temp_11))
    val_label_msp_temp_11 = np.concatenate((ind_label_msp_temp_11, ood_label_msp_temp_11))

    prc_auc_msp = average_precision_score(val_label_msp, val_out_msp) # AUPRC: precisione nel trovare le anomalie per msp
    prc_auc_maxentropy = average_precision_score(val_label_maxentropy, val_out_maxentropy)
    prc_auc_maxlogit = average_precision_score(val_label_maxlogit, val_out_maxlogit)
    prc_auc_rba = average_precision_score(val_label_rba, val_out_rba)
    prc_auc_msp_temp_05 = average_precision_score(val_label_msp_temp_05, val_out_msp_temp_05)
    prc_auc_msp_temp_075 = average_precision_score(val_label_msp_temp_075, val_out_msp_temp_075)
    prc_auc_msp_temp_11 = average_precision_score(val_label_msp_temp_11, val_out_msp_temp_11)

    fpr_msp = fpr_at_95_tpr(val_out_msp, val_label_msp) #FPR95 per msp
    fpr_maxentropy = fpr_at_95_tpr(val_out_maxentropy, val_label_maxentropy)
    fpr_maxlogit = fpr_at_95_tpr(val_out_maxlogit, val_label_maxlogit)
    fpr_rba = fpr_at_95_tpr(val_out_rba, val_label_rba)
    fpr_msp_temp_05 = fpr_at_95_tpr(val_out_msp_temp_05, val_label_msp_temp_05)
    fpr_msp_temp_075 = fpr_at_95_tpr(val_out_msp_temp_075, val_label_msp_temp_075)
    fpr_msp_temp_11 = fpr_at_95_tpr(val_out_msp_temp_11, val_label_msp_temp_11) 

    # printa nei result e nel terminale i risultati
    print(f'AUPRC score with MSP: {prc_auc_msp*100.0}')
    print(f'FPR@TPR95 with MSP: {fpr_msp*100.0}')
    print(f'AUPRC score with MaxEntropy: {prc_auc_maxentropy*100.0}')
    print(f'FPR@TPR95 with MaxEntropy: {fpr_maxentropy*100.0}')
    print(f'AUPRC score with MaxLogit: {prc_auc_maxlogit*100.0}')
    print(f'FPR@TPR95 with MaxLogit: {fpr_maxlogit*100.0}')
    print(f'AUPRC score with RbA: {prc_auc_rba*100.0}')
    print(f'FPR@TPR95 with RbA: {fpr_rba*100.0}')
    print(f'AUPRC score with MSP (temp 0.5): {prc_auc_msp_temp_05*100.0}')
    print(f'FPR@TPR95 with MSP (temp 0.5): {fpr_msp_temp_05*100.0}')
    print(f'AUPRC score with MSP (temp 0.75): {prc_auc_msp_temp_075*100.0}')
    print(f'FPR@TPR95 with MSP (temp 0.75): {fpr_msp_temp_075*100.0}')
    print(f'AUPRC score with MSP (temp 1.1): {prc_auc_msp_temp_11*100.0}')
    print(f'FPR@TPR95 with MSP (temp 1.1): {fpr_msp_temp_11*100.0}')

    file.write(('    AUPRC score with MSP:' + str(prc_auc_msp*100.0) + '   FPR@TPR95 with MSP:' + str(fpr_msp*100.0) + '\n'))
    file.write(('    AUPRC score with MaxEntropy:' + str(prc_auc_maxentropy*100.0) + '   FPR@TPR95 with MaxEntropy:' + str(fpr_maxentropy*100.0) + '\n'))
    file.write(('    AUPRC score with MaxLogit:' + str(prc_auc_maxlogit*100.0) + '   FPR@TPR95 with MaxLogit:' + str(fpr_maxlogit*100.0) + '\n'))
    file.write(('    AUPRC score with RbA:' + str(prc_auc_rba*100.0) + '   FPR@TPR95 with RbA:' + str(fpr_rba*100.0) + '\n'))
    file.write(('    AUPRC score with MSP (temp 0.5):' + str(prc_auc_msp_temp_05*100.0) + '   FPR@TPR95 with MSP (temp 0.5):' + str(fpr_msp_temp_05*100.0) + '\n'))
    file.write(('    AUPRC score with MSP (temp 0.75):' + str(prc_auc_msp_temp_075*100.0) + '   FPR@TPR95 with MSP (temp 0.75):' + str(fpr_msp_temp_075*100.0) + '\n'))
    file.write(('    AUPRC score with MSP (temp 1.1):' + str(prc_auc_msp_temp_11*100.0) + '   FPR@TPR95 with MSP (temp 1.1):' + str(fpr_msp_temp_11*100.0) + '\n'))
    file.close()


# esegui tutto il codice che c'è dentro main()
if __name__ == '__main__':
    main()


