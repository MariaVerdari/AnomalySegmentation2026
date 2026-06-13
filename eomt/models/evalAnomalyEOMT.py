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
from huggingface_hub import hf_hub_download   #per scaricare i pesi da Hugging Face, alla fine non usato
import warnings
from huggingface_hub.utils import RepositoryNotFoundError
import torch.nn.functional as F
#from models.vit import ViT
from vit import ViT
import math
import gc



seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


NUM_CHANNELS = 3 # rgb
NUM_CLASSES = 19 # EOMT ne aggiunge una cioè la void

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Griglia di temperature per lo sweep di MSP (per scegliere una T globale).
# Dai risultati precedenti l'effetto è piccolo e T>1 aiuta più di T<1.
TEMPS = [0.5, 0.75, 1.0, 1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


def _dataset_name_from_input(input_glob):
    # ".../RoadObsticle21/images/*.webp" -> "RoadObsticle21"
    parts = str(input_glob).replace("\\", "/").split("/")
    if "images" in parts:
        i = parts.index("images")
        if i > 0:
            return parts[i - 1]
    return "dataset"


input_transform = Compose(
    [
        Resize((1024, 1024), Image.BILINEAR), # 1024x1024: risoluzione nativa di EoMT (coerente con training, mIoU ed estensione)
        ToTensor() # trasforma in tensore e mette (Canali, Altezza, Larghezza)
        #Normalize([.485, .456, .406], [.229, .224, .225]), 
    ]
)

target_transform = Compose( #trasforamzioni per la label
    [
        Resize((1024, 1024), Image.NEAREST),
    ]
)


def main():

    # per passare immagini dal terminale
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
    parser.add_argument('--subset', default="val")  # che dataset prende 
   
    #parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--datadir', default="/content/drive/MyDrive/Validation_Dataset/")
   
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    # --temps: sottoinsieme di temperature da valutare in QUESTO run (per spezzare lo sweep
    # su più run e non esaurire la RAM sui dataset grandi). Default: tutta la griglia TEMPS.
    parser.add_argument('--temps', type=float, nargs='+', default=None)
    args = parser.parse_args() #legge dal terminale (o quando lancio su colab)

    temps_to_use = args.temps if args.temps else TEMPS

    #creo tutte le liste per salvare i risultati dei vari metodi
    anomaly_score_msp_list = [] # punteggi di anomalia con msp
    anomaly_score_maxentropy_list = [] # punteggi di anomalia calcolati con l'entropia
    anomaly_score_maxlogit_list = [] # punteggi di anomalia calcolati con maxlogit
    anomaly_score_rba_list = [] # punteggi di anomalia calcolati con rba
    # Sweep temperatura per MSP: per ogni T salvo SOLO i pixel validi (1D, float16),
    # non le mappe intere -> RAM bassa, posso tenere molte temperature senza OOM.
    anomaly_score_temp_lists = {t: [] for t in temps_to_use}
    temp_label_list = []  # label 0/1 dei pixel validi, allineate alle liste sopra

    
    ood_gts_list = [] # MEMORIZZA "Ground Truth" ovvero la verità assoluta delle anomalie

    if not os.path.exists('results.txt'): # file dei risultati
        open('results.txt', 'w').close() # crea se non esiste
    file = open('results.txt', 'a') # se esiste scrive in coda

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    #AGGIUNTA ENCODER
    encoder = ViT(
        img_size=(1024, 1024),   # risoluzione nativa di EoMT (griglia 64x64 = pos_embed del checkpoint, nessuna interpolazione)
        patch_size=16,
        backbone_name="vit_base_patch14_reg4_dinov2"    )


    model = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=100,
        num_blocks=3
    ) # creo l'istanza della classe (prende in input il numero delle classi da distinguere, il numero di queries e l'encoder)
    

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
    #NUOVA VERSIONE 
    def load_my_state_dict(model, state_dict):  
            own_state = model.state_dict()
            for name, param in state_dict.items(): 
                
                # pulisce il nome della chiave dai prefissi extra
                clean_name = name
                if clean_name.startswith("network."):
                    clean_name = clean_name.replace("network.", "")
                if clean_name.startswith("module."):
                    clean_name = clean_name.replace("module.", "")
                    
                # si controlla se la chiave pulita esiste nel modello
                if clean_name not in own_state:
                    print(name, " non caricato (chiave pulita cercata:", clean_name, ")")
                    continue
                else:
                    # copia i pesi se le dimensioni combaciano    SONO ARRIVATA QUI
                    if own_state[clean_name].shape == param.shape:
                        own_state[clean_name].copy_(param)
                    
                    # se è il pos_embed, lo si interpola dinamicamente
                    elif "pos_embed" in clean_name:
                        
                        # I pesi sono [1, 4096, 768] (griglia 64x64). Vogliamo [1, 2048, 768] (griglia 32x64).
                        dim = param.shape[-1]
                        
                        # Calcoliamo la griglia originale (radice quadrata di 4096 = 64)
                        orig_size = int(param.shape[1] ** 0.5) 
                        
                        # Immagini a 1024x1024 e patch 16 -> griglia 64x64 (combacia col checkpoint:
                        # questo ramo non scatta più, ma resta corretto se si cambia risoluzione)
                        H_new = 1024 // 16  # 64
                        W_new = 1024 // 16  # 64
                        
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

            # result sarà [1, 20, 1024, 1024]
            result = F.interpolate(result, size=(1024, 1024), mode="bilinear", align_corners=False) # stessa risoluzione di input_transform e target_transform
            
        

                



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

        # temperature scaling per MSP: una mappa MSP per ogni T della griglia (un solo forward)
        msp_temp_results = {}
        for t in temps_to_use:
            scaled_soft_result = torch.softmax(result / t, dim=1)
            msp_temp_results[t] = 1.0 - np.max(scaled_soft_result.squeeze(0).data.cpu().numpy(), axis=0)


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
             # === VECCHIO ACCUMULO (mappe intere) — commentato per eventuale ripristino ===
             # ood_gts_list.append(ood_gts)
             # anomaly_score_msp_list.append(anomaly_msp_result)
             # anomaly_score_maxentropy_list.append(anomaly_entropy_result)
             # anomaly_score_maxlogit_list.append(anomaly_maxlogit_result)
             # anomaly_score_rba_list.append(anomaly_rba_result)

             # Tengo solo i pixel validi (0/1) in float32, per TUTTI i metodi (base + sweep):
             # poca RAM (scarto i pixel 255 e niente mappe intere a piena risoluzione),
             # numeri identici a prima. Indispensabile sui dataset grandi (es. LostFound).
             valid_sweep = (ood_gts == 0) | (ood_gts == 1)
             ood_gts_list.append(None)  # serve solo a contare le immagini tenute (heatmap sotto)
             temp_label_list.append(ood_gts[valid_sweep].astype(np.uint8))
             anomaly_score_msp_list.append(anomaly_msp_result[valid_sweep].astype(np.float32))
             anomaly_score_maxentropy_list.append(anomaly_entropy_result[valid_sweep].astype(np.float32))
             anomaly_score_maxlogit_list.append(anomaly_maxlogit_result[valid_sweep].astype(np.float32))
             anomaly_score_rba_list.append(anomaly_rba_result[valid_sweep].astype(np.float32))
             for t in temps_to_use:
                 anomaly_score_temp_lists[t].append(msp_temp_results[t][valid_sweep].astype(np.float32))

   
             # === LOGICA SALVATAGGIO HEATMAP SU DRIVE ===

             #ROSSO RBA PIù ALTA, BLU PIù BASSO 

             cosidered_anomaly_result =anomaly_msp_result #da CAMBIARE A PIACERE


             if len(ood_gts_list) <= 10:
                 debug_stem = os.path.splitext(os.path.basename(path))[0]
                 output_drive_dir = "/content/drive/MyDrive/Validation_Dataset/RoadObsticle21/heatmaps_rba"
                 os.makedirs(output_drive_dir, exist_ok=True)

                 # 1. Normalizzazione min-max locale a [0, 255]
                 cmap_min, cmap_max = cosidered_anomaly_result.min(), cosidered_anomaly_result.max()
                 if cmap_max - cmap_min > 1e-8:
                     normalized_map = (cosidered_anomaly_result - cmap_min) / (cmap_max - cmap_min)
                 else:
                     normalized_map = np.zeros_like(cosidered_anomaly_result)
                 map_u8 = (normalized_map * 255).astype(np.uint8)

                 # 2. Applicazione della colormap JET (Rosso = Anomalia)
                 heatmap = cv2.applyColorMap(map_u8, cv2.COLORMAP_JET)

                 # 3. Lettura e ridimensionamento dell'immagine originale a 1024x512
                 img_bgr = cv2.imread(path)
                 if img_bgr is not None:
                     h, w = cosidered_anomaly_result.shape
                     img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

                     # 4. Sovrapposizione (trasparenza al 50%)
                     overlay = cv2.addWeighted(heatmap, 0.5, img_bgr, 0.5, 0)

                     # 5. Salvataggio su Google Drive
                     debug_name = os.path.join(output_drive_dir, f"heatmap_{debug_stem}.jpg")
                     cv2.imwrite(debug_name, overlay)
                     print(f"-> [HEATMAP TRASPARENTE {len(ood_gts_list)}/10] Salvata in: {debug_name}")



        del result, anomaly_msp_result, anomaly_entropy_result, anomaly_maxlogit_result, anomaly_rba_result, msp_temp_results, ood_gts, mask  # libera memoria una volta salvate le info
        torch.cuda.empty_cache()

    file.write( "\n")

    # === VECCHIO SCHEMA (mappe intere + maschera) — tenuto commentato per eventuale ripristino.
    # NB: per riattivarlo va ripristinato anche l'accumulo delle mappe intere (vedi sopra). ===
    '''
    ood_gts = np.array(ood_gts_list)
    anomaly_scores_msp = np.array(anomaly_score_msp_list)
    anomaly_scores_maxentropy = np.array(anomaly_score_maxentropy_list)
    anomaly_scores_maxlogit = np.array(anomaly_score_maxlogit_list)
    anomaly_scores_rba = np.array(anomaly_score_rba_list)

    # crea maschere per fare distinzione tra pixel anomali e normali, e per escludere quelli off topic (255)
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0) # anche 255 viene 0

    # divide quindi in due arrays1D in base a queste maschere, quello che era una matrice diventa due arrays, per tutti e tre i metodi
    ood_out_msp = anomaly_scores_msp[ood_mask]
    ood_out_maxentropy = anomaly_scores_maxentropy[ood_mask]
    ood_out_maxlogit = anomaly_scores_maxlogit[ood_mask]
    ood_out_rba = anomaly_scores_rba[ood_mask]

    ind_out_msp = anomaly_scores_msp[ind_mask]
    ind_out_maxentropy = anomaly_scores_maxentropy[ind_mask]
    ind_out_maxlogit = anomaly_scores_maxlogit[ind_mask]
    ind_out_rba = anomaly_scores_rba[ind_mask]

    ood_label_msp = np.ones(len(ood_out_msp)) # arrays di 1 per i pixel anomali
    ood_label_maxentropy = np.ones(len(ood_out_maxentropy))
    ood_label_maxlogit = np.ones(len(ood_out_maxlogit))
    ood_label_rba = np.ones(len(ood_out_rba))

    ind_label_msp = np.zeros(len(ind_out_msp)) # arrays di 0 per i pixel normali
    ind_label_maxentropy = np.zeros(len(ind_out_maxentropy))
    ind_label_maxlogit = np.zeros(len(ind_out_maxlogit))
    ind_label_rba = np.zeros(len(ind_out_rba))

    val_out_msp = np.concatenate((ind_out_msp, ood_out_msp))
    val_label_msp = np.concatenate((ind_label_msp, ood_label_msp))
    val_out_maxentropy = np.concatenate((ind_out_maxentropy, ood_out_maxentropy))
    val_label_maxentropy = np.concatenate((ind_label_maxentropy, ood_label_maxentropy))
    val_out_maxlogit = np.concatenate((ind_out_maxlogit, ood_out_maxlogit))
    val_label_maxlogit = np.concatenate((ind_label_maxlogit, ood_label_maxlogit))
    val_out_rba = np.concatenate((ind_out_rba, ood_out_rba))
    val_label_rba = np.concatenate((ind_label_rba, ood_label_rba))

    prc_auc_msp = average_precision_score(val_label_msp, val_out_msp)
    prc_auc_maxentropy = average_precision_score(val_label_maxentropy, val_out_maxentropy)
    prc_auc_maxlogit = average_precision_score(val_label_maxlogit, val_out_maxlogit)
    prc_auc_rba = average_precision_score(val_label_rba, val_out_rba)

    fpr_msp = fpr_at_95_tpr(val_out_msp, val_label_msp)
    fpr_maxentropy = fpr_at_95_tpr(val_out_maxentropy, val_label_maxentropy)
    fpr_maxlogit = fpr_at_95_tpr(val_out_maxlogit, val_label_maxlogit)
    fpr_rba = fpr_at_95_tpr(val_out_rba, val_label_rba)
    '''

    # === NUOVO SCHEMA: punteggi già ristretti ai soli pixel validi (0/1) in float32 -> concateno 1D.
    # Risultato IDENTICO al vecchio (stessi pixel), ma con molta meno RAM (niente mappe intere,
    # i pixel 255 scartati in accumulo). Indispensabile sui dataset grandi (es. LostFound). ===
    val_label = np.concatenate(temp_label_list)   # ground truth 0/1 dei pixel validi
    val_out_msp = np.concatenate(anomaly_score_msp_list)
    val_out_maxentropy = np.concatenate(anomaly_score_maxentropy_list)
    val_out_maxlogit = np.concatenate(anomaly_score_maxlogit_list)
    val_out_rba = np.concatenate(anomaly_score_rba_list)

    prc_auc_msp = average_precision_score(val_label, val_out_msp)
    prc_auc_maxentropy = average_precision_score(val_label, val_out_maxentropy)
    prc_auc_maxlogit = average_precision_score(val_label, val_out_maxlogit)
    prc_auc_rba = average_precision_score(val_label, val_out_rba)

    fpr_msp = fpr_at_95_tpr(val_out_msp, val_label)
    fpr_maxentropy = fpr_at_95_tpr(val_out_maxentropy, val_label)
    fpr_maxlogit = fpr_at_95_tpr(val_out_maxlogit, val_label)
    fpr_rba = fpr_at_95_tpr(val_out_rba, val_label)

    # printa nei result e nel terminale i risultati
    print(f'AUPRC score with MSP: {prc_auc_msp*100.0}')
    print(f'FPR@TPR95 with MSP: {fpr_msp*100.0}')
    print(f'AUPRC score with MaxEntropy: {prc_auc_maxentropy*100.0}')
    print(f'FPR@TPR95 with MaxEntropy: {fpr_maxentropy*100.0}')
    print(f'AUPRC score with MaxLogit: {prc_auc_maxlogit*100.0}')
    print(f'FPR@TPR95 with MaxLogit: {fpr_maxlogit*100.0}')
    print(f'AUPRC score with RbA: {prc_auc_rba*100.0}')
    print(f'FPR@TPR95 with RbA: {fpr_rba*100.0}')

    file.write(('    AUPRC score with MSP:' + str(prc_auc_msp*100.0) + '   FPR@TPR95 with MSP:' + str(fpr_msp*100.0) + '\n'))
    file.write(('    AUPRC score with MaxEntropy:' + str(prc_auc_maxentropy*100.0) + '   FPR@TPR95 with MaxEntropy:' + str(fpr_maxentropy*100.0) + '\n'))
    file.write(('    AUPRC score with MaxLogit:' + str(prc_auc_maxlogit*100.0) + '   FPR@TPR95 with MaxLogit:' + str(fpr_maxlogit*100.0) + '\n'))
    file.write(('    AUPRC score with RbA:' + str(prc_auc_rba*100.0) + '   FPR@TPR95 with RbA:' + str(fpr_rba*100.0) + '\n'))

    # Libero la RAM dei metodi base (mappe intere a piena risoluzione + array 1D): non
    # servono più, e tenerli durante lo sweep raddoppiava il picco di memoria -> OOM.
    del val_out_msp, val_out_maxentropy, val_out_maxlogit, val_out_rba
    del anomaly_score_msp_list, anomaly_score_maxentropy_list, anomaly_score_maxlogit_list, anomaly_score_rba_list
    gc.collect()

    # --- Sweep temperatura MSP: AUPRC/FPR per ogni T, stampa + CSV per scegliere la T globale ---
    # Liste di soli pixel validi (1D); libero ogni temperatura subito dopo averla usata.
    val_label_sweep = np.concatenate(temp_label_list)   # label 0/1 dei pixel validi
    dataset_name = _dataset_name_from_input(args.input[0])
    print(f"\n=== Temperature sweep (MSP) - dataset: {dataset_name} ===")
    csv_path = "temp_sweep.csv"
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a") as csv_f:
        if write_header:
            csv_f.write("dataset,T,AUPRC,FPR\n")
        for t in temps_to_use:
            scores_t = np.concatenate(anomaly_score_temp_lists[t])   # solo pixel validi (float32)
            anomaly_score_temp_lists[t] = None   # libero la lista accumulata di questa T
            auprc_t = average_precision_score(val_label_sweep, scores_t) * 100.0
            fpr_t = fpr_at_95_tpr(scores_t, val_label_sweep) * 100.0
            del scores_t
            gc.collect()
            print(f'  T={t:<4}  AUPRC={auprc_t:6.2f}  FPR={fpr_t:6.2f}  (AUPRC-FPR={auprc_t - fpr_t:7.2f})')
            file.write(f'    MSP (T={t}):  AUPRC:{auprc_t}   FPR@TPR95:{fpr_t}\n')
            csv_f.write(f"{dataset_name},{t},{auprc_t},{fpr_t}\n")

    file.close()


# esegui tutto il codice che c'è dentro main()
if __name__ == '__main__':
    main()


