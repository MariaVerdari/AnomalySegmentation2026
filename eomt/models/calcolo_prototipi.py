# per ogni classe tiriamo fuori matrice var cov e media delle queries



import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor
import torch.nn.functional as F


from eomt_estensione import EoMT_estensione #modello che estrae le queries

from vit import ViT


SEED = 42
NUM_CLASSES = 19
CONFIDENCE_THRESHOLD = 0.7  # Ignoriamo le query con probabilità inferiore al 70%

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


input_transform = Compose([
    Resize((1024, 1024), Image.BILINEAR),
    ToTensor()
])

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
    parser = ArgumentParser()
    parser.add_argument('--datadir', default="/content/cityscapes_local/")
    parser.add_argument('--loadDir', default="/content/drive/MyDrive/Validation_Dataset/") 
    parser.add_argument('--loadWeights', default="eomt_cityscapes.bin")
    parser.add_argument('--prototypes_file', default="cityscapes_prototypes.pt")
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    weightspath = os.path.join(args.loadDir, args.loadWeights)
    print("Caricamento pesi modello da:", weightspath)
    encoder = ViT(img_size=(1024, 1024), patch_size=16, backbone_name="vit_base_patch14_reg4_dinov2")
    model = EoMT_estensione(encoder=encoder, num_classes=NUM_CLASSES, num_q=100, num_blocks=3) 
    model = load_my_state_dict(model, torch.load(weightspath, map_location='cpu'))
    print ("Model and weights LOADED successfully")

    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda() 
    
    model.eval() 


    ##### ESTRAZIONE DELLE QUERIES ####
    vettori_per_classe = {i: [] for i in range(NUM_CLASSES)} #dizionario di liste
    search_pattern = os.path.join(args.datadir, "leftImg8bit", "*", "*", "*.png")
    image_paths = glob.glob(search_pattern)
    # Se la cartella è piatta o strutturata diversamente, usiamo un piano B di ricerca ricorsiva
    if len(image_paths) == 0:
        search_pattern_flat = os.path.join(args.datadir, "**", "*.png")
        image_paths = glob.glob(search_pattern_flat, recursive=True)

    if len(image_paths) == 0:
        print(f"ERRORE: Nessuna immagine .png trovata nel percorso: {args.datadir}")
        return
    else:
        print(f"Trovate {len(image_paths)} immagini da analizzare per i prototipi.")

    with torch.no_grad():
        for path in image_paths:
            images = input_transform(Image.open(path).convert('RGB')).unsqueeze(0).float()
            if not args.cpu:
                images = images.cuda()

            # Estrazione feature dal modello (Gestione DataParallel)
            if isinstance(model, torch.nn.DataParallel):  #parallelizza se si può
                updated_queries, _, class_logits_list = model.module(images) #passa immagini a modello originale
            else:
                updated_queries, _, class_logits_list = model(images) # se non è parallelizzato
            


            # Logit dell'ultimo layer
            class_logits = class_logits_list[-1]  # [Batch, Queries, Num_Classes + 1]

            class_probs = torch.softmax(class_logits, dim=-1) # diventano probabilità  [Batch, Queries, Num_Classes + 1]

            max_probs, pred_classes = torch.max(class_probs, dim=-1)  #max_probs è la probabilità massima, pred_classes è la classe predetta [Batch, Queries]

            # Rimuoviamo la dimensione batch 
            updated_queries = updated_queries.squeeze(0) # da [Batch, Queries, lunghezza queries] a [Queries, lunghezza queries]
            max_probs = max_probs.squeeze(0)  # da [Batch, Queries] a  [Queries]      
            pred_classes = pred_classes.squeeze(0)   # da [Batch, Queries] a  [Queries]        

            # Smistamento vettori nelle rispettive classi
            for cls_id in range(NUM_CLASSES):  # non prende il void
                valid_mask = (pred_classes == cls_id) & (max_probs > CONFIDENCE_THRESHOLD) 
                cls_queries = updated_queries[valid_mask] # prende queries che hanno come classe predetta la classe in questione e hanno una probabilità superiore alla soglia
                
                if cls_queries.shape[0] > 0:
                    vettori_per_classe[cls_id].append(cls_queries.cpu()) #inserisco nella lista corrispondente un tensore bidimensionale [Queries selezionate, lunghezza queries] 



    ##### CALCOLO MEDIA E COVARIANZA ####

    # no pca (le queries hanno poche dim)

    print("\nEstrazione completata")

    #creo dizionari (chiavi sono id delle classi)
    prototipi = {} #rappresentanti delle classi (vettori medi delle queries)
    covarianze = {} #matrici

    for cls_id in range(NUM_CLASSES):
        if len(vettori_per_classe[cls_id]) > 0: #se c'è almeno una queries nella lista della classe

            # Unione di tutti i vettori della classe
            matrice_totale = torch.cat(vettori_per_classe[cls_id], dim=0) #diventa tensore bidimensionale [Queries selezionate, lunghezza queries] 
            N = matrice_totale.shape[0] #numero queries totali per classe
            
            # Calcolo Media (Prototipo)
            media = torch.mean(matrice_totale, dim=0) # vettore [lunghezza queries] 
            
            # Calcolo Covarianza
            matrice_centrata = matrice_totale - media #  [Queries selezionate, lunghezza queries] 
            if N > 1:
                cov = (matrice_centrata.T @ matrice_centrata) / (N - 1)
            else:
                cov = torch.zeros((768, 768))

            # Regolarizzazione per garantire l'invertibilità della matrice (aggiungo epsilon sulla diagonale)
            cov += torch.eye(768) * 0.01 

            prototipi[cls_id] = media #inserisco nel dizionario
            covarianze[cls_id] = cov #inserisco nel dizionario
            print(f"Classe {cls_id}: Modello generato con {N} vettori")
        else:
            print(f"ATTENZIONE: Nessun vettore valido trovato per la Classe {cls_id}.")


    torch.save({
        "prototipi": prototipi,
        "covarianze": covarianze
    }, args.prototypes_file)
    print(f"\nStatistiche salvate con successo in: {args.prototypes_file}")
if __name__ == '__main__':
    main()