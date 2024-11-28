
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:01:37 2024

@author: aina
"""
import pickle
import numpy as np 
import cv2
from sklearn import metrics  
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import seaborn as sns
 

imatges_cropped_path = '/Users/aina/Desktop/uni/4rt/psiv/repte 3/pickels/imatges_cropped_totals.pkl' 
with open(imatges_cropped_path, 'rb') as f:
    imatges_cropped = pickle.load(f)  
metadades_cropped_path = '/Users/aina/Desktop/uni/4rt/psiv/repte 3/pickels/metadades_cropped_totals.pkl' 
with open(metadades_cropped_path, 'rb') as f:
    metadades_cropped = pickle.load(f)  
#reconstruccions_cropped_path =  '/Users/aina/Desktop/uni/4rt/psiv/repte 3/pickels/reconstruccions_cropped_totals_model1.pkl' 
reconstruccions_cropped_path =  '/Users/aina/Desktop/uni/4rt/psiv/repte 3/pickels/reconstruccions_cropped_totals_model2.pkl' 
with open(reconstruccions_cropped_path, 'rb') as f:
    reconstruccions_cropped = pickle.load(f)  

imatges_annotated_path =  '/Users/aina/Desktop/uni/4rt/psiv/repte 3/pickels/imatges_annotated.pkl' 
with open(imatges_annotated_path, 'rb') as f:
    imatges_annotated = pickle.load(f)  
metadades_annotated_path =  '/Users/aina/Desktop/uni/4rt/psiv/repte 3/pickels/metadades_annotated.pkl' 
with open(metadades_annotated_path, 'rb') as f:
    metadades_annotated = pickle.load(f)  
#reconstruccions_annotated_path =  '/Users/aina/Desktop/uni/4rt/psiv/repte 3/pickels/reconstruccions_annotated_model1.pkl' 
reconstruccions_annotated_path =  '/Users/aina/Desktop/uni/4rt/psiv/repte 3/pickels/reconstruccions_annotated_model2.pkl' 
with open(reconstruccions_annotated_path, 'rb') as f:
    reconstruccions_annotated = pickle.load(f)  

imgxfolder = 50
index = 0
for pacient, dades in metadades_cropped.items():
    dades['Imatges'] = imatges_cropped[index : index + imgxfolder]
    dades['Reconstruccions'] = reconstruccions_cropped[index : index + imgxfolder]
    index += imgxfolder

index = 0
for pacient, dades in metadades_annotated.items():
    imgxfolder = len(dades['Presence']) 
    dades['Presence'] = [0 if p == -1 else p for p in dades['Presence']]
    dades['Imatges'] = imatges_annotated[index : index + imgxfolder]
    dades['Reconstruccions'] = reconstruccions_annotated[index : index + imgxfolder]
    index += imgxfolder
annotated = metadades_annotated
cropped = metadades_cropped 
#######################################################################################################################################################################################################################################################################################################################################################################################
 
def freq_red(imatges_original, imatges_autoencoder):
    fred = []                                    
    for i in range (len(imatges_original)):
        imatge_original_hsv = cv2.cvtColor(imatges_original[i], cv2.COLOR_RGB2HSV)
        imatge_autoencoder_hsv = cv2.cvtColor(imatges_autoencoder[i], cv2.COLOR_RGB2HSV) 
        imatge_original_hsv_hue = imatge_original_hsv[:, :, 0]
        imatge_autoencoder_hsv_hue = imatge_autoencoder_hsv[:, :, 0]    
        num_orig = np.sum(((imatge_original_hsv_hue > 160) & (imatge_original_hsv_hue <= 1)) | (imatge_original_hsv_hue < 20)) 
        num_auto = np.sum(((imatge_autoencoder_hsv_hue > 160) & (imatge_autoencoder_hsv_hue <= 1)) | (imatge_autoencoder_hsv_hue < 20))
        if num_auto == 0:  
            fred.append(0)   
        else:
            fred.append(num_orig/num_auto)
    return fred     
    

def threshold_pt(y_true, y_score, tipus, fold_number):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)   
    dist_euclidean = np.sqrt(fpr**2 + (1 - tpr)**2) 
    idx = np.argmin(dist_euclidean)
    th_optim = thresholds[idx]
    print('Threshold òptim:', th_optim)   
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--') 
    if tipus == 'fred':
        plt.title(f"Corba ROC Train Threshold Fred - Fold {fold_number}", fontsize=15)
    else:
        plt.title(f"Corba ROC Train Threshold Finestres - Fold {fold_number}", fontsize=15)  
    ax.set_xlabel("False Positive Rate", fontsize=15)
    ax.set_ylabel("True Positive Rate", fontsize=15)
    ax.set_xticks(np.linspace(0, 1, 11)) 
    ax.set_yticks(np.linspace(0, 1, 11))   
    return th_optim


def avaluacio_th(presence_test, pred_test, fold):
    cm = confusion_matrix(presence_test, pred_test)   
    TN, FP, FN, TP = cm.ravel() 
    recall_positiu = TP / (TP + FN) if (TP + FN) > 0 else 0
    recall_negatiu = TN / (TN + FP) if (TN + FP) > 0 else 0
    precisio_positiva = TP / (TP + FP) if (TP + FP) > 0 else 0
    precisio_negativa = TN / (TN + FN) if (TN + FN) > 0 else 0
    plt.figure(figsize=(6, 6)) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Presència", "Presència"], yticklabels=["No Presència", "Presència"])
    plt.title(f'Matriu de Confusió - Fold {fold}')
    plt.xlabel('Predicció')
    plt.ylabel('Verdader')
    plt.show()
    print(f"Evaluació Fold {fold}:")
    print(f"  Recal Positiu (Sensibilitat): {recall_positiu:.4f}")
    print(f"  Recal Negatiu (Especificitat): {recall_negatiu:.4f}")
    print(f"  Precision Positiva: {precisio_positiva:.4f}")
    print(f"  Precision Negativa: {precisio_negativa:.4f}")
    print('\n')   
    metriques = {
        "recall_positiu": recall_positiu,
        "recall_negatiu": recall_negatiu,
        "precisio_positiva": precisio_positiva,
        "precisio_negativa": precisio_negativa,
        "confusio_matrix": cm}
    return metriques


#######################################################################################################################################################################################################################################################################################################################################################################################

pacients = list(annotated.keys()) 
kf = KFold(n_splits=10, shuffle=True, random_state=42) 
fold = 1
pacients_fold = []
metriques_th_fred = []
metriques_th_finestres = []
th_fred = [] 
th_finestres = [] 

print('Cross Vaidation Threshold Fred')
for train_index, test_index in kf.split(pacients): 
    train_keys = [pacients[i] for i in train_index]
    test_keys = [pacients[i] for i in test_index] 
    pacients_fold.append({"train": train_keys, "test": test_keys})
     
    imatges_train = []
    presence_train = []
    reconstruccions_train = [] 
    imatges_test = []
    presence_test = []
    reconstruccions_test = []

    for key in train_keys:
        presence_train.extend(annotated[key]["Presence"])
        imatges_train.extend(annotated[key]["Imatges"]) 
        reconstruccions_train.extend(annotated[key]["Reconstruccions"])
     
    for key in test_keys:
        presence_test.extend(annotated[key]["Presence"])
        imatges_test.extend(annotated[key]["Imatges"])
        reconstruccions_test.extend(annotated[key]["Reconstruccions"])
 
    fred_train = freq_red(imatges_train, reconstruccions_train)   
    fred_test = freq_red(imatges_test, reconstruccions_test)   
    th_optim = threshold_pt(presence_train, fred_train, 'fred', fold)   
    th_fred.append(th_optim)
     
    pred_test = [1 if imatge >= th_optim else 0 for imatge in fred_test] 
    metriques = avaluacio_th(presence_test, pred_test, fold)
    metriques_th_fred.append(metriques)

    fold += 1


print('\n')
print('\n Cross Vaidation Threshold Finsetres')
for fold in range(1, 11):
    train_keys = pacients_fold[0]['train']
    test_keys = pacients_fold[0]['test']

    imatges_train = []
    presence_train = []
    reconstruccions_train = [] 
    imatges_test = []
    presence_test = []
    reconstruccions_test = []

    for key in train_keys:
        try: 
            presence_train.append(cropped[key]["Presence"])
            imatges_train.extend(cropped[key]["Imatges"]) 
            reconstruccions_train.extend(cropped[key]["Reconstruccions"])
        except KeyError:
            print(f"Clave {key} no encontrada en cropped.")

    for key in test_keys:
        try: 
            presence_test.append(cropped[key]["Presence"])
            imatges_test.extend(cropped[key]["Imatges"])
            reconstruccions_test.extend(cropped[key]["Reconstruccions"])
        except KeyError:
            print(f"Clave {key} no encontrada en cropped.")

    fred_train = freq_red(imatges_train, reconstruccions_train) 
    fred_test = freq_red(imatges_test, reconstruccions_test)  
    pred_train = [1 if imatge >= th_fred[fold - 1] else 0 for imatge in fred_train]
    pred_test = [1 if imatge >= th_fred[fold - 1] else 0 for imatge in fred_test]
    
    finestres_train = [sum(pred_train[i:i+50])/50 for i in range(0, len(pred_train), 50)] 
    finestres_test = [sum(pred_test[i:i+50])/50 for i in range(0, len(pred_test), 50)]

    th_optim = threshold_pt(presence_train, finestres_train, 'finestres', fold)
    th_finestres.append(th_optim)
    pred_test = [1 if imatge >= th_optim else 0 for imatge in finestres_test] 
    metriques = avaluacio_th(presence_test, pred_test, fold)
    metriques_th_finestres.append(metriques) 
 

#######################################################################################################################################################################################################################################################################################################################################################################################

metriques_avaluar = ["recall_positiu", "recall_negatiu", "precisio_positiva", "precisio_negativa"]
resultats_th_fred = {}
resultats_th_finestres = {} 
for metrica in metriques_avaluar: 
    valors_fred = [d[metrica] for d in metriques_th_fred]
    mitjana_fred = np.mean(valors_fred)
    std_fred = np.std(valors_fred)
    resultats_th_fred[metrica] = {"mitjana": mitjana_fred, "std": std_fred} 
    valors_finestres = [d[metrica] for d in metriques_th_finestres]
    mitjana_finestres = np.mean(valors_finestres)
    std_finestres = np.std(valors_finestres)
    resultats_th_finestres[metrica] = {"mitjana": mitjana_finestres, "std": std_finestres}

cm_acumulada_fred = np.zeros((2, 2), dtype=int) 
cm_acumulada_finestres = np.zeros((2, 2), dtype=int)
for fold in metriques_th_fred:
    cm_acumulada_fred += fold["confusio_matrix"]  
for fold in metriques_th_finestres:
    cm_acumulada_finestres += fold["confusio_matrix"] 
 
mitjana_th_fred = np.mean(th_fred)
std_th_fred = np.std(th_fred)
mitjana_th_finestres = np.mean(th_finestres)
std_th_finestres = np.std(th_finestres)
 
print('\nMètriques Thresholds Fred')
for metric, stats in resultats_th_fred.items():
    print(f"{metric}: Mitjana = {stats['mitjana']:.4f}, Std = {stats['std']:.4f}") 
print('\nIntèrval Thresholds Fred')
print(f"Mitjana = {mitjana_th_fred:.4f}, Std = {std_th_fred:.4f}")
plt.figure(figsize=(6, 6))
sns.heatmap(cm_acumulada_fred, annot=True, fmt='d', cmap='Blues', xticklabels=["No Presència", "Presència"], yticklabels=["No Presència", "Presència"])
plt.title("Matriu de Confusió Acumulada Thresholds Fred")
plt.xlabel('Predicció')
plt.ylabel('Verdader')
plt.show()

print('\nMètriques Thresholds Finestres')
for metric, stats in resultats_th_finestres.items():
    print(f"{metric}: Mitjana = {stats['mitjana']:.4f}, Std = {stats['std']:.4f}")
print('\nIntèrval Thresholds Finestres')
print(f"Mitjana = {mitjana_th_finestres:.4f}, Std = {std_th_finestres:.4f}")
plt.figure(figsize=(6, 6))
sns.heatmap(cm_acumulada_finestres, annot=True, fmt='d', cmap='Blues', xticklabels=["No Presència", "Presència"], yticklabels=["No Presència", "Presència"])
plt.title("Matriu de Confusió Acumulada Thresholds Finestres")
plt.xlabel('Predicció')
plt.ylabel('Verdader')
plt.show()


#######################################################################################################################################################################################################################################################################################################################################################################################

with open('th_fred_model1.pkl', 'wb') as file:   
    pickle.dump(th_fred, file)                
with open('th_finestres_model1.pkl', 'wb') as file:   
    pickle.dump(th_finestres, file)     

