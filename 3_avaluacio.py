import pickle
import numpy as np 
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product


th_fred_path = '/Users/aina/Desktop/uni/4rt/psiv/repte 3/pickels/th_fred_model1.pkl' 
#th_fred_path = '/Users/aina/Desktop/uni/4rt/psiv/repte 3/pickels/th_fred_model2.pkl' 
with open(th_fred_path, 'rb') as f:
    ll_th_fred = pickle.load(f)  
th_finestres_path = '/Users/aina/Desktop/uni/4rt/psiv/repte 3/pickels/th_finestres_model1.pkl' 
#th_finestres_path = '/Users/aina/Desktop/uni/4rt/psiv/repte 3/pickels/th_finestres_model2.pkl' 
with open(th_finestres_path, 'rb') as f:
    ll_th_finestres = pickle.load(f)  

holdout_imatges_path = '/Users/aina/Desktop/uni/4rt/psiv/repte 3/pickels/imatges_holdout.pkl' 
with open(holdout_imatges_path, 'rb') as f:
    holdout_imatges = pickle.load(f)  
holdout_metadades_path = '/Users/aina/Desktop/uni/4rt/psiv/repte 3/pickels/metadades_holdout.pkl' 
with open(holdout_metadades_path, 'rb') as f:
    holdout_metadades = pickle.load(f)  
holdout_reconstruccions_path = '/Users/aina/Desktop/uni/4rt/psiv/repte 3/pickels/reconstruccions_holdout_model1.pkl' 
#holdout_reconstruccions_path = '/Users/aina/Desktop/uni/4rt/psiv/repte 3/pickels/reconstruccions_holdout_model2.pkl' 
with open(holdout_reconstruccions_path, 'rb') as f:
    holdout_reconstruccions = pickle.load(f)  


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

def avaluacio_th(presence_test, pred_test, th_fred, th_finestres):
    cm = confusion_matrix(presence_test, pred_test)   
    TN, FP, FN, TP = cm.ravel() 
    recall_positiu = TP / (TP + FN) if (TP + FN) > 0 else 0
    recall_negatiu = TN / (TN + FP) if (TN + FP) > 0 else 0
    precisio_positiva = TP / (TP + FP) if (TP + FP) > 0 else 0
    precisio_negativa = TN / (TN + FN) if (TN + FN) > 0 else 0 
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Presència", "Presència"], yticklabels=["No Presència", "Presència"])
    plt.title(f'Matriu de Confusió - Fred={th_fred} Finestres={th_finestres}')
    plt.xlabel('Predicció')
    plt.ylabel('Verdader')
    plt.show()  
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

def avaluacio_th2(presence_test, pred_test):
    cm = confusion_matrix(presence_test, pred_test)   
    TN, FP, FN, TP = cm.ravel() 
    recall_positiu = TP / (TP + FN) if (TP + FN) > 0 else 0
    recall_negatiu = TN / (TN + FP) if (TN + FP) > 0 else 0
    precisio_positiva = TP / (TP + FP) if (TP + FP) > 0 else 0
    precisio_negativa = TN / (TN + FN) if (TN + FN) > 0 else 0 
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Presència", "Presència"], yticklabels=["No Presència", "Presència"])
    plt.title(f'Matriu de Confusió MaxVolting')
    plt.xlabel('Predicció')
    plt.ylabel('Verdader')
    plt.show()  
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

ll_th_fred = list(set(ll_th_fred))
ll_th_finestres = list(set(ll_th_finestres))
millors_metriques = []
combinacions = list(product(ll_th_fred, ll_th_finestres))
resultats_pacients = []
presence = [valor["Presence"] for valor in holdout_metadades.values()]

for i, (th_fred, th_finestres) in enumerate(combinacions):
    fred = freq_red(holdout_imatges, holdout_reconstruccions)
    pred_fred = [1 if imatge >= th_fred else 0 for imatge in fred] 
    finestres = [sum(pred_fred[j:j + 50]) / 50 for j in range(0, len(pred_fred), 50)]
    pred_finestres = [1 if imatge >= th_finestres else 0 for imatge in finestres]
    resultats_pacients.append(pred_finestres)

    print('Threshold Fred:', th_fred)
    print('Threshold Finestres:', th_finestres)
    metriques_finestres = avaluacio_th(presence, pred_finestres, round(th_fred, 5), round(th_finestres, 5))
    metriques = {
        "th_fred": th_fred,
        "th_finestres": th_finestres,
        "metriques_finestres": metriques_finestres
    }
    millors_metriques.append(metriques)

percentatge_positius = []
for pacient in zip(*resultats_pacients):
    percentatge_positius.append(sum(pacient)/len(pacient))
resultats1 = [1 if pacient >= 0.5 else 0 for pacient in percentatge_positius] 
resultats1_metriques = avaluacio_th2(presence, resultats1)
print(resultats1_metriques)

'''
millors_metriques.sort(key=lambda x: x["metriques_finestres"]["recall"], reverse=True)
top_3 = millors_metriques[:3]
for idx, metrica in enumerate(top_3, start=1):
    print(f"Top {idx}:")
    print(f"  Threshold Fred: {metrica['th_fred']}")
    print(f"  Threshold Finestres: {metrica['th_finestres']}")
    print(f"  Mètriques FINESTRES: {metrica['metriques_finestres']}\n")

Si falsos negativos son más críticos:
    - Usa recall o sensibilidad.
    - Ejemplo: Diagnosticar cáncer, donde un falso negativo podría retrasar el tratamiento.

Si falsos positivos son más críticos:
    - Usa precisión o especificidad.
    - Ejemplo: Diagnosticar una enfermedad rara con tratamientos costosos e invasivos.
'''


























