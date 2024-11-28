#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:54:44 2024

@author: aina
"""

import pandas as pd
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
import pickle
import random
import cv2
 
def carpetes_negatives(fitxercsv):
    codis=[]
    with open(fitxercsv, 'r') as file:
        next(file)
        for line in file:
            codi, densitat = line.strip().split(',')
            if densitat == "NEGATIVA":
                codis.append(codi)
    return codis

def carpetes_totals(fitxercsv):
    codis={} 
    with open(fitxercsv, 'r') as file:
        next(file)
        for line in file:
            codi, densitat = line.strip().split(',')
            if densitat == "NEGATIVA": 
                codis[codi]=0
            else:
                codis[codi]=1 
    return codis
 

def load_cropped_negatives(carpetes, imgxfolder):
    imatges = []
    metadades = [] 
    ruta = Path('/Users/aina/Desktop/uni/4rt/psiv/repte 3/cross-validation/Cropped/')
    i=0
    for carpeta in ruta.iterdir():
        if carpeta.is_dir() and carpeta.name[:-2] in carpetes: 
            print(i)
            i+=1
            noms_imatges = [filename for filename in carpeta.iterdir() if filename.suffix.lower() in ['.png', '.jpg', '.jpeg']] 
            random.shuffle(noms_imatges)
            for filename in noms_imatges[:imgxfolder]:  
                img=cv2.imread(filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imatges.append(img) 
                metadades.append({ 
                    'PatID': carpeta.name[:-2],
                    'FileName': filename,
                    'Diagnostic': int(carpeta.name[-1])}) 
    with open('imatges_cropped_negatives.pkl', 'wb') as file:
        pickle.dump(imatges, file)
    with open('metadades_cropped_negatives.pkl', 'wb') as file:
        pickle.dump(metadades, file)  
    return imatges, metadades
 
 
def load_cropped_total(carpetes, imgxfolder):
    imatges = []
    metadades = {} 
    ruta = Path('/Users/aina/Desktop/uni/4rt/psiv/repte 3/cross-validation/Cropped/')
    i=0
    for carpeta in ruta.iterdir():
        if carpeta.is_dir() and carpeta.name[:-2] in carpetes.keys():  
            filenames_carpeta=[]
            noms_imatges = [filename for filename in carpeta.iterdir() if filename.suffix.lower() in ['.png', '.jpg', '.jpeg']] 
            random.shuffle(noms_imatges)
            for filename in noms_imatges[:imgxfolder]:  
                img=cv2.imread(filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imatges.append(img) 
                filenames_carpeta.append(filename)   
            metadades[carpeta.name[:-2]] = {'FileName': filenames_carpeta, 'Presence': carpetes[carpeta.name[:-2]]}
    with open('imatges_cropped_totals.pkl', 'wb') as file:
        pickle.dump(imatges, file)
    with open('metadades_cropped_totals.pkl', 'wb') as file:
        pickle.dump(metadades, file)  
    return imatges, metadades
 
 
def load_annotated(fitxerexcel):
    imatges_annotated = []
    metadades = {}
    target_size = (256, 256, 3)
    excel = pd.read_excel(fitxerexcel)
    ruta = Path('/Users/aina/Desktop/uni/4rt/psiv/repte 3/cross-validation/Annotated/')
    i=0
    for carpeta in ruta.iterdir():
        if carpeta.is_dir() and carpeta.name[:-2] in excel['Pat_ID'].values:   
            filenames_carpeta = []
            presencies = []
            for filename in carpeta.iterdir():
                if filename.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    img=cv2.imread(filename)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    imatges_annotated.append(img)  
                    presencia = int(excel.loc[excel['Pat_ID'] == carpeta.name[:-2], 'Presence'].iloc[0]) 
                    filenames_carpeta.append(presencia)
                    presencies.append(presencia)
                metadades[carpeta.name[:-2]] = {'FileName': filenames_carpeta, 'Presence': presencies}
    with open('imatges_annotated.pkl', 'wb') as file:   
        pickle.dump(imatges_annotated, file)                
    with open('metadades_annotated.pkl', 'wb') as file:   
        pickle.dump(metadades, file)                    
    return imatges_annotated, metadades


#codis_negatives = carpetes_negatives('/Users/aina/Desktop/uni/4rt/psiv/repte 3/cross-validation/PatientDiagnosis.csv')
codis_totals = carpetes_negatives('/Users/aina/Desktop/uni/4rt/psiv/repte 3/cross-validation/PatientDiagnosis.csv')
#imatges_cropped_negatives, metadades_cropped_negatives = load_cropped_negatives(codis_negatives, 50) 
imatges_cropped_totals, metadades_cropped_totals = load_cropped_total(codis_totals, 50)
imatges_annotated, metadades_annotated = load_annotated('/Users/aina/Desktop/uni/4rt/psiv/repte 3/cross-validation/HP_WSI-CoordAllAnnotatedPatches.xlsx')



