
import sys
import argparse
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import imutils
from imutils import perspective
import numpy as np
import pytesseract
from pytesseract import Output
from fuzzywuzzy import fuzz 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import json

import torch  # comentar esta linear para correr el modelo con keras
import params


model_save_path = "./weights/"
weight_file = "mnist_svhn_resnet_gtmnist_"
out_image_save_path = "./output/images/"
out_json_save_path = "./output/json/"
if not os.path.exists(out_image_save_path):
    os.mkdir(out_image_save_path)
if not os.path.exists(out_json_save_path):
    os.mkdir(out_json_save_path)

max_size = 1500

# default width and height, pixels
dw = 750
dh = 1000

ar = 1000.0/750.0 # page aspect ratio
w = 0.05          # aspect ratio range
ar_low = (1.0 - w)*ar
ar_high = (1.0 +w)*ar

def build_arg_parser():
    """
    Build command line argument parser
    Return:
        Dictionary containing arguments and their values
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = False, help = "Path to the image")
    ap = ap.parse_args()
    return ap  

def FindResize(img, w, h):
    """
    Encontrar las 4 esquinas de la hoja del acta, 'warpearla',
    redimensionar
    """ 
    success = False

    scale = max_size / float(max(img.shape))
    img = cv2.resize(img, None, fx=scale, fy = scale) 

    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    #gray = cv2.dilate(gray, None, iterations = 3)
    edged = cv2.Canny(gray, 50, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    rects = [cv2.boxPoints(cv2.minAreaRect(cnt)).astype('int') for cnt in cnts]

    rects = sorted(rects, key = cv2.contourArea, reverse = True)[:5]
    #return cv2.drawContours(img,rects,0,(0,0,255),4)

    # area ratio
    img_area = img.shape[0] * img.shape[1]
    acta_area = cv2.contourArea(rects[0])
    area_ratio = acta_area / img_area

    # aspect ratio
    mins = np.min(rects[1], axis = 0)
    maxs = np.max(rects[1], axis = 0)
    rect_h = maxs[1] - mins[1]
    rect_w = maxs[0] - mins[0]
    aspect_ratio = (1.0*rect_h)/rect_w

    
    if ((aspect_ratio > ar_low) and (aspect_ratio < ar_high) and (area_ratio > 0.5)):
        print("aspect_ratio: {}, area_ratio: {}".format(aspect_ratio, area_ratio))
        img = perspective.four_point_transform(img, rects[0].reshape(4, 2) * scale)
        success = True

    #return cv2.resize(img, (dw, dh), interpolation = cv2.INTER_AREA), success 
    return img, success, scale

def FuzzyMatch(word,matching_words):
    """
    Buscar el mejor matching de 'word' en una lista de palabras 'matching_words'
    Adaptado de https://github.com/leaguilar/election_count_helper
    """
    ratio  = 0
    for i, mw in enumerate(matching_words): 
        # verificar que la longitud sea al menos 50% similar
        if(len(mw) > int(0.5*len(word))): 
            ratio = fuzz.partial_ratio(word, mw)                  
            if ratio > 90:
                #print(" \n\n >>Hubo matching entre {} y {}: {:.2f}\n\n".format(word, mw, ratio))
                return True, i, ratio, mw
        else:
            continue
    return False, -1, ratio, mw

def OCR(img, whtlst = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"):   
    """
    Hacer OCR en la imagen.
    Adaptado de https://github.com/leaguilar/election_count_helper
    """
    whitelist = "-c tessedit_char_whitelist={}".format(whtlst)
    d = pytesseract.image_to_data(img, output_type=Output.DICT, config=whitelist)
    return d

def GetIndex(d,word,first=True):
    """
    Obtener indice de 'word' en el diccionario 'd'
    Adaptado de https://github.com/leaguilar/election_count_helper
    """
    if word in d['text']:
        if first:
            return d['text'].index(word)
        else:
            return len(d['text']) - 1 - d['text'][::-1].index(word)
    else:
        print("{} not in text".format(word))
        return -1

def DrawBoxes(img,d):
    """
    Helper function to draw boxes in a tesseract OCR dictionary
    Adaptado de https://github.com/leaguilar/election_count_helper
    """
    img2=img.copy()
    n_boxes = len(d['level'])
    for i in range(n_boxes):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img2

def GetKeywordIndex(d, keywords):
    """
    Obtener los indice en el diccionario d['text'] de una lista de frases 'keywords'
    """
    indices = []
    words = []
    for keyword in keywords:
        success, index, ratio, matching_word = FuzzyMatch(keyword, d['text'])
        if success:
            indices.append(index)
            words.append(matching_word)
    return indices, words

def GetBoundingBox(d, t1_x_offset, t1_y_offset, t2_x_offset, t2_y_offset):
    """
    """
    keywords = ['ACTAFINALCIERREYESCRUTINIOS', 'PRESIDENTEYVICEPRESIDENTE']

    # default values
    x = 350#225
    y = 650
    w = 320
    h = 80 


    coord = []

    idx, words = GetKeywordIndex(d, keywords)
    success = False
    if len(idx) == 2:
        t1_x = d['left'][idx[0]]
        t1_y = d['top'][idx[0]]
        t1_w = d['width'][idx[0]]
        t1_h = d['height'][idx[0]]

        p1_y = t1_y + t1_y_offset if (t1_y + t1_y_offset) > 0 else 0
        p1_x = (t1_x + t1_x_offset) if (t1_x + t1_x_offset) > 0 else 0 

        coord.append([p1_x, p1_y, w, h])

        t2_x = d['left'][idx[1]]
        t2_y = d['top'][idx[1]]
        t2_w = d['width'][idx[1]]
        t2_h = d['height'][idx[1]]

        p1_y = t2_y + t2_y_offset if (t2_y + t2_y_offset) > 0 else 0
        p1_x = t2_x + t2_x_offset if (t2_x + t2_x_offset) > 0 else 0 

        coord.append([p1_x, p1_y, w, h])
        success = True
        #return coord, True

    elif len(idx) == 1:
        
        word = d['text'][idx[0]]
        for i, keyword in enumerate(keywords):
            if( word[0] == keyword[0]):
                break

        t_x = d['left'][idx[0]]
        t_y = d['top'][idx[0]]
        t_w = d['width'][idx[0]]
        t_h = d['height'][idx[0]]

        if i == 0:

            p1_y = t_y + t1_y_offset if (t_y + t1_y_offset) > 0 else 0
            p1_x = (t_x + t1_x_offset) if (t_x + t1_x_offset) > 0 else 0 
            p1_w = int(params.bbox_w * t_w / params.t1_w )
            p1_h = int(params.bbox_h * t_h / params.t1_h) 
            coord.append([p1_x, p1_y, w, h])

        elif i == 1 :

            p1_y = t_y + t2_y_offset if (t_y + t2_y_offset) > 0 else 0
            p1_x = t_x + t2_x_offset if (t_x + t2_x_offset) > 0 else 0
            p1_w = int(params.bbox_w * t_w / params.t2_w )
            p1_h = int(params.bbox_h * t_h / params.t2_h) 
            coord.append([p1_x, p1_y, w, h])

        else:
            print("Imposible keyword")

        success = True
        #return coord, True

    else:
        coord.append([x, y, w, h])
        success = False
        #return coord, False

    coord = np.array(coord).reshape(-1,4)
    coord = np.mean(coord, axis = 0)
    return coord, success

def GetBoundingBoxPapeletas(d):
    """
    """
    keywords = ['ACTAFINALCIERREYESCRUTINIOS', 'PRESIDENTEYVICEPRESIDENTE']

    # default values
    x = 350#225
    y = 650
    w = 320
    h = 80 
                              # Valores para 2da vuelta simuladas
    t1_x_offset = -350        # -200
    t1_y_offset = 370         # 350
    t2_x_offset = -240         # -70
    t2_y_offset = 40          # 20

    coord = []

    idx, words = GetKeywordIndex(d, keywords)
    success = False
    if len(idx) == 2:
        t1_x = d['left'][idx[0]]
        t1_y = d['top'][idx[0]]
        t1_w = d['width'][idx[0]]
        t1_h = d['height'][idx[0]]

        p1_y = t1_y + t1_y_offset #int(params.t1_p1_h_offset * t1_h / params.t1_h) + t1_y
        p1_x = (t1_x + t1_x_offset) if (t1_x + t1_x_offset) > 0 else 0 #int(params.t1_p1_w_offset * t1_w / params.t1_w) + t1_x

        coord.append([p1_x, p1_y, w, h])

        t2_x = d['left'][idx[1]]
        t2_y = d['top'][idx[1]]
        t2_w = d['width'][idx[1]]
        t2_h = d['height'][idx[1]]

        p1_y = t2_y + t2_y_offset 
        p1_x = t2_x + t2_x_offset if (t2_x + t2_x_offset) > 0 else 0 

        coord.append([p1_x, p1_y, w, h])
        success = True
        #return coord, True

    elif len(idx) == 1:
        
        word = d['text'][idx[0]]
        for i, keyword in enumerate(keywords):
            if( word[0] == keyword[0]):
                break

        t_x = d['left'][idx[0]]
        t_y = d['top'][idx[0]]
        t_w = d['width'][idx[0]]
        t_h = d['height'][idx[0]]

        if i == 0:

            p1_y = t_y + t1_y_offset if (t_y + t1_y_offset) > 0 else 0#int(params.t1_p1_h_offset * t_h / params.t1_h) + t_y
            p1_x = (t_x + t1_x_offset) if (t_x + t1_x_offset) > 0 else 0 #int(params.t1_p1_w_offset * t_w / params.t1_w) + t_x
            p1_w = int(params.bbox_w * t_w / params.t1_w )
            p1_h = int(params.bbox_h * t_h / params.t1_h) 
            coord.append([p1_x, p1_y, w, h])

        elif i == 1 :

            p1_y = t_y + t2_y_offset if (t_y + t2_y_offset) > 0 else 0#int(params.t2_p1_h_offset * t_h / params.t2_h) + t_y
            p1_x = t_x + t2_x_offset if (t_x + t2_x_offset) > 0 else 0#int(params.t2_p1_w_offset * t_w / params.t2_w) + t_x
            p1_w = int(params.bbox_w * t_w / params.t2_w )
            p1_h = int(params.bbox_h * t_h / params.t2_h) 
            coord.append([p1_x, p1_y, w, h])

        else:
            print("Imposible keyword")

        success = True
        #return coord, True

    else:
        coord.append([x, y, w, h])
        success = False
        #return coord, False

    coord = np.array(coord).reshape(-1,4)
    coord = np.mean(coord, axis = 0)
    return coord, success

def GetBoundingBoxMesa(d):
    """
    """
    keywords = ['ACTAFINALCIERREYESCRUTINIOS', 'PRESIDENTEYVICEPRESIDENTE']

    # default values
    x = 350#225
    y = 650
    w = 200
    h = 100 

    t1_x_offset = 0    # 70
    t1_y_offset = 60    # 60
    t2_x_offset = 130   # 200
    t2_y_offset = -270  # -270

    coord = []

    idx, words = GetKeywordIndex(d, keywords)
    success = False
    if len(idx) == 2:
        t1_x = d['left'][idx[0]]
        t1_y = d['top'][idx[0]]
        t1_w = d['width'][idx[0]]
        t1_h = d['height'][idx[0]]

        p1_y = t1_y + t1_y_offset #int(params.t1_p1_h_offset * t1_h / params.t1_h) + t1_y
        p1_x = (t1_x + t1_x_offset) if (t1_x + t1_x_offset) > 0 else 0 #int(params.t1_p1_w_offset * t1_w / params.t1_w) + t1_x

        coord.append([p1_x, p1_y, w, h])

        t2_x = d['left'][idx[1]]
        t2_y = d['top'][idx[1]]
        t2_w = d['width'][idx[1]]
        t2_h = d['height'][idx[1]]

        p1_y = t2_y + t2_y_offset #int(params.t2_p1_h_offset * t2_h / params.t2_h) + t2_y
        p1_x = t2_x + t2_x_offset #int(params.t2_p1_w_offset * t2_w / params.t2_w) + t2_x

        coord.append([p1_x, p1_y, w, h])
        success = True
        #return coord, True

    elif len(idx) == 1:
        
        word = d['text'][idx[0]]
        for i, keyword in enumerate(keywords):
            if( word[0] == keyword[0]):
                break

        t_x = d['left'][idx[0]]
        t_y = d['top'][idx[0]]
        t_w = d['width'][idx[0]]
        t_h = d['height'][idx[0]]

        if i == 0:

            p1_y = t_y + t1_y_offset #int(params.t1_p1_h_offset * t_h / params.t1_h) + t_y
            p1_x = (t_x + t1_x_offset) if (t_x + t1_x_offset) > 0 else 0 #int(params.t1_p1_w_offset * t_w / params.t1_w) + t_x
            p1_w = int(params.bbox_w * t_w / params.t1_w )
            p1_h = int(params.bbox_h * t_h / params.t1_h) 
            coord.append([p1_x, p1_y, w, h])

        elif i == 1 :

            p1_y = t_y + t2_y_offset #int(params.t2_p1_h_offset * t_h / params.t2_h) + t_y
            p1_x = t_x + t2_x_offset #int(params.t2_p1_w_offset * t_w / params.t2_w) + t_x
            p1_w = int(params.bbox_w * t_w / params.t2_w )
            p1_h = int(params.bbox_h * t_h / params.t2_h) 
            coord.append([p1_x, p1_y, w, h])

        else:
            print("Imposible keyword")

        success = True
        #return coord, True

    else:
        coord.append([x, y, w, h])
        success = False
        #return coord, False

    coord = np.array(coord).reshape(-1,4)
    coord = np.mean(coord, axis = 0)
    return coord, success

def GetBoundingBoxTotales(d):
    """
    """
    keywords = ['ACTAFINALCIERREYESCRUTINIOS', 'PRESIDENTEYVICEPRESIDENTE']# , 'PARTIDOA', 'PARTIDOB']

    # default values
    x = 350#225
    y = 650
    w = 240
    h = 530 

    t1_x_offset = 5        #25
    t1_y_offset = 450       #470
    t2_x_offset = 185       #205
    t2_y_offset = 105       #125

    coord = []

    idx, words = GetKeywordIndex(d, keywords)
    success = False
    if len(idx) == 2:
        t1_x = d['left'][idx[0]]
        t1_y = d['top'][idx[0]]
        t1_w = d['width'][idx[0]]
        t1_h = d['height'][idx[0]]

        p1_y = t1_y + t1_y_offset #int(params.t1_p1_h_offset * t1_h / params.t1_h) + t1_y
        p1_x = (t1_x + t1_x_offset) if (t1_x + t1_x_offset) > 0 else 0 #int(params.t1_p1_w_offset * t1_w / params.t1_w) + t1_x
        p1_w = int(params.bbox_w * t1_w / params.t1_w )
        p1_h = int(params.bbox_h * t1_h / params.t1_h) 

        coord.append([p1_x, p1_y, w, h])

        t2_x = d['left'][idx[1]]
        t2_y = d['top'][idx[1]]
        t2_w = d['width'][idx[1]]
        t2_h = d['height'][idx[1]]

        p1_y = t2_y + t2_y_offset #int(params.t2_p1_h_offset * t2_h / params.t2_h) + t2_y
        p1_x = t2_x + t2_x_offset #int(params.t2_p1_w_offset * t2_w / params.t2_w) + t2_x
        p1_w = int(params.bbox_w * t2_w / params.t2_w )
        p1_h = int(params.bbox_h * t2_h / params.t2_h )

        coord.append([p1_x, p1_y, w, h])
        success = True
        #return coord, True

    elif len(idx) == 1:
        
        word = d['text'][idx[0]]
        for i, keyword in enumerate(keywords):
            if( word[0] == keyword[0]):
                break

        t_x = d['left'][idx[0]]
        t_y = d['top'][idx[0]]
        t_w = d['width'][idx[0]]
        t_h = d['height'][idx[0]]

        if i == 0:

            p1_y = t_y + t1_y_offset #int(params.t1_p1_h_offset * t_h / params.t1_h) + t_y
            p1_x = (t_x + t1_x_offset) if (t_x + t1_x_offset) > 0 else 0 #int(params.t1_p1_w_offset * t_w / params.t1_w) + t_x
            p1_w = int(params.bbox_w * t_w / params.t1_w )
            p1_h = int(params.bbox_h * t_h / params.t1_h) 
            coord.append([p1_x, p1_y, w, h])

        elif i == 1 :

            p1_y = t_y + t2_y_offset #int(params.t2_p1_h_offset * t_h / params.t2_h) + t_y
            p1_x = t_x + t2_x_offset #int(params.t2_p1_w_offset * t_w / params.t2_w) + t_x
            p1_w = int(params.bbox_w * t_w / params.t2_w )
            p1_h = int(params.bbox_h * t_h / params.t2_h) 
            coord.append([p1_x, p1_y, w, h])

        else:
            print("Imposible keyword")

        success = True
        #return coord, True

    else:
        coord.append([x, y, w, h])
        success = False
        #return coord, False

    coord = np.array(coord).reshape(-1,4)
    coord = np.mean(coord, axis = 0)
    return coord, success
        
def ExtractRectangles(image, type_flag = cv2.RETR_EXTERNAL):
    """
    Obtener rectangular que encierran a cada digito
    Adaptado de https://github.com/leaguilar/election_count_helper
    """
    # Find contours in the image
    ctrs, hier = cv2.findContours(image, type_flag, cv2.CHAIN_APPROX_SIMPLE)
    # Get only parent rectangles that contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    return rects

def removeChildrenRects(rects):
    """
    Remover rectangulos que estan dentro de un rectangulo mas grande y dejar
    solo los mas externos
    """
    cleaned_rects_vcenter=[]
    cleaned_rects_hcenter=[]
    only_external = []
    is_external = []
    for i in range(len(rects)):
        cx, cy, cw, ch = rects[i]
        has_parent = False
        for j in range(len(rects)):
            if (j != i):
                x,y,w,h = rects[j]
                if ( (cx >= x) and (cy >= y) and ((cx + cw) <= (x+w)) and ((cy + ch) <= (y + h))):
                    has_parent = True
                    break
                #else:
                #    continue
            #else:
            #    continue

        if(has_parent):
            is_external.append(False)
        else:
            is_external.append(True)

    for i in range(len(rects)):
        if(is_external[i]):
            only_external.append(rects[i])
            cleaned_rects_vcenter.append(rects[i][1] + rects[i][3]//2)
            cleaned_rects_hcenter.append(rects[i][0] + rects[i][2]//2)

    cleaned_rects_vcenter=np.array(cleaned_rects_vcenter).reshape(-1,1)
    cleaned_rects_hcenter=np.array(cleaned_rects_hcenter).reshape(-1,1)

    return only_external, cleaned_rects_vcenter, cleaned_rects_hcenter

def CleanRectangles(rects, h_min=35, w_min=25, h_max = 75, w_max = 75):
    """
    Eliminar rectangulos que son muy pequenos para ser digitos.
    Los valores por defecto: h_min=25, w_min=25, h_max = 75, w_max = 75
    parecen funcionar bien.
    Adaptado de https://github.com/leaguilar/election_count_helper
    """
    cleaned_rects=[]
    cleaned_rects_vcenter=[]
    cleaned_rects_hcenter=[]
    widths = []
    heights = []

    for j, rect in enumerate(rects):
        # Draw the rectangles
        if (( h_min <= rect[3] <= h_max) and ( w_min <= rect[2] <= w_max)):
            cleaned_rects.append(rect)
            cleaned_rects_vcenter.append(rect[1] + rect[3]//2)
            cleaned_rects_hcenter.append(rect[0] + rect[2]//2)
            widths.append(rect[2])
            heights.append(rect[3])
    #print(widths)
    mean_width = int(np.mean(widths))
    mean_heigth = int(np.mean(heights))
    cleaned_rects_vcenter=np.array(cleaned_rects_vcenter).reshape(-1,1)
    cleaned_rects_hcenter=np.array(cleaned_rects_hcenter).reshape(-1,1)
    return (cleaned_rects,cleaned_rects_vcenter,cleaned_rects_hcenter, mean_width, mean_heigth)

def GetLabels(data,nclust):
    """
    Agrupar centroides de cada rectangulo
    Adaptado de https://github.com/leaguilar/election_count_helper
    """
    labels=[]  
    kmeans = KMeans(n_clusters=nclust, random_state=0).fit(data)
    means=kmeans.cluster_centers_.mean(axis=1)
    idx = np.argsort(means)
    lut = np.zeros_like(idx)
    lut[idx] = np.arange(nclust)
    for i in range(len(kmeans.labels_)):
        labels.append(lut[kmeans.labels_[i]])
    return (labels,np.sort(means))

def GetStandardRects(w,h, x_ps, y_ps):
    """
    Definir una ubicacion 'standard' de los digitos en base a los
    rectangulos detectados. Esto permite colocar rectangulos incluso
    en digitos que no pudieron ser detectados.
    """
    standard_rects = []
    totals_lbl = []
    digit_lbl = []
    for k, y_pos  in enumerate(np.sort(y_ps)):
        for j, x_pos in enumerate(np.flip(np.sort(x_ps))):
            x = int(x_pos - w//2)
            y = int(y_pos - h//2)
            standard_rects.append((x, y, w, h ))
            totals_lbl.append(k)
            digit_lbl.append(j)

    return standard_rects, totals_lbl, digit_lbl

def resizeRect(rect, scale, imgshape):
    """
    Redimensionar un rectangulo por un factor 'scale'
    """
    h, w, channels = imgshape
    # Make the rectangular region around the digit
    leng = int(rect[3] * scale)
    y = int(rect[1] + rect[3] // 2 - leng // 2)
    x = int(rect[0] + rect[2] // 2 - leng // 2)
    if y < 0:
        y=0
    if x < 0:
        x=0
    return (x, y, leng, leng)

def predictPytorchModel(roi, model, img_size):
    """
    Hacer prediccion utilizando MNISTResNet
    """
    from torchvision import transforms
    # verificar si hay GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(" >> Found {} device".format(device))

    model = model.to(device)
    model.eval()

    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ret, roi = cv2.threshold(roi, 90, 255, cv2.THRESH_BINARY_INV)

    svhn_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        #transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])
    with torch.no_grad():

        roi = svhn_transform(roi).unsqueeze(0)
        roi = roi.to(device)
        prediction = model(roi)
        #print(prediction)
        val = np.argmax(prediction.to('cpu').numpy())

    roi = roi.squeeze(0).permute(1,2,0).to('cpu').numpy()
    return val, roi

def predictKerasModel(roi, model, img_size):
    """
    Hacer prediccion utilizando WideResNet28_10
    """
    #Resize the image
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ret, roi = cv2.threshold(roi, 90, 255, cv2.THRESH_BINARY_INV)
    roi = cv2.resize(roi, (img_size, img_size), interpolation=cv2.INTER_AREA)
    #roi = cv2.dilate(roi, (3, 3))
    #roi = cv2.erode(roi, (6, 6))
    roi = roi / 255

    ext_digit = roi.reshape(1,28,28,1)
    prediction= model.predict(ext_digit, verbose = 0)
    print(prediction)
    val = np.argmax(prediction[0])
    return val, roi

def GetNumericTotals(img, rects, total_lbl, digit_lbl, model, resize = 1.2):
    """
    Given an image, digits bounding box coordinates, and their labels,
    compute and return a list with the detected totales
    Adaptado de https://github.com/leaguilar/election_count_helper
    """

    img_size = 28 # for Keras,    32 for pytorch

    totals = [0]*(max(total_lbl)+1)
    img_n = img.copy()
    a = np.zeros((img_size,img_size,1))

    for i, rect in enumerate(rects):

        x,y,w,h = resizeRect(rect, resize, img.shape)
        roi = img[y:y+h, x:x+h]

        val, roi = predictPytorchModel(roi,model, img_size)
        #val, roi = predictKerasModel(roi.copy(), model, img_size)

        totals[total_lbl[i]] += val*(10**digit_lbl[i])
        img_n = cv2.rectangle(img_n,(x, y),(x + w, y + h),(0, 255, 0),3)
        img_n = cv2.putText(img_n, str(val), (x ,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)        
        #img_n = cv2.putText(img_n, str(digit_lbl[i]), (x ,y), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2,cv2.LINE_AA)
        #img_n = cv2.putText(img_n, str(total_lbl[i]), (10 ,y), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),2,cv2.LINE_AA)
        roi =  roi.reshape(img_size,img_size,-1) #cv2.resize(roi,(img_size,img_size))
        a = np.concatenate((a,roi), axis = 0)

    #img_n = a
    return totals, img_n

def loadPytorchModel():
    """
    Cargar modelo hecho en Pytorch
    """
    import torch
    from models.mnist_resnet import MNISTResNet
    model = MNISTResNet()

    # load weights and state
    checkpoint = torch.load(model_save_path+weight_file+"180.pt", map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print("Loaded {}".format(model_save_path+weight_file+"180.pt"))
    return model 

def loadKerasModel():
    """
    Cargar un modelo hecho con Keras
    """
    from utils2 import load_mnist
    from models.wide_resnet_28_10 import WideResNet28_10

    model_name = "WideResNet28_10"
    model=WideResNet28_10()
    model.compile()
    model.load_weights(model_save_path + model_name + '.h5')
    print("Loaded {}".format(model_save_path + model_name + '.h5'))
    return model

def getVotosData(img,d, model, scale):
    """
    Obtener un crop de todos los totales de votos (partidos, validos, invalidos, etc),
    reconocer los digitos y dichos totales, y devolver los datos, junto con un crop con
    los datos superpuestos y una bandera de problemas.
    """

    problem = False
    coords, success = GetBoundingBoxTotales(d)#, 25, 470, 205, 125)
    coords = coords*(1.0/scale)

    (tx, ty, tw, th) = (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))
    totales_crop = img[ty:ty+th, tx:tx+tw]
    
    gray = cv2.cvtColor(totales_crop.copy(), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray.copy(), 50, 200, 5)
    #ret, clean = cv2.threshold(gray.copy(), 0, 255, cv2.THRESH_OTSU)
    rects = ExtractRectangles(edges)
    (cleaned_rects,cleaned_rects_vcenter,cleaned_rects_hcenter, avg_w, avg_h) = CleanRectangles(rects)
    cleaned_rects,cleaned_rects_vcenter,cleaned_rects_hcenter = removeChildrenRects(cleaned_rects)

    ntotales=7
    ndigitos=3
    if (len(cleaned_rects_vcenter)>ntotales):
        totals_lbl,totals_mean=GetLabels(cleaned_rects_vcenter,ntotales)
    else:
        print(" >>Se esperan {} pero se encontraron {} totales".format(ntotales, len(cleaned_rects_vcenter)))
        return 0,0,-1,0
    
    if (len(cleaned_rects_vcenter)>ntotales):
        digit_lbl,digit_mean=GetLabels(cleaned_rects_hcenter,ndigitos)
        digit_lbl = [-1*lbl+2 for lbl in digit_lbl]  # convertir digit_lbl a las correctas potencias de 10
    else:
        print(" >> Se esperan {} pero se encontraron {} digitos".format(ndigitos, len(cleaned_rects_hcenter)))
        return 0,0,-2,0

    totals_problem, tdist = bad_cluster(totals_mean, avg_h//2)
    digits_problem, ddist = bad_cluster(digit_mean, avg_w//2)
    if( totals_problem or digits_problem):
        print("Bad Clustering in totales: {}  digits: {}".format(totals_problem, digits_problem))
        #print("Totales dist: {} digits dist: {}".format(tdist, ddist))
        #print("avg height: {} avg widht: {}".format(avg_h, avg_w))
        return 0,0,-3,0, 

    cleaned_rects, totals_lbl, digit_lbl = GetStandardRects(avg_w, avg_h, digit_mean, totals_mean)
    totales, img_totales = GetNumericTotals(totales_crop.copy(), cleaned_rects, totals_lbl, digit_lbl, model)
    
    return totales, img_totales, problem, coords

def getMesaData(img, d, scale):
    """
    Obtener el crop que contiene el numero de mesa, municipio y departamento,
    identificar dichos numeros y devolverlos junto con el crop con los datos
    identificados superpuestos.
    """

    coords, success = GetBoundingBoxMesa(d)#, 0, 60, 130, -270)
    coords = coords*(1.0/scale)
    (mx, my, mw, mh) = (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))

    mesa_crop = img[my:my+mh, mx:mx+mw]
    d = OCR(mesa_crop.copy(), "0123456789")
    mesa_data_raw = []
    pass_next = False
    for j,element in enumerate(d['text']):

        if pass_next:
            pass_next = False
            continue
        else:
            # Si se detecto un elemento de 2 caracteres, seguro es uno de los valores
            if (len(element)) == 2:
                mesa_data_raw.append((j,element))

            # a veces se detectan los digitos por separado, intentar capturar 
            elif( len(element) == 1):
                if( (j+1) != len(d['text'])):
                    if (len(d['text'][j+1]) == 1):
                        val = element+d['text'][j+1]
                        mesa_data_raw.append((j, val))
                        pass_next = True
                    else:
                        continue
                else:
                    continue
            else:
                pass_next = False

    mesa_mun_dep =[int(ele[1]) for ele in mesa_data_raw]
    for i, val in mesa_data_raw:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        mesa_crop = cv2.putText(mesa_crop, val, (x ,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)        

    return mesa_mun_dep,mesa_crop,0,coords

def getImpugnacionesData(img, d, scale):
    return 0,0,0,0

def getPapeletasRecibidas(img, d, model,scale):
    """
    Obtener el total de papeletas recibidas
    """
    coords, success = GetBoundingBox(d, -200, 350, -70, 20)
    coords = coords*(1.0/scale)
    (px, py, pw, ph) = (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))
    papeletas_crop = img[py:py+ph, px:px+pw]

    gray = cv2.cvtColor(papeletas_crop.copy(), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray.copy(), 50, 200, 5)
    #ret, clean = cv2.threshold(gray.copy(), 0, 255, cv2.THRESH_OTSU)
    rects = ExtractRectangles(edges)
    rects = sorted(rects, key = (lambda x: x[2]*x[3]), reverse = True)

    # If the digits appear inside the box
    (dx, dy, dw, dh) = rects[0]
    digits_crop = papeletas_crop[ dy:dy+dh, dx:dx+dw ]
    gray = cv2.cvtColor(digits_crop.copy(), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray.copy(), 50, 200, 5)
    rects = ExtractRectangles(edges, type_flag = cv2.RETR_LIST)


    (cleaned_rects,cleaned_rects_vcenter,cleaned_rects_hcenter, avg_w, avg_h) = CleanRectangles(rects, h_min = 20, w_min = 15)
    #cleaned_rects = sorted(cleaned_rects, key = (lambda x: x[2]*x[3]), reverse = True)[0:3]
    #cleaned_rects,cleaned_rects_vcenter,cleaned_rects_hcenter = removeChildrenRects(cleaned_rects)  



    ntotales=1
    ndigitos=3
    if (len(cleaned_rects_vcenter)>ntotales):
        totals_lbl,totals_mean=GetLabels(cleaned_rects_vcenter,ntotales)
    else:
        print(" >>Se esperan {} pero se encontraron {} totales".format(ntotales, len(cleaned_rects_vcenter)))
        return 0,0,-1,0
    
    if (len(cleaned_rects_vcenter)>ntotales):
        digit_lbl,digit_mean=GetLabels(cleaned_rects_hcenter,ndigitos)
        digit_lbl = [-1*lbl+2 for lbl in digit_lbl]  # convertir digit_lbl a las correctas potencias de 10
    else:
        print(" >> Se esperan {} pero se encontraron {} digitos".format(ndigitos, len(cleaned_rects_hcenter)))
        return 0,0,-2,0

    cleaned_rects, totals_lbl, digit_lbl = GetStandardRects(avg_w, avg_h, digit_mean, totals_mean)
    cleaned_rects = [(r[0]+dx, r[1]+dy, r[2], r[3]) for r in cleaned_rects ]
    totales, img_totales = GetNumericTotals(papeletas_crop.copy(), cleaned_rects, totals_lbl, digit_lbl, model, 1.2 )


    #for rect in cleaned_rects:
    #    (x, y, w, h) = rect #rects[0]
    #    cv2.rectangle(digits_crop, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #(x, y, w, h) = rects[0]
    #cv2.rectangle(papeletas_crop, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return totales,img_totales,0,coords

def processActa(img , model):
    """
    """

    data = {}

    image, success, scale = FindResize(img.copy(), dw, dh)
    d = OCR(image.copy())

    totales, totales_img, totales_problem, tcoords = getVotosData(img.copy(), d, model, scale)
    mesa_data, mesa_img, mesa_problem, mcoords = getMesaData(img.copy(),d, scale)
    #impug_data, impug_img, impug_problem, icoords = getImpugnacionesData(img.copy(),d, scale)
    papel_data, papel_img, papel_problem, pcoords = getPapeletasRecibidas(img.copy(), d, model, scale)

    data['partidoa'] = totales[0]
    data['partidob'] = totales[1]
    data['validos'] = totales[2]
    data['nulos'] = totales[3]
    data['blancos'] = totales[4]
    data['emitidos'] = totales[5]
    data['invalidos'] = totales[6]
    data['mesa'] = mesa_data[0]
    data['municipio'] = mesa_data[1]
    data['departamento'] = mesa_data[2]
    data['papeletas_recibidas'] = papel_data[0]


    tx,ty,tw,th = (int(tcoords[0]), int(tcoords[1]), int(tcoords[2]), int(tcoords[3]))
    img[ty:ty+th,tx:tx+tw] = totales_img
    
    mx,my,mw,mh = (int(mcoords[0]), int(mcoords[1]), int(mcoords[2]), int(mcoords[3]))
    img[my:my+mh,mx:mx+mw] = mesa_img
    
    px,py,pw,ph = (int(pcoords[0]), int(pcoords[1]), int(pcoords[2]), int(pcoords[3]))
    img[py:py+ph,px:px+pw] = papel_img


    problem = 1
    return data, img, problem

def bad_cluster(means, threshold):
    """
    """
    min_dist = 2000
    means = np.sort(means).tolist()
    for i in range(len(means)-1):
        dist = means[i + 1] - means[i]
        if(dist < min_dist):
            min_dist = dist

    if(min_dist <= threshold):
        return True, min_dist
    else:
        return False, min_dist

if __name__ == '__main__':

    actas_dir = './datasets/2davuelta/sim/'
    actas_dir_primera = './actas_original/presi_vice/'
    actas_filenames = os.listdir(actas_dir)[:]

    model = loadPytorchModel()
    #model = loadKerasModel()
    times = []
    success = 0
    for i, file in enumerate(actas_filenames):

        print("Imagen {}: {}".format(i,file))
        original = cv2.imread(actas_dir + file)
        try:
            t1 = time.time()
            data, img_with_data, problem = processActa(original.copy(), model)
            t2 = time.time()

            times.append(t2 - t1)
            
            if problem < 0:
                if (problem == -1):
                    print(" >> No se reconocion la cantidad esperada de totales")
                elif(problem == -2):
                    print(" >> No se encontro la cantidad esperada de digitos")
                else:
                    print("No existe este codigo de problem.")
            else:
                json_data = {}
                for key in data.keys():
                    json_data[key] = str(data[key])
                #print(json_data)
                json_name = file.split('.')[0] + '.json'
                with open(out_json_save_path + json_name, 'x') as json_file:
                    json.dump(json_data, json_file)

                #cv2.imshow('fig',cv2.resize(img_with_data, None, fx=1.0, fy = 1.0))
                #cv2.imwrite(out_image_save_path+file,cv2.resize(img_with_data, None, fx=0.4, fy = 0.4))
                #cv2.waitKey()
                #cv2.destroyAllWindows() 
                success = success+1
        except:
            print("  >> Could not process this image.")  

    print("Processing time: {:.3f}s avg, {:.3f}s std, {:.3f}s max, {:.3f}s min".format(np.mean(times), np.std(times), np.max(times), np.min(times)))     
    print("Succesfully processed: {}, ({:.2f}%)".format(success, 100*success/len(actas_filenames)))

"""

    for j, coor in enumerate(coords):
        (x,y,w,h) = (int(coor[0]), int(coor[1]), int(coor[2]), int(coor[3]))
        if success:
            c1 = np.random.randint(0,256)
            c2 = np.random.randint(0,256)
            c3 = np.random.randint(0,256)
            img = cv2.rectangle(img,(x, y),(x + w, y + h),(c1, c2, c3),3)
            img = cv2.putText(img,str(j),(x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 2,(c1, c2, c3), 2, cv2.LINE_AA)
        else:
            img = cv2.rectangle(img,(x,y),(x + w, y + h),(0,0,255),3)




    coords = np.mean(coords, axis = 0)
    (x, y, w, h) = (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('fig',img)
    cv2.waitKey()
    cv2.destroyAllWindows()
"""

# TODO
# -Confude 7 con 1, espacio vacio con 1, 
# -Remover el recuadro que encierra a los digitos. Tiene un efecto negativo en la prediccion
# -Anadir numeros testados al dataset para aunque sea indicar que un humano debe revisar el acta
# - Definir codigos para cada tipo de problema
# - Hacer algo como Non Minimum Max Supression en lugar de removeChildrenRects