
import sys
import argparse
import time
import os

import cv2
import imutils
from imutils import perspective
import numpy as np
import pytesseract
from pytesseract import Output
from fuzzywuzzy import fuzz 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



import params
#from models.mnist_resnet import MNISTResNet
from models.wide_resnet_28_10 import WideResNet28_10
from utils2 import load_mnist


model_save_path = "./weights/"
weight_file = "mnist_svhn_resnet"


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

def OCR(img):   
    """
    Hacer OCR en la imagen.
    Adaptado de https://github.com/leaguilar/election_count_helper
    """
    d = pytesseract.image_to_data(img, output_type=Output.DICT, config="-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
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

def GetBoundingBox(img, d):
    """
    """
    keywords = ['ACTAFINALCIERREYESCRUTINIOS', 'PRESIDENTEYVICEPRESIDENTE']# , 'PARTIDOA', 'PARTIDOB']

    # default values
    x = 350#225
    y = 650
    w = 150
    h = 550 

    t1_x_offset = 50
    t1_y_offset = 470
    t2_x_offset = 180 
    t2_y_offset = 125

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
        success = True
        #return coord, False

    coord = np.array(coord).reshape(-1,4)
    coord = np.mean(coord, axis = 0)
    return coord, success
        
def ExtractRectangles(image):
    """
    Obtener rectangular que encierran a cada digito
    Adaptado de https://github.com/leaguilar/election_count_helper
    """
    # Find contours in the image
    ctrs, hier = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get only parent rectangles that contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    return rects

def removeChildrenRects(rects):
    """
    Remover rectangulos que estan dentro de un rectangulo mas grande y dejar
    solo los mas externos
    """
    only_external = []
    is_external = []
    for i in range(len(rects)):
        cx, cy, cw, ch = rects[i]
        has_parent = False
        for j in range(len(rects)):
            if (j != i):
                x,y,w,h = rects[j]
                if ( (cx > x) and (cy > y) and ((cx + cw) < (x+w)) and ((cy + ch) < (y + h))):
                    has_parent = True
                    break
                else:
                    continue
            else:
                continue

        if(has_parent):
            is_external.append(False)
        else:
            is_external.append(True)

    for i in range(len(rects)):
        if(is_external[i]):
            only_external.append(rects[i])

    return only_external

def CleanRectangles(rects, h_min=50, w_min=50, h_max = 75, w_max = 75):
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
    print(" >> Found {} device".format(device))

    model = model.to(device)
    model.eval()

    svhn_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
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
    roi = cv2.resize(roi, (img_size, img_size), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    roi = cv2.erode(roi, (6, 6))
    roi = roi / 255

    ext_digit = roi.reshape(1,28,28,1)
    prediction= model.predict(ext_digit, verbose = 0)
    val = np.argmax(prediction[0])
    return val, roi

def GetNumericTotals(img, rects, total_lbl, digit_lbl, model):
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

        x,y,w,h = resizeRect(rect, 1.2, img.shape)
        roi = img[y:y+h, x:x+h]

        #val = predictPytorchModel(roi,model, img_size)
        val,roi = predictKerasModel(roi.copy(), model, img_size)

        totals[total_lbl[i]] += val*(10**digit_lbl[i])
        img_n = cv2.putText(img_n, str(val), (x ,y), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2,cv2.LINE_AA)
        roi =  roi.reshape(img_size,img_size,-1) #cv2.resize(roi,(img_size,img_size))
        a = np.concatenate((a,roi), axis = 0)

    img_n = a
    return totals, img_n

def loadPytorchModel():
    """
    Load the digit recognition model
    """
    import torch
    model = MNISTResNet()

    # load weights and state
    checkpoint = torch.load(model_save_path+weight_file+"200.pt", map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print("Loaded {}".format(model_save_path+weight_file+"200.pt"))
    return model 

def loadKerasModel():
    """
    """
    model_name = "WideResNet28_10"
    model=WideResNet28_10()
    model.compile()
    model.load_weights(model_save_path + model_name + '.h5')
    print("Loaded {}".format(model_save_path + model_name + '.h5'))
    return model

def processActa(img):
    """
    """
    image, success, scale = FindResize(img.copy(), dw, dh)
    d = OCR(image.copy())

    coords, success = GetBoundingBox(image, d)
    coords = coords*(1.0/scale)
    (x, y, w, h) = (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))

    totales_crop = img[y:y+h, x:x+w]
    gray = cv2.cvtColor(totales_crop.copy(), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray.copy(), 50, 200, 5)
    #ret, clean = cv2.threshold(gray.copy(), 0, 255, cv2.THRESH_OTSU)
    rects = ExtractRectangles(edges)
    (cleaned_rects,cleaned_rects_vcenter,cleaned_rects_hcenter, avg_w, avg_h) = CleanRectangles(rects)
    cleaned_rects = removeChildrenRects(cleaned_rects)

    ntotales=7
    ndigitos=3
    if (len(cleaned_rects_vcenter)>ntotales):
        totals_mean,totals_mean=GetLabels(cleaned_rects_vcenter,ntotales)
    else:
        print(" >>Se esperan {} pero se encontraron {} totales".format(ntotales, len(cleaned_rects_vcenter)))
        return (True,-3)
    
    if (len(cleaned_rects_vcenter)>ntotales):
        digit_lab,digit_mean=GetLabels(cleaned_rects_hcenter,ndigitos)
    else:
        print(" >> Se esperan {} pero se encontraron {} digitos".format(ndigitos, len(cleaned_rects_hcenter)))
        return (True,-4)

    cleaned_rects, totals_lbl, digit_lbl = GetStandardRects(avg_w, avg_h, digit_mean, totals_mean)

    totales, img_n = GetNumericTotals(totales_crop.copy(), cleaned_rects, totals_lbl, digit_lbl, model)
    return totales, img_n

if __name__ == '__main__':

    args = build_arg_parser()
    actas_dir = './datasets/2davuelta/sim/'
    actas_filenames = os.listdir(actas_dir)[:10]

    #model = loadPytorchModel()
    model = loadKerasModel()

    # buenas keywords siglas, ACTAFINALCIERREYESCRUTINIOS, PRESIDENTEYVICEPRESIDENTE
    keywords = ['OBSERVACIONES', 'PRESIDENTEYVICEPRESIDENTE']
    k1 = []
    k2 = []
    both = 0
    one = 0
    none  = 0
    for i, file in enumerate(actas_filenames):

        print("Imagen {}: {}".format(i,file))
        original = cv2.imread(actas_dir + file)

        totales, img_crop = processActa(original.copy())
        print (totales)
        cv2.imshow('fig',cv2.resize(img_crop, None, fx=1.0, fy = 1.0))
        cv2.waitKey()
        cv2.destroyAllWindows()        
        
