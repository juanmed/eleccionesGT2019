
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

import matplotlib.pyplot as plt

import params


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
    return img, success

def FuzzyMatch(word,matching_words):
    """
    Buscar el mejor matching de 'word' en una lista de palabras 'matching_words'
    De https://github.com/leaguilar/election_count_helper
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
    De https://github.com/leaguilar/election_count_helper
    """
    d = pytesseract.image_to_data(img, output_type=Output.DICT, config="-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    return d

def GetIndex(d,word,first=True):
    """
    Obtener indice de 'word' en el diccionario 'd'
    De https://github.com/leaguilar/election_count_helper
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
    De https://github.com/leaguilar/election_count_helper
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

        return coord, True

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

        return coord, True

    else:
        coord.append([x, y, w, h])
        return coord, False

        


if __name__ == '__main__':
    args = build_arg_parser()

    actas_dir = './datasets/2davuelta/sim/'
    actas_filenames = os.listdir(actas_dir)[:]

    t1 = time.time()

    titulo1 = 'ACTAFINALCIERREYESCRUTINIOS'
    titulo2 = 'PRESIDENTEYVICEPRESIDENTE'

    successes = 0
    successes_title1 = 0
    successes_title2 = 0

    ratio1_s = []   
    ratio2_s = []
    w1 = []
    w2 = []
    h1 = []
    h2 = []

    # buenas keywords siglas, ACTAFINALCIERREYESCRUTINIOS, PRESIDENTEYVICEPRESIDENTE
    keywords = ['OBSERVACIONES', 'PRESIDENTEYVICEPRESIDENTE']
    k1 = []
    k2 = []
    both = 0
    one = 0
    none  = 0
    for i, file in enumerate(actas_filenames):

        print("Imagen {}: {}".format(i,file))
        image = cv2.imread(actas_dir + file)
        image, success = FindResize(image, dw, dh)
        d = OCR(image.copy())
        
        
        coords, success = GetBoundingBox(image, d)
        if (success):
            print(" >> Se encontraron palabras")
        else:
            print(" >> No se encontraron palabras")
        coords = np.array(coords).reshape(-1,4)

        for j, coor in enumerate(coords):
            (x,y,w,h) = (coor[0], coor[1], coor[2], coor[3])
            if success:
                c1 = np.random.randint(0,256)
                c2 = np.random.randint(0,256)
                c3 = np.random.randint(0,256)
                #image = cv2.circle(image, (x,y), 3, (0,255,0), 2)
                image = cv2.rectangle(image,(x, y),(x + w, y + h),(c1, c2, c3),3)
                image = cv2.putText(image,str(j),(x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 2,(c1, c2, c3), 2, cv2.LINE_AA)
            else:
                #image = cv2.circle(image, (x,y), 6, (0,0,255), 2)
                image = cv2.rectangle(image,(x,y),(x + w, y + h),(0,0,255),3)

        bbox = np.mean(coords, axis = 0)
        (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)

        cv2.imshow('fig',cv2.resize(image, None, fx=0.5, fy = 0.5))
        cv2.waitKey()
        cv2.destroyAllWindows()        
        
    """
    kw_found = 0
    for i, keyword in enumerate(keywords): 
        success, index, ratio, matching_word = FuzzyMatch(keyword, d['text'])
        if success:
            kw_found = kw_found + 1
            if (i == 0):
                k1.append(ratio)
            elif (i == 1):
                k2.append(ratio)
            else:
                continue

    if (kw_found == 2):
        both  = both + 1
    elif (kw_found == 1):
        one = one + 1
    elif(kw_found == 0):
        none = none + 1
    else:
        continue
        
    print("{} se encontro en {}, {}% actas con un avg ratio {} std {}  max {} min {}".format(keywords[0], len(k1), 100.0*len(k1)/len(actas_filenames), np.mean(k1), np.std(k1), np.min(k1), np.max(k1)))
    print("{} se encontro en {}, {}% actas con un avg ratio {} std {}  max {} min {}".format(keywords[1], len(k2), 100.0*len(k2)/len(actas_filenames), np.mean(k2), np.std(k2), np.min(k2), np.max(k2)))
    print("Se encontro ambas en {} actas {}%".format(both, 100.0*both/len(actas_filenames)))
    print("Se encontro solo 1 palabra en {} actas {}%".format(one, 100.0*one/len(actas_filenames)))
    print("No se encontro nada en {} actas {}%".format(none, 100.0*none/len(actas_filenames)))
    """



    """



        success1, index, ratio = FuzzyMatch(titulo1, d['text'])
        if success1:
            successes_title1 = successes_title1 + 1
            ratio1_s.append(ratio)
            w1.append(d['width'][index])
            h1.append(d['height'][index])
        else:
            print('No se encontro {}\n'.format(titulo1))
            print(d['text'])


        success2, index, ratio = FuzzyMatch(titulo2, d['text'])
        if success2:
            successes_title2 = successes_title2 + 1
            ratio2_s.append(ratio)
            h2.append(d['height'][index])
            w2.append(d['width'][index])
        else:
            print('No se encontro {}\n'.format(titulo2))
            print(d['text'])

        if success1 and success2:
            successes = successes + 1

    t2 = time.time()

    print("OCR Time: {:.4f}".format(t2 - t1))
    print("Se encontro ambas en {} de {} ({}%)".format(successes, len(actas_filenames), 100.0*successes/len(actas_filenames)))
    print("Se encontro {} en {} de {} ({:.2f}%)".format(titulo1 ,successes_title1, len(actas_filenames), 100.0*successes_title1/len(actas_filenames)))
    print("    Width  avg {}  std {} min {} max {}".format(np.mean(w1), np.std(w1),  np.min(w1), np.max(w1)))
    print("    Height avg {}  std {} min {} max {}".format(np.mean(h1), np.std(h1),  np.min(h1), np.max(h1)))
    print("Se encontro {} en {} de {} ({:.2f}%)".format(titulo2 ,successes_title2, len(actas_filenames), 100.0*successes_title2/len(actas_filenames)))
    print("    Width  avg {}  std {} min {} max {}".format(np.mean(w2), np.std(w2),  np.min(w2), np.max(w2)))
    print("    Height avg {}  std {} min {} max {}".format(np.mean(h2), np.std(h2),  np.min(h2), np.max(h2)))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    
    actas_eval = ['295.jpg', '120.jpg']
    for acta in actas_eval:
        image = cv2.imread(actas_dir + acta)
        image, success = FindResize(image, dw, dh)
        d = ExtractData(image.copy())
        print(d['text'])
        success, word = FuzzyMatch(titulo1, d['text'])
        if success:
            print("Se encontro {}".format(word))
        else:
            print("No se encontro")
        image = DrawBoxes(image,d)
    """

    #ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #plt.show()
    #i = input('Presionar para ver {}'.format(acta))

