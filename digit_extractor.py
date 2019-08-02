
import os
import cv2
import imutils
import numpy as np
import pytesseract
from pytesseract import Output
from fuzzywuzzy import fuzz 
from sklearn.cluster import KMeans

import pandas as pd



class DigitExtractor():
    
    def __init__(self, max_size = 1500):

        # default width and height for analysis
        self.max_size = max_size

        ar = 1000.0/750.0 # page aspect ratio
        w = 0.05          # aspect ratio range
        self.ar_low = (1.0 - w)*ar
        self.ar_high = (1.0 +w)*ar

        self.output_size = 28

    def processarActa(self, img):
        """
        Extraer los digitos y sus etiquetas y almacenarlos.
        """
        success = True

        image, success, scale = self.FindResize(img.copy(), self.max_size)
        d = self.OCR(image.copy())

        cleaned_rects, totals_lbl, digit_lbl, coords = self.getDigitRectangles(img.copy(), d, scale)

        tx, ty, tw, th = (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))
        for i , rect in enumerate(cleaned_rects):
            x,y,w,h = self.resizeRect(rect, 1.2, img.shape)
            cleaned_rects[i] = (tx + x, ty + y, w, h)

        return cleaned_rects, totals_lbl, digit_lbl

    def resizeRect(self, rect, scale, imgshape):
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

    def getDigitRectangles(self, img, d, scale):
        problem = False
        coords, success = self.GetBoundingBoxTotales(d)#, 25, 470, 205, 125)
        coords = coords*(1.0/scale)

        (tx, ty, tw, th) = (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))
        totales_crop = img[ty:ty+th, tx:tx+tw]
        
        gray = cv2.cvtColor(totales_crop.copy(), cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray.copy(), 50, 200, 5)
        #ret, clean = cv2.threshold(gray.copy(), 0, 255, cv2.THRESH_OTSU)
        rects = self.ExtractRectangles(edges)
        (cleaned_rects,cleaned_rects_vcenter,cleaned_rects_hcenter, avg_w, avg_h) = self.CleanRectangles(rects)
        cleaned_rects,cleaned_rects_vcenter,cleaned_rects_hcenter = self.removeChildrenRects(cleaned_rects)

        ntotales=7
        ndigitos=3
        if (len(cleaned_rects_vcenter)>ntotales):
            totals_lbl,totals_mean=self.GetLabels(cleaned_rects_vcenter,ntotales)
        else:
            print(" >>Se esperan {} pero se encontraron {} totales".format(ntotales, len(cleaned_rects_vcenter)))
            return 0,0,-1,0
        
        if (len(cleaned_rects_vcenter)>ntotales):
            digit_lbl,digit_mean=self.GetLabels(cleaned_rects_hcenter,ndigitos)
            digit_lbl = [-1*lbl+2 for lbl in digit_lbl]  # convertir digit_lbl a las correctas potencias de 10
        else:
            print(" >> Se esperan {} pero se encontraron {} digitos".format(ndigitos, len(cleaned_rects_hcenter)))
            return 0,0,-2,0

        cleaned_rects, totals_lbl, digit_lbl = self.GetStandardRects(avg_w, avg_h, digit_mean, totals_mean)        

        return cleaned_rects, totals_lbl, digit_lbl, coords

    def GetLabels(self, data,nclust):
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

    def GetStandardRects(self, w,h, x_ps, y_ps):
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

    def removeChildrenRects(self, rects):
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

    def ExtractRectangles(self, image, type_flag = cv2.RETR_EXTERNAL):
        """
        Obtener rectangular que encierran a cada digito
        Adaptado de https://github.com/leaguilar/election_count_helper
        """
        # Find contours in the image
        ctrs, hier = cv2.findContours(image, type_flag, cv2.CHAIN_APPROX_SIMPLE)
        # Get only parent rectangles that contains each contour
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]
        return rects

    def CleanRectangles(self, rects, h_min=35, w_min=25, h_max = 75, w_max = 75):
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

    def GetBoundingBoxTotales(self, d):
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

        idx, words = self.GetKeywordIndex(d, keywords)
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
                coord.append([p1_x, p1_y, w, h])

            elif i == 1 :

                p1_y = t_y + t2_y_offset #int(params.t2_p1_h_offset * t_h / params.t2_h) + t_y
                p1_x = t_x + t2_x_offset #int(params.t2_p1_w_offset * t_w / params.t2_w) + t_x
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

    def GetKeywordIndex(self, d, keywords):
        """
        Obtener los indice en el diccionario d['text'] de una lista de frases 'keywords'
        """
        indices = []
        words = []
        for keyword in keywords:
            success, index, ratio, matching_word = self.FuzzyMatch(keyword, d['text'])
            if success:
                indices.append(index)
                words.append(matching_word)
        return indices, words

    def FuzzyMatch(self, word, matching_words):
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

    def OCR(self,img, whtlst = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"):   
        """
        Hacer OCR en la imagen.
        Adaptado de https://github.com/leaguilar/election_count_helper
        """
        whitelist = "-c tessedit_char_whitelist={}".format(whtlst)
        d = pytesseract.image_to_data(img, output_type=Output.DICT, config=whitelist)
        return d

    def FindResize(self, img, max_size):
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

        
        if ((aspect_ratio > self.ar_low) and (aspect_ratio < self.ar_high) and (area_ratio > 0.5)):
            print("aspect_ratio: {}, area_ratio: {}".format(aspect_ratio, area_ratio))
            img = perspective.four_point_transform(img, rects[0].reshape(4, 2) * scale)
            success = True

        #return cv2.resize(img, (dw, dh), interpolation = cv2.INTER_AREA), success 
        return img, success, scale

class LabelExtractor():

    def __init__(self):
        self.a=0

    def processLabels(self, df, acta):

        columns = ['PARTIDOA', 'PARTIDOB', 'VALIDOS', 'NULOS', 'BLANCOS', 'EMITIDOS', 'INVALIDOS']
        labels = []
        for field in columns:
            labels.append("{:03d}".format(df.loc[acta][field]))
        return labels


def main():
    extractor = DigitExtractor()
    lbl_extractor = LabelExtractor()

    actas_dir = './datasets/2davuelta/sim/'
    actas_filenames = os.listdir(actas_dir)[:10]

    labels_dir = './datasets/2davuelta/sim_actas_data.xlsx'
    columns = ['RECIBIDAS','PARTIDOA', 'PARTIDOB', 'VALIDOS', 'NULOS', 'BLANCOS', 'EMITIDOS', 'INVALIDOS']
    df = pd.read_excel(labels_dir, index_col = 0)
    df.columns = columns

    for i, file in enumerate(actas_filenames):
        print("Imagen {}: {}".format(i,file))
        image = cv2.imread(actas_dir + file)
        cleaned_rects, totals_lbl, digit_lbl = extractor.processarActa(image.copy())

        image_no = int(file.split('.')[0]) 
        labels = lbl_extractor.processLabels(df, image_no)
        print(labels)

        for i , rect in enumerate(cleaned_rects):
            x, y, w, h = rect
            image = cv2.rectangle(image,(x,  y),( x + w,  y + h),(0, 255, 0),3)
            image = cv2.putText(image, str(digit_lbl[i]), (x , y), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2,cv2.LINE_AA)
            image = cv2.putText(image, str(labels[totals_lbl[i]]), ( 200 , y), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),2,cv2.LINE_AA)

        cv2.imshow('fig',cv2.resize(image, None, fx=0.25, fy = 0.25))
        cv2.waitKey()
        cv2.destroyAllWindows()  
        

if __name__ == '__main__':
    main()

