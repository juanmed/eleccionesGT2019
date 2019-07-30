# Verificación Automática de Resultados de Elecciones Generales Guatemala 2019

Este proyecto utiliza algoritmos de inteligencia artificial y visión por computadora para procesar imágenes de actas de votaciones y obtener los resultados de votos por partido y totales relevantes (votos válidos, nulos, etc).

El flujo es sencillo: se obtiene un acta, se "dibujan" rectangulos que contengan cada dato de interés, se extrae el total en cada rectángulo, y se genera un reporte (en formato .json).

![alt tag](https://github.com/juanmed/eleccionesGT2019/blob/master/content/result_small_303.jpg)
![alt tag](https://github.com/juanmed/eleccionesGT2019/blob/master/content/303_nocuadro.png)
![alt tag](https://github.com/juanmed/eleccionesGT2019/blob/master/content/000102.jpg)

# Instalacion

Para utilizar este repositorio, se necesitan algunas herramientas previo a ejecutar los scripts. Los scripts fueron creados para ser interpretados en Python 3.x en Ubuntu16.04.


a) Descargar este repositorio

```bash
git clone https://github.com/juanmed/eleccionesGT2019
cd eleccionesGT2019
```

b) Crea un ambiente virtual (recomendado pero opcional)

Descarga e instala virtualenv si aun no lo tienes:

```bash
pip3 install virtualenv
```
Crea y carga un ambiente virtual que utilice Python3.x. Por ejemplo, en este caso, python 3.5 y con nombre '.tse'

```bash
virtualenv -p /usr/lib/python3.5 .tse
source .tse/bin/activate
```
c) Instala los paquetes y herramientas necesarias

Los siguientes comandos instalara (casi) todos los paquetes necesarios para ejecutar los scripts (torch, torchvision, numpy, etc).

```bash
python -m pip install -r requerimientos.txt
```

Es necesario installar tesseract (usando el código fuente) y pytesseract. En [este link](https://lengerrong.blogspot.com/2017/03/how-to-build-latest-tesseract-leptonica.html) se puede encontrar el procedimiento. Tesseract necesita algunos [archivos de extensión .traineddata](https://github.com/tesseract-ocr/tessdata) para trabajar con distintos lenguajes: estos se deberían almacenar en ```/usr/local/share/tessdata```. Cuando yo seguí este procedimiento, dicho folder estaba vacío y mi código no funcionaba. La solución es descargarlos según el lenguaje a utilizar y guardarlos en la carpeta indicada. Yo descargué y guarde en esa carpeta el archivo ```eng.traineddata``` que corresponde al idioma inglés.

* Si hubiese algún problema con la instalación de pytesseract, seguir la [información oficial](https://github.com/tesseract-ocr/tesseract/wiki).

* Si utilizas CUDA10, por favor ver las instrucciones de instalación de Pytorch y torchvision más abajo

e) Descargar e instala pycocotools

```bash
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
```

f) Descarga el repositorio de Torchvision

En dicho repositorio se encuentran algunos scripts que este repositorio necesita.

```bash
git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.3.0

cp references/detection/utils.py ../
cp references/detection/transforms.py ../
cp references/detection/coco_eval.py ../
cp references/detection/engine.py ../
cp references/detection/coco_utils.py ../
cd ../
```
# Ejecutar para analizar actas de la 2da vuelta

a) Crear una carpeta con el nombre 'weights'  y dentro de esta guardar el archivo de pesos para la red neuronal que se encuentra en este link:

https://drive.google.com/open?id=181ygFISww_ZWK7Y1pH3awqjebhd9vrJL 

b) Descarga fotos de actas simuladas de la 2da vuelta y guardalas en el directorio 'datasets/2davuelta/sim/'

https://drive.google.com/open?id=1kdtIRkpx3XH-p-0PKmP3V56w5579uBRx

c) Abre una terminal y:

```bash

cd <eleccionesGT2019 directory>
python preprocess.py
```

El script de analisis se ejecutara, mostrará la imagen de cada acta con la información reconocida en cada una de ellas. Pulsa cualquier tecla para ver las siguiente acta. 


## Instalar Pytorch y Torchvision con CUDA10

Primero desinstalar pytorch y torchvision:

```python
pip uninstall torch
pip uninstall torchvision
```

Instalar de nuevo utilizando

```python
pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl
```

Verificar utilizando

```bash
>>> import torch
>>> print(torch.__version__)
1.1.0
>>> print(torch.version.cuda)
10.0.130
>>> 
```


## Referencias

https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb#scrollTo=UYDb7PBw55b-

https://github.com/pytorch/vision/blob/v0.3.0/references/detection/train.py

https://github.com/pytorch/vision/tree/v0.3.0/references/detection

https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py


SVHN Dataset
http://ufldl.stanford.edu/housenumbers/

SVHN Annotation in PASCAL-VOC format
https://github.com/penny4860/svhn-voc-annotation-format
