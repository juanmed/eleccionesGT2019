# Verificación Automática de Resultados de Elecciones Generales Guatemala 2019

Este proyecto utiliza algoritmos de inteligencia artificial y visión por computadora para procesar imágenes de actas de votaciones y obtener los resultados de votos por partido y totales relevantes (votos válidos, nulos, etc).

El flujo es sencillo: se obtiene un acta, se "dibujan" rectangulos que contengan cada total de interés, se extrae el total en cada rectángulo, y se genera un reporte (en formato .json).

![alt tag](https://github.com/juanmed/eleccionesGT2019/blob/master/content/000011.jpg)
![alt tag](https://github.com/juanmed/eleccionesGT2019/blob/master/content/000102.jpg)
![alt tag](https://github.com/juanmed/eleccionesGT2019/blob/master/content/7.png)


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

El siguiente comando instalara (casi) todos los paquetes necesarios para ejecutar los scripts (torch, torchvision, numpy, etc).

```bash
python -m pip install -r requerimientos.txt
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
```

* Si hubiese algún problema con en la instalación de pytesseract, seguir la [información oficial](https://github.com/tesseract-ocr/tesseract/wiki).

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

g) Descarga los datasets de actas de votos de las Elecciones Generales de Guatemala 2019:

```bash
PENDIENTE!
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
