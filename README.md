# Verificación Automática de Resultados de Elecciones Generales Guatemala 2019

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
c) Instala los paquetes necesarios

El siguiente comando instalara (casi) todos los paquetes necesarios para ejecutar los scripts (torch, torchvision, numpy, etc).

```bash
python -m pip install -r requerimientos.txt
```

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
