import cv2
import numpy as np 
import json

out_json_save_path = "./datasets/gtmnist/"
json_name = "gtmnist.json"


with open(out_json_save_path + json_name, 'r') as json_file:
    a = json.load(json_file)

for key in a.keys():
    image = np.array(a[key]['image'])
    label = a[key]['label']
    print("Label: {}".format(label))
    cv2.imshow('fig',cv2.resize(image, None, fx=1, fy = 1))
    cv2.waitKey()
    cv2.destroyAllWindows()     
