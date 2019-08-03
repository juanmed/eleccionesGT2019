import cv2
import numpy as np 
import json

out_json_save_path = "./datasets/gtmnist/"
json_name = "gtmnist.json"


with open(out_json_save_path + json_name, 'r') as json_file:
    a = json.load(json_file)

print("Hay {} imagenes".format(len(a.keys())))
count = [0]*10
for key in a.keys():
    image = np.array(a[key]['image'])
    label = a[key]['label']
    count[int(label)] = count[int(label)] + 1
    print("Label: {}, Key: {}".format(label, key))
    cv2.imshow('fig',cv2.resize(image, None, fx=1, fy = 1))
    cv2.waitKey()
    cv2.destroyAllWindows()     

print(count)
total_imgs = np.sum(count)
for i, number in enumerate(count):
    print("Digit {}: {} images {:.2f}".format(i, number, 100*number/total_imgs))
