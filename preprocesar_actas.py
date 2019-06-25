import os
from PIL import Image
import numpy as np

actas_dir = './actas_original/'
save_dir = './actas_procesadas/'

svhn_dir = './svhn_dataset/test/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)



def center_crop(img, new_width=None, new_height=None):        

    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img

def main():
	"""
		Tomar todas las images de actas en actas_dir y recortarlas a 
		dimensiones estandar. Este paso es necesario ya que la red neuronal
		necesita que todas las images tengan las mismas dimensiones
	"""
	
	img_names = os.listdir(svhn_dir)
	img_names = [x for x in img_names if '.xml' not in x]
	w = []
	h = []

	for img_name in img_names:

		img = Image.open(svhn_dir + img_name)
		width, height = img.size

		w.append(width)
		h.append(height)

	print("Original Width avg {:.2f}  std_dev {:.2f} min {} max {}".format(np.mean(w), np.std(w), np.min(w), np.max(w)))
	print("Original Height avg {:.2f} std_dev {:.2f} min {} max {}".format(np.mean(height), np.std(h), np.min(h), np.max(h)))

	new_width = np.min(w)
	new_height = np.min(h)
	"""
	# center crop
	for img_name in img_names:

		img = Image.open(actas_dir + img_name)
		width, height = img.size

		# ver https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
		left = (width - new_width)/2
		top = (height - new_height)/2
		right = (width + new_width)/2
		bottom = (height + new_height)/2

		img = img.crop((left, top, right, bottom))
		img.save(save_dir + img_name)
	

	w = []
	h = []
	new_img_names = os.listdir(save_dir)

	for img_name in new_img_names:

		img = Image.open(save_dir + img_name)
		width, height = img.size

		w.append(width)
		h.append(height)
	
	print("\nNew Width avg {:.2f}  std_dev {:.2f} min {} max {}".format(np.mean(w), np.std(w), np.min(w), np.max(w)))
	print(" New Height avg {:.2f} std_dev {:.2f} min {} max {}".format(np.mean(height), np.std(w), np.min(h), np.max(h)))
	"""

if __name__ == '__main__':
	main()