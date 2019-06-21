import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET



import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from torchsummary import summary

from engine import train_one_epoch, evaluate
import utils
import transforms as T

model_save_path = "./detector_total_models/"
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)


classes = {'vam':21, 'tod':3, 'une':6, 'uni':7, 'con':13,
           'cre':14, 'fcn':12, 'win':16, 'ppt': 19, 'unid': 18,
           'eg': 10, 'urn':5, 'vic':15, 'phg':23, 'viv':11,
           'ava':22, 'lib':27, 'pan':1, 'mlp': 24, 'val':2, 'vali':40,
           'nul':41, 'bla':42, 'vale': 43, 'inv': 44, 'fue':17,
           'ucn': 9,  'pc':25, 'sem':26, 'pod':4, 'bie':8 }

num_classes = len(classes)+1

#hyperparametros
num_epochs = 10
learning_rate = 0.0005
momentum = 0.5
weight_decay = 0.0005
save_model = False
save_frequency = 2

batch_size_train = 2
batch_size_test = 5

img_width = 1634
img_height = 2182


def get_transform():
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    #transforms.append(T.Resize((img_height, img_width)))
    return T.Compose(transforms)

class ActasLoader(torch.utils.data.Dataset):

    def __init__(self, root, split='train', transform = None, target_transform = None):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split

        self.train_dir = os.listdir(self.root + 'train/')
        self.test_dir = os.listdir(self.root + 'test/')
        self.val_dir = os.listdir(self.root + 'val/')

        if (self.split == 'train'):
            #self.data = map(lambda x: (self.root + 'train/' + x) if '.xml' not in x, self.train_dir)
            self.data = [(self.root + 'train/' + x) for x in self.train_dir if '.xml' not in x]
        elif(self.split == 'test'):
            self.data = [(self.root + 'test/' + x) for x in self.test_dir if '.xml' not in x]
        elif(self.split == 'val'):
            self.data = [(self.root + 'val/' + x) for x in self.val_dir if '.xml' not in x]
        else:
            raise RuntimeError('{} is not a valid split. Use -test-, -train- or -val-. '.format(self.split))

    def __getitem__(self, index):
        """
        """
        img_path = self.data[index]
        img = Image.open(img_path).convert("L")
        print(">> Procesando: {}, {}".format(img_path, img.size))
        target_dict = self.get_labels(img_path.replace('.jpg','.xml'))

        if self.transform is not None:
            img, target = self.transform(img, target_dict)

        return img, target

    def __len__(self):
        """
        """
        return len(self.data)
           
    def get_labels(self, label_path):
        """
        """
        tree = ET.parse(label_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        target_dict = {}

        size = root.find('size')
        width = int(size[0].text)*1.0
        height = int(size[1].text)*1.0
        
        for member in root.findall('object'):
            #get class label
            lbl = (classes[member[0].text])
            labels.append(lbl)

            # get bounding box
            xmin =int(member[4][0].text)#/width
            ymin = int(member[4][1].text)#/height
            xmax = int(member[4][2].text)#/width
            ymax = int(member[4][3].text)#/height
            bbox = [xmin, ymin, xmax, ymax]
            bboxes.append(bbox)

        if (len(labels) == len(bboxes)):
            pass
        else: 
            raise RuntimeError('Number of labels {} and boxes {} do not correspond'.format(len(labels), len(bboxes)))

        target_dict['boxes'] = torch.as_tensor(bboxes, dtype=torch.float32)
        target_dict['labels'] = torch.as_tensor(labels, dtype=torch.int64)

        return target_dict

def main():

    # verificar si hay GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #torch.device('cpu')
    print(" >> Found {} device".format(device))

    # crear red
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model = model.to(device)
    #print(model)
    #summary(model, input_size=(1, img_width, img_height), batch_size=batch_size_train, device='cuda')

    # crear optimizador
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay )

    # programador de learning rate
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


    do_learn = True
    if(do_learn):
        # load datasets
        #trans = transforms.Compose([ transforms.Resize((img_height, img_width)), transforms.ToTensor()])
        trans = get_transform()
        train_dataset = ActasLoader(root = './actas_dataset/', split='train', transform = trans)
        test_dataset = ActasLoader(root = './actas_dataset/', split='test', transform = trans)
        
        for item in train_dataset:
            pass

        train_loader = torch.utils.data.DataLoader( train_dataset, batch_size = batch_size_train, shuffle = True, collate_fn=utils.collate_fn)
        test_loader = torch.utils.data.DataLoader( test_dataset, batch_size = batch_size_test, shuffle = True, collate_fn=utils.collate_fn)

        for epoch in range(num_epochs):

            train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=1)
            lr_scheduler.step()  # refrescar taza de aprendizaje
            evaluate(model, test_loader, device=device)

            if save_model and ((epoch % save_frequency) == 0) and epoch != 0 :
                torch.save(modelo.state_dict(), model_save_path+'mnist_nn_{:03}.pt'.format(epoch))
                print(">> Saving model: {}".format(model_save_path+'mnist_nn_{:03}.pt'.format(epoch)))
            else:
                # not moment to save
                pass

        #img, target = train_dataset[0]
        #for box, lbl in zip(target['boxes'], target['labels']):
        #    print(box,lbl)
    else:
        pass


    #plt.imshow(img)
    #plt.show()

    
    """
    class_id = []
    for key in classes.keys():
        class_id.append(classes[key])

    a = np.arange(30)
    print (a)
    for id in a:
        if id not in class_id:
            print(id)
        else:
            pass
    """
if __name__ == '__main__':
    main()