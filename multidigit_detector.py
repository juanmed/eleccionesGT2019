import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import cv2

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms
from torchsummary import summary

from engine import train_one_epoch, evaluate
import utils
import transforms as T

model_save_path = "./multidigit_models/"
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)




classes = {'10':0, '1':1, '2':2, '3':3, '4':4,
           '5':5, '6':6, '7':7, '8': 8, '9': 9}

classes_inv = {'0':'0', '1':'1', '2':'2', '3':'3', '4':'4',
               '5':'5', '6':'6', '7':'7', '8': '8', '9': '9'}

num_classes = len(classes)
print(">> Total classes: {}".format(num_classes))

#hyperparametros
num_epochs = 11
learning_rate = 0.005
momentum = 0.5
weight_decay = 0.005
save_model = True
save_frequency = 5
step_size = 80
gamma = 0.5

batch_size_train = 4
batch_size_test = 8
batch_size_val = 8

img_width = 1634
img_height = 2182

def get_mobilenet_model(num_classes):
    """
        Seguir ejemplo en https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py
    """

    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7,sampling_ratio=2)

    # stats for test images
    #Original Width avg 172.58  std_dev 122.58 min 31 max 1083
    #Original Height avg 105.00 std_dev 52.75 min 13 max 516

    model = FasterRCNN(backbone, num_classes=num_classes, min_size=100, max_size=300, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    return model

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
        self.target_transform = target_transform
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

        self.data = self.data[0:len(self.data)//128]

    def __getitem__(self, index):
        """

            Referencia: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        """
        img_path = self.data[index]
        img = Image.open(img_path).convert("RGB")
        #img = cv2.imread(img_path)
        #print(img.shape)
        #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        #img = Image.fromarray(img, mode='RGB')
        target_dict = self.get_labels(img_path.replace('.png','.xml'))
        target_dict['image_id'] = torch.tensor([index])

        boxes = target_dict['boxes']
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target_dict["area"] = area

        num_objs = len(boxes)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target_dict["iscrowd"] = iscrowd

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target_dict = self.target_transform(target_dict)

        return img, target_dict

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
            xmin =int(member[1][0].text)#/width
            ymin = int(member[1][1].text)#/height
            xmax = int(member[1][2].text)#/width
            ymax = int(member[1][3].text)#/height
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #
    print(" >> Found {} device".format(device))

    # crear red
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)    
    #nodel = FasterRCNN(pretrained = True)
    model = get_mobilenet_model(num_classes)
    #in_features = model.roi_heads.box_predictor.cls_score.in_features
    #model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model = model.to(device)
    print(model)
    #summary(model, input_size=(1, img_width, img_height), batch_size=batch_size_train, device='cuda')

    # crear optimizador
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay )

    # programador de learning rate
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    resume = True
    if(resume):
        checkpoint = torch.load(model_save_path+"multidigit_nn_010.pt", map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler']) 


    do_learn = True
    if(do_learn):
        # load datasets
        trans = transforms.Compose([transforms.ToTensor()]) #transforms.Resize((img_height, img_width)),
        #trans = get_transform()
        train_dataset = ActasLoader(root = './svhn_dataset/', split='train', transform = trans)
        test_dataset = ActasLoader(root = './svhn_dataset/', split='test', transform = trans)

        train_loader = torch.utils.data.DataLoader( train_dataset, batch_size = batch_size_train, shuffle = True, collate_fn=utils.collate_fn, num_workers=4)
        test_loader = torch.utils.data.DataLoader( test_dataset, batch_size = batch_size_test, shuffle = True, collate_fn=utils.collate_fn, num_workers=4)

        for epoch in range(num_epochs):

            train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=1)
            lr_scheduler.step()  # refrescar taza de aprendizaje
            evaluate(model, test_loader, device=device)

            if save_model and ((epoch % save_frequency) == 0) and epoch != 0 :
                
                utils.save_on_master({'model': model.state_dict(),
                                      'optimizer': optimizer.state_dict(),
                                      'lr_scheduler': lr_scheduler.state_dict()},
                                      #'args':
                                      os.path.join(model_save_path, 'multidigit_nn_{:03}.pt'.format(epoch)))

                print(">> Saving model: {}".format(model_save_path+'multidigit_nn_{:03}.pt'.format(epoch)))
            else:
                # not moment to save
                pass

    else:
        # load eval data
        trans = get_transform()
        val_dataset = ActasLoader(root = './svhn_dataset/', split='test', transform = trans)        
        val_loader = torch.utils.data.DataLoader( val_dataset, batch_size = batch_size_val, shuffle = False, collate_fn=utils.collate_fn, num_workers=4)
        
        #checkpoint = torch.load(model_save_path+"multidigit_nn_010.pt", map_location='cpu')
        #model.load_state_dict(checkpoint['model'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #lr_scheduler.load_state_dict(checkpoint['lr_scheduler']) 
        model.eval()

        # evaluate batch
        #coco_eval, output = eval(model, val_loader, device)
        #print(coco_eval)
        #print(output)

        # evaluate single images, draw boxes and save
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)


        val_dir = './svhn_dataset/test/'
        img_dirs = os.listdir(val_dir)
        img_dirs = [(val_dir + x) for x in img_dirs if '.xml' not in x]

        for img_dir in img_dirs[20:30]:

            image = cv2.imread(img_dir)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image2 = Image.fromarray(image, mode='RGB')
            image2 = transforms.ToTensor()(image2)

            image2 = [image2.to(device)]
            output = model(image2)[0]
            print(output)

            #image = image.to(torch.device('cpu')).permute(1, 2, 0).numpy()*255 

            for i , (box, lbl, conf) in enumerate(zip(output['boxes'], output['labels'], output['scores'])):

                if(conf > 0.4):
                    # get colors for bounding boxes
                    col1 = np.random.randint(0,256)
                    col2 = np.random.randint(0,256)
                    col3 = np.random.randint(0,256)
                    
                    #print(box)
                    xmin = int(box[0].item())
                    ymin = int(box[1].item())
                    xmax = int(box[2].item())
                    ymax = int(box[3].item())
                    #print(xmin, ymin, xmax, ymax)
                    #cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(col1,col2,col3),1)
                    lbl = classes_inv[str(lbl.item())]
                    #print(lbl, '{:.2f}'.format(conf))
                    cv2.putText(image, lbl, (xmin + 5, ymin + 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (col1, col2, col3), 1)
                    #cv2.putText(image, '{:.2f}'.format(conf), (xmin + 20, ymin +10),cv2.FONT_HERSHEY_SIMPLEX, 0.25, (col1, col2, col3), 1)

            ax.imshow(image)
            fig.savefig("{}".format(img_dir.split('/')[-1]))




if __name__ == '__main__':
    main()