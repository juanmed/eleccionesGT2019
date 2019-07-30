import os

import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn

import torchvision
from torchvision import transforms
import torch.optim as optim
from torchsummary import summary
import torch.nn.functional as F 

from models.lenet import LeNet 
from models.mnist_resnet import MNISTResNet
import utils


model_save_path = "./weights/"
weight_file = "mnist_svhn_resnet_rand_"
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

#hyperparametros
num_epochs = 200
learning_rate = 0.005
momentum = 0.5
weight_decay = 0.005
save_model = True
save_frequency = 30
step_size = 150
gamma = 0.5
img_size = 32


batch_size_train = 16
batch_size_test = 16
batch_size_val = 1

img_width = 1634
img_height = 2182


class MNIST_SVHN_Dataset(torch.utils.data.Dataset):

    def __init__(self, root = './datasets/', split = "train", mnist_transform = None, svhn_transform = None):
        """
        """
        self.mnist_transform = mnist_transform
        self.svhn_transform = svhn_transform

        if (split == "train"):
            # obtener MNIST dataset
            self.mnist_data = torchvision.datasets.MNIST(root, train=True, download = True, transform = None)
            self.svhn_data = torchvision.datasets.SVHN(root, split='train', download = True, transform = None)

        elif(split == "test"):
            self.mnist_data = torchvision.datasets.MNIST(root, train=False, download = True, transform = None)
            self.svhn_data = torchvision.datasets.SVHN(root, split='test', download = True, transform = None)

        elif(split == "val"):
            raise RunTimeError('Validation dataset still unavailable')
        else:
            raise RuntimeError('{} is not a valid split. Use -test-, -train- or -val-. '.format(self.split))

    def __getitem__(self,index):
        
        if(index < len(self.mnist_data)):
            img, target  = self.mnist_data[index]
            if self.mnist_transform is not None:
                img = self.mnist_transform(img)
        else:
            index = index - len(self.mnist_data)
            img, target = self.svhn_data[index]
            if self.svhn_transform is not None:
                img = self.svhn_transform(img)
        return img, target

    def __len__(self):
        return len(self.mnist_data) #+ len(self.svhn_data)

def train(modelo, dispositivo, train_loader, epoca, optimizador, criterion):
    """
    """
    modelo.train()
    loss = 0.0
    running_loss = 0.0
    correctos = 0.0
    for batch_idx, (data, etiquetas) in enumerate(train_loader):

        # mover el batch al dispositivo disponible
        data = data.to(dispositivo)
        etiquetas = etiquetas.to(dispositivo) 

        # entrenar
        optimizador.zero_grad()

        output = modelo(data)       # obtener predicciones
        loss = criterion(output, etiquetas)
        loss.backward()

        optimizador.step()
        running_loss = running_loss + loss

        # obtener precision
        pred = output.data.max(1, keepdim=True)[1]
        correctos = correctos + pred.eq(etiquetas.data.view_as(pred)).sum()

    print("Train Epoch: {}, Avg Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(epoca, running_loss.item()/len(train_loader.dataset), correctos, len(train_loader.dataset),  100.0*correctos.item()/len(train_loader.dataset)))
    
    return (running_loss.item()/len(train_loader.dataset), correctos.item()/len(train_loader.dataset))

def test(modelo, epoca, dispositivo, test_loader, criterion):
    """
    """
    modelo.eval()
    running_loss = 0.0
    correctos = 0.0

    with torch.no_grad():

        for batch_idx, (data, etiquetas) in enumerate(test_loader):

            data= data.to(dispositivo)
            etiquetas = etiquetas.to(dispositivo)

            output = modelo(data)
            loss = criterion(output, etiquetas)

            running_loss = running_loss + loss

            # obtener precision
            pred = output.data.max(1, keepdim=True)[1]
            correctos = correctos + pred.eq(etiquetas.data.view_as(pred)).sum()

        print("Test Epoch : {}: Avg Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(epoca, running_loss.item()/len(test_loader.dataset), correctos, len(test_loader.dataset),  100.0*correctos.item()/len(test_loader.dataset)))

    return (running_loss.item()/len(test_loader.dataset), correctos.item()/len(test_loader.dataset))

def show_samples(dataset):
    """
    """
    svhn_img, starget = dataset[len(dataset)-1]
    mnist_img, mtarget = dataset[0]

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    ax1.imshow(np.array(svhn_img))
    ax1.set_title(str(starget))
    ax2.imshow(np.array(mnist_img))
    ax2.set_title(str(mtarget))
    plt.show()

def main():
    """
        Cargar dataset, entrenar, y hacer unas pruebas
    """

    # verificar si hay GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(" >> Found {} device".format(device))


    # crear red y parametros
    #model = LeNet().to(device)
    model = MNISTResNet().to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum = momentum)
    summary(model, input_size=(1, img_size, img_size), batch_size=batch_size_train, device='cuda')

    # programador de learning rate
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # load previously trained model
    resume = False
    if(resume):
        checkpoint = torch.load(model_save_path+weight_file+"200.pt", map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        print("Loaded {}".format(model_save_path+weight_file+"002.pt")) 



    do_learn = True
    if do_learn:
        # training
        mnist_transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomAffine(degrees = 20, translate=(0.2,0.2)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,)),
            ##transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
            
        svhn_transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))
        ])
        
        # obtener MNIST dataset
        msd_train = MNIST_SVHN_Dataset(split = "train", mnist_transform = mnist_transform, svhn_transform = svhn_transform)
        msd_test = MNIST_SVHN_Dataset(split = "test", mnist_transform = mnist_transform, svhn_transform = svhn_transform)

        # Cargar datasets
        train_loader = torch.utils.data.DataLoader( msd_train, batch_size = batch_size_train, shuffle = True)
        test_loader = torch.utils.data.DataLoader( msd_test, batch_size = batch_size_test, shuffle = False)

        # loss function
        #loss = F.nll_loss(output, etiquetas)
        loss = nn.CrossEntropyLoss()
        
        # train
        train_history = []
        test_history = []
        for epoch in range(num_epochs):

            # one train pass, one tests pass
            train_params = train(model, device, train_loader, epoch, optimizer, loss)
            lr_scheduler.step()  # refrescar taza de aprendizaje
            test_params = test(model, epoch, device, test_loader, loss)
            train_history.append(train_params)
            test_history.append(test_params)

            if save_model and ((epoch % save_frequency) == 0) and epoch != 0 :

                utils.save_on_master({'model': model.state_dict(),
                                      'optimizer': optimizer.state_dict(),
                                      'lr_scheduler': lr_scheduler.state_dict()},
                                      #'args':
                                      os.path.join(model_save_path, weight_file+'{:03}.pt'.format(epoch)))

                print(">> Saving model: {}".format(model_save_path+weight_file+'{:03}.pt'.format(epoch)))
            

            else:
                # not moment to save
                pass

        # draw train history
        fig = plt.figure(figsize=(20,10))
        fig.suptitle("MNIST training results")
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)

        ax1.plot(list(map(lambda x: x[0], train_history)), color = 'b', label = 'train')   # train loss
        ax1.plot(list(map(lambda x: x[0], test_history)), color = 'r', label = 'test')    # test loss
        ax2.plot(list(map(lambda x: x[1], train_history)), color = 'b', label = 'train')   # train accuracy
        ax2.plot(list(map(lambda x: x[1], test_history)), color = 'r', label = 'test')   # train accuracy

        ax1.set_title("Loss")
        ax1.legend(loc='upper right')
        ax2.set_title("Accuracy")
        ax2.legend(loc='lower right')

        fig.savefig("train_history", dpi=200)
        #plt.show()


    else:
        #evaluation

        # training
        mnist_transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomAffine(degrees = 20, translate=(0.2,0.2)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,)),
            ##transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
            
        svhn_transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))
        ])        
        msd_test = MNIST_SVHN_Dataset(split = "test", mnist_transform = mnist_transform, svhn_transform = svhn_transform)
        test_loader = torch.utils.data.DataLoader( msd_test, batch_size = batch_size_test, shuffle = True)
        loss = nn.CrossEntropyLoss()
        test_params = test(model, "NaN", device, test_loader, loss)
        pass

if __name__ == '__main__': 
    #summary(model, input_size=(1, img_size, img_size), batch_size=batch_size_train, device='cpu')
    main()
