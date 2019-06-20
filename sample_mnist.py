import os

import torch
import torchvision
import matplotlib.pyplot as plt 

import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torchsummary import summary




# hyperparametros
num_epochs = 5
batch_size_train = 64
batch_size_test = 100
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
img_size = 28
save_model = True
save_frequency = 2

# 
mnist_dir = './mnist/'
model_save_path = "./digit_models/"

if not os.path.exists(mnist_dir):
    os.mkdir(mnist_dir)

if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

class Net(nn.Module):

    def __init__(self):
        """
            Inicializar capas de la red
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv2_drop = nn.Dropout2d()
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
            Procesar entrada x a traves de la red
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.relu(x)

        x = F.dropout(x, training= self.training)
        x = self.fc2(x)

        x = F.log_softmax(x)
        return x

def desplegar_muestras(data_loader):
    """
        Mostrar algunos numeros con su etiqueta
    """
    n = 6
    fig = plt.figure()
    muestras = enumerate(data_loader)
    batch_idx, (batch, etiquetas) = next(muestras)
    for i in range(n):
        ax = fig.add_subplot(2,3, i+1)
        ax.imshow(batch[i][0], cmap = 'gray', interpolation = 'none')
        ax.set_title(" Etiqueta: {}".format(etiquetas[i]))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def train(modelo, dispositivo, train_loader, epoca, optimizador):
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
        loss = F.nll_loss(output, etiquetas)
        loss.backward()

        optimizador.step()
        running_loss = running_loss + loss

        # obtener precision
        pred = output.data.max(1, keepdim=True)[1]
        correctos = correctos + pred.eq(etiquetas.data.view_as(pred)).sum()

    print("Train Epoch: {}, Avg Loss: {:.2f}, Accuracy: {}/{} ({:.2f}%)".format(epoca, running_loss.item()/len(train_loader.dataset), correctos, len(train_loader.dataset),  100.0*correctos.item()/len(train_loader.dataset)))
    
    return (running_loss.item()/len(train_loader.dataset), correctos.item()/len(train_loader.dataset))

def test(modelo, epoca, dispositivo, test_loader):
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
            loss = F.nll_loss(output, etiquetas)
            running_loss = running_loss + loss

            # obtener precision
            pred = output.data.max(1, keepdim=True)[1]
            correctos = correctos + pred.eq(etiquetas.data.view_as(pred)).sum()

        print("Test{}: Avg Loss: {:.2f}, Accuracy: {}/{} ({:.2f}%)".format(epoca, running_loss.item()/len(test_loader.dataset), correctos, len(test_loader.dataset),  100.0*correctos.item()/len(test_loader.dataset)))

    return (running_loss.item()/len(test_loader.dataset), correctos.item()/len(test_loader.dataset))

def main():
    """
        Cargar dataset, entrenar, y hacer unas pruebas
    """

    # verificar si hay GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(" >> Found {} device".format(device))


    # crear red y parametros
    modelo = Net().to(device)
    optimizador = optim.SGD(modelo.parameters(), lr=learning_rate, momentum = momentum)
    print(type(modelo))
    summary(modelo, input_size=(1, img_size, img_size), batch_size=batch_size_train, device='cuda')

    do_learn = True
    if do_learn:
        # training
        trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize((0.1307,),(0.3081,))
                                        ])
        
        # obtener MNIST dataset
        mnist_train = torchvision.datasets.MNIST(mnist_dir, train=True, download = True, transform = trans)
        mnist_test = torchvision.datasets.MNIST(mnist_dir, train=False, download = True, transform = trans)

        # Cargar datasets
        train_loader = torch.utils.data.DataLoader( mnist_train, batch_size = batch_size_train, shuffle = True)
        test_loader = torch.utils.data.DataLoader( mnist_test, batch_size = batch_size_test, shuffle = False)

        # mostrar algunos numeros
        #desplegar_muestras(test_loader)

        for epoch in range(num_epochs):

            # one train pass, one tests pass
            train_history = train(modelo, device, train_loader, epoch, optimizador)
            test_history = test(modelo, epoch, device, test_loader)

            if save_model and ((epoch % save_frequency) == 0) and epoch != 0 :
                torch.save(modelo.state_dict(), model_save_path+'mnist_nn_{:03}.pt'.format(epoch))
                print(">> Saving model: {}".format(model_save_path+'mnist_nn_{:03}.pt'.format(epoch)))
            else:
                # not moment to save
                pass
    else:
        #evaluation
        pass





if __name__ == '__main__':
    main()

