import os
import numpy as np
import argparse
import torch
import torch.optim as optim
from tqdm import *
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision
import matplotlib.pyplot as plt
from option import Options
from datasets import oneShotdataset
from logger import Logger
from torch.optim import lr_scheduler
import copy
import time
rootdir = os.getcwd()

args = Options().parse()

logger = Logger('./logs/'+args.tensorname)

#def __init__(self, dataroot = '/home/zitian/data/miniImagenet', type = 'train',ways=5,shots=1,test_num=1,epoch=100):
image_datasets = {}
image_datasets['train'] = oneShotdataset.miniImagenetOneshotDataset(type='train',ways=args.trainways,shots=args.shots,test_num=args.test_num,epoch=200)
image_datasets['val'] = oneShotdataset.miniImagenetOneshotDataset(type='val',ways=args.ways,shots=args.shots,test_num=args.test_num,epoch=20)
image_datasets['test'] = oneShotdataset.miniImagenetOneshotDataset(type='test',ways=args.ways,shots=args.shots,test_num=args.test_num,epoch=20)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=(x=='train'), num_workers=args.nthreads)
              for x in ['train', 'val','test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}

######################################################################
# Check The dataLoader
'''
epoch = 0
for i,(inputs,labels) in tqdm(enumerate(dataloaders['train'])):
    if i==1:
        info =  {
                'images': (inputs[0:10].cpu().numpy().transpose((0,2,3,1))*np.array([0.485, 0.456, 0.406])+np.array([0.229, 0.224, 0.225])).transpose((0,3,1,2))
                }
        for tag, images in info.items():
            logger.image_summary(tag, images, epoch+1)
    print(i,inputs.size(),labels.size())
'''


######################################################################
# Define the Embedding Network
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class EmbeddingNetwork(nn.Module):
    def __init__(self):
        super(EmbeddingNetwork, self).__init__()
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

        self.encoder = nn.Sequential( # 3*84*84
            conv_block(3, 64), # 64*42*42
            conv_block(64, 64), # 64*21*21
            conv_block(64, 64), # 64*10*10
            conv_block(64, 64), # 64*5*5
            Flatten() # 1600
        )
        print(self.encoder)

    def forward(self,inputs):

        """                 
        inputs: Batchsize*3*224*224
        outputs: Batchsize*100
        """
        outputs = self.encoder(inputs)
        
        return outputs

embeddingNetwork = EmbeddingNetwork().cuda()
#############################################
#Test the Embedding network
'''
inputs = torch.rand(32,3,224,224)
outputs = embeddingNetwork(Variable(inputs.cuda()))
print(outputs)
'''


#############################################
#Define the optimizer

optimizer_embedding = optim.Adam([
                {'params': embeddingNetwork.parameters()},
            ], lr=0.001)

embedding_lr_scheduler = lr_scheduler.StepLR(optimizer_embedding, step_size=10, gamma=0.5)


######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
#torch.save(embeddingNetwork.state_dict(),os.path.join(rootdir,'models/'+str(args.tensorname)+'heeh.t7'))


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def train_model(model, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000000000.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in [ 'train' ,'val' ,'test']:
            
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
            

            running_loss = 0.0 
            running_accuracy = 0

            # Iterate over data.
            for i,(supportInputs,supportLabels,testInputs,testLabels) in tqdm(enumerate(dataloaders[phase])):

                # wrap them in Variable

                supportInputs = Variable(supportInputs.squeeze(0).cuda())
                supportLabels = Variable(supportLabels.squeeze(0).cuda())

                testInputs = Variable(testInputs.squeeze(0).cuda())
                testLabels = Variable(testLabels.squeeze(0).cuda())

                ways = supportInputs.size(0)/args.shots

                #print(supportInputs.size(),supportLabels.size(),testInputs.size(),testLabels.size(),ways)

                #print(inputs.size())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                supportFeatures = model(supportInputs)
                testFeatures = model(testInputs)

                #print(supportFeatures.size())

                # caculate the center of each class

                center = supportFeatures.view(ways,args.shots,-1).mean(1)
                dists = euclidean_dist(testFeatures,center) # [ways*test_num,ways]

                #print(dists)

                log_p_y = F.log_softmax(-dists,dim=1).view(ways, args.test_num, -1) # [ways,test_num,ways]

                loss_val = -log_p_y.gather(2, testLabels.view(ways,args.test_num,1)).squeeze().view(-1).mean()
                
                _,y_hat = log_p_y.max(2)

                acc_val = torch.eq(y_hat, testLabels.view(ways,args.test_num)).float().mean()

                # statistics
                running_loss += loss_val.data[0]
                running_accuracy += acc_val.data[0]
                #print( loss_val.data[0], acc_val.data[0])

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss_val.backward()
                    optimizer.step()


            epoch_loss = running_loss / (dataset_sizes[phase]*1.0)
            epoch_accuracy = running_accuracy / (dataset_sizes[phase]*1.0)
            info = {
                phase+'loss': epoch_loss,
                phase+'accuracy': epoch_accuracy,
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch+1)
            #epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(
                phase, epoch_loss,epoch_accuracy))

            # deep copy the model
            if phase == 'test' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        
        print()
        if epoch%30 == 0 :
            torch.save(best_model_wts,os.path.join(rootdir,'models/'+str(args.tensorname)+'.t7'))
            print('save!')
        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Loss: {:4f}'.format(best_loss))
    

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


embeddingNetwork = train_model(embeddingNetwork, optimizer_embedding,
                         embedding_lr_scheduler, num_epochs=args.trainepoch)

# ... after training, save your model 
torch.save(embeddingNetwork.state_dict(),os.path.join(rootdir,'models/'+str(args.tensorname)+'.t7'))

# .. to load your previously training model:
#model.load_state_dict(torch.load('mytraining.pt'))

