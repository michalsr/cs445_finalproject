import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
# import sklearn.metrics as sm
# import pandas as pd
# import sklearn.metrics as sm
import random
import numpy as np
from tqdm import tqdm
import json
from datasets.create_dataset import *
from models.resnet import dct_resnet
import hydra
from omegaconf import DictConfig, OmegaConf

def accuracy(output,target):
    """Calculates model accuracy

    Arguments:
        mdl {nn.model} -- nn model
        X {torch.Tensor} -- input data
        Y {torch.Tensor} -- labels/target values

    Returns:
        [torch.Tensor] -- accuracy
    """
    _, preds = torch.max(output, 1)
    n = preds.size(0)  # index 0 for extracting the # of elements
    #print(preds, 'preds')
    #print(target.data,'target')
    train_acc = torch.sum(preds == target.data)
    return train_acc.item()/n

def get_optimizer(config,model):
    if config.optimizer == 'sgd':
        optimizer_model = torch.optim.SGD(model.parameters(), config.lr,
                                          momentum=config.momentum, weight_decay=config.weight_decay)
    return optimizer_model


def test(model, test_loader, device, writer, config, epoch,test=False):
    if test:
        eval_type = 'test'
    else:
        eval_type = 'val'

    model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inp, targets) in tqdm(enumerate(test_loader)):
            inp, targets = inp.to(device), targets.to(device)
            outputs,c = model(inp)
            print(outputs.size())
            test_loss += F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    writer.add_scalar(eval_type+ ' loss', test_loss, epoch)
    writer.add_scalar( eval_type+ ' accuracy', accuracy, epoch)

    tqdm.write('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(eval_type,
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    fname = os.path.join(config.save_dir.format(**config), eval_type+'_results/' + 'epoch_' + str(epoch) + '.json')
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'w') as f:
        json.dump({eval_type+'_loss': test_loss, eval_type+'_accuracy': accuracy}, f)
        tqdm.write(f'Saved {eval_type} results to {fname}')
    return accuracy
def configure_lr(epoch,lr,optimizer):
    new_lr = lr*0.1**(epoch%50)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return optimizer
def track_channels(channel_output):
    array = np.zeros((192))

def train(train_loader,model,optimizer,epoch,device,writer,config):
    print('\nEpoch: %d' % epoch)
    print(len(train_loader))
    train_loss = 0
    train_accuracy = []
    freq_array = torch.zeros(192,device=device)

    for batch_idx, (inp, targets) in tqdm(enumerate(train_loader)):
        model.train()
        #dct_y, dct_cb, dct_cr, targets = dct_y.to(device), dct_cb.to(device), dct_cr.to(device), targets.to(device)
        inp, targets= inp.to(device), targets.to(device)
        outputs,channels = model(inp)

        #print(channels)

        cross_entropy = nn.CrossEntropyLoss(reduction='none')
        cost = cross_entropy(outputs, targets)
        prec_train = accuracy(outputs, targets)

        freq_array+=channels.sum(axis=0)[0]
        cost += config.reg*channels.sum(axis=1)[0]
        cost = torch.sum(cost,axis=0)/64
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        train_loss += cost.item()
        train_accuracy.append(prec_train)
        #np.save('frequencies.npy', freq_array.cpu().detach().numpy())
        step_adjust = int(len(train_loader.dataset) / 100) * epoch

        if (batch_idx + 1) % 10 == 0:
            tqdm.write('Epoch: [%d/%d]\t'
                       'Iters: [%d/%d]\t'
                       'Loss: %.4f\t'

                       'Prec@1 %.4f\t'
                       % (
                           epoch, config.epochs, batch_idx + 1, len(train_loader.dataset) / 64,
                           (train_loss / (batch_idx + 1)),
                           np.mean(train_accuracy)))
        if (batch_idx+1) %500 ==0:
            writer.add_scalar('Train Loss', train_loss / (batch_idx + 1), batch_idx+step_adjust)
            writer.add_scalar('Train Accuracy', np.mean(train_accuracy), batch_idx+step_adjust)


    model_checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),'epoch':epoch}
    model_save_dir = config.save_dir.format(**config) + '/model/'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    torch.save(model_checkpoint, model_save_dir + 'model'+'.pt')
    freq_array = torch.div(freq_array,len(train_loader.dataset))
    return freq_array



def get_datasets():
    train_dataset = ImagenetteTrain('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = ImagenetteTrain('val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_data = ImagenetteTest()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader
def adjust_learning_rate(optimizer, epochs, config):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

@hydra.main(config_name='conf/conf.yaml')
def main(config):
    use_cuda = True
    model = dct_resnet()
    device = torch.device("cuda" if use_cuda else "cpu")
    log_dir = config.save_dir.format(**config)
    writer = SummaryWriter(log_dir=log_dir)
    train_loader, val_loader, test_loader = get_datasets()

    #optim = get_optimizer(config,model)
    optim = torch.optim.SGD(model.parameters(),config.lr)
    model.cuda()
    start_epoch = 0
    model_path = config.save_dir.format(**config)+'/model/'+'model.pt'
    #model_path = '/home/michal5/cs445/outputs/experiment_namee=pretrained/opt=sgd,wd=4e-05,reg=0.1,pretrained=True,lr=0.1,momentum=0.9/model/model.pt'
    model,start_epoch,optim = load_model(model_path,model,optim)
    model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'max')
    #initial = test(model,test_loader,device,writer,config,1,True)
    epoch_reached = 0
    overall_freq_array = torch.zeros(192,device=device)

    if config.train:
        for epoch in range(start_epoch,config.epochs):
            optim = configure_lr(epoch,config.lr,optim)
            frequencies = train(train_loader,model,optim,epoch,device,writer,config)
            overall_freq_array+=frequencies
            overall_freq_array = torch.div(overall_freq_array,epoch+1)
            np.save('frequencies_full.npy',overall_freq_array.cpu().detach().numpy())
            if epoch%5 == 0:
                val_acc = test(model, test_loader, device, writer, config, epoch, test=False)
                model_scheduler.step(val_acc)

    test_acc = test(model, test_loader, device, writer, config, epoch_reached, test=True)
    fname = os.path.join(config.save_dir.format(**config), 'best_accuracy' + '.json')
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'w') as f:
        json.dump({'test_accuracy': test_acc}, f)
        tqdm.write(f'Saved accuracy results to {fname}')
    print('test accuracy:', test_acc)

def load_model(model_path,model,optim):
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['state_dict'])
    optim.load_state_dict(ckpt['optimizer'])
    epoch = ckpt['epoch']

    return model, epoch,optim


if __name__ == '__main__':
    main()

