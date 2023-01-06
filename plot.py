import argparse

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def moving_average(x):
    w = 10
    x = np.array(x).flatten()
    return np.convolve(x, np.ones(w), 'valid') / w


def plot_moving_acc(loss, train_acc, test_acc):
    x = np.arange(total_epoch)
    plt.figure(1)
    plt.plot(x[:end], moving_average(loss), label='loss')
    plt.plot(x[:end], moving_average(train_acc), label='train acc')
    plt.plot(x[:end], moving_average(test_acc), label='test acc')

    plt.xlabel('epoch')
    plt.ylabel('loss and acc')
    plt.title('Smoothed plot of Mnist dataset')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)

    args = parser.parse_args()

    path_0 = 'C:\\Users\\82581\\Desktop\\DualSTG\\DPLog\\{}_ldp_0.csv'.format(args.dataset)
    path_01 = 'C:\\Users\\82581\\Desktop\\DualSTG\\DPLog\\{}_ldp_01.csv'.format(args.dataset)
    path_02 = 'C:\\Users\\82581\\Desktop\\DualSTG\\DPLog\\{}_ldp_02.csv'.format(args.dataset)
    path_03 = 'C:\\Users\\82581\\Desktop\\DualSTG\\DPLog\\{}_ldp_03.csv'.format(args.dataset)

    data_0 = pd.read_csv(path_0) 
    data_01 = pd.read_csv(path_01) 
    data_02 = pd.read_csv(path_02) 
    data_03 = pd.read_csv(path_03) 

    if args.dataset == "arcene" or args.dataset == "gisette":
        total_epoch = 40
    elif args.dataset == "basehock":
        total_epoch = 80
    elif args.dataset == "coil" or args.dataset == "isolet":
        total_epoch = 200
    elif args.dataset == "pcmac":
        total_epoch = 30
    elif args.dataset == "relathe":
        total_epoch = 60

    print('total epoch', total_epoch)
    end = total_epoch - 10 + 1

    x = np.arange(total_epoch)

    train_loss_0, test_acc_0 = data_0['train_loss'], data_0['test_acc']
    train_loss_01, test_acc_01 = data_01['train_loss'], data_01['test_acc']
    train_loss_02, test_acc_02 = data_02['train_loss'], data_02['test_acc']
    train_loss_03, test_acc_03 = data_03['train_loss'], data_03['test_acc']


    plt.figure(1)
    plt.plot(x[:end], moving_average(train_loss_0), label = 'zetac = 0.0')
    plt.plot(x[:end], moving_average(train_loss_01), label = 'zetac = 0.1')
    plt.plot(x[:end], moving_average(train_loss_02), label = 'zetac = 0.2')
    plt.plot(x[:end], moving_average(train_loss_03), label = 'zetac = 0.3')

    plt.xlabel('epoch')
    plt.ylabel('train loss')
    plt.title('{}'.format(args.dataset))
    plt.legend()
    plt.show()


    plt.figure(2)
    plt.plot(x[:end], moving_average(test_acc_0), label = 'zetac = 0.0')
    plt.plot(x[:end], moving_average(test_acc_01), label = 'zetac = 0.1')
    plt.plot(x[:end], moving_average(test_acc_02), label = 'zetac = 0.2')
    plt.plot(x[:end], moving_average(test_acc_03), label = 'zetac = 0.3')

    plt.xlabel('epoch')
    plt.ylabel('test acc')
    plt.title('{}'.format(args.dataset))
    plt.legend()
    plt.show()

