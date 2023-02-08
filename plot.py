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


arcene_zeta = np.array([0.0, 0.1, 0.3, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0])
basehock_zeta = np.array([0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0])
coil_zeta = np.array([0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0])
gisette_zeta = np.array([0.0, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
isolet_zeta = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0])
pcmac_zeta = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0])
relathe_zeta = np.array([0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)

    args = parser.parse_args()

    if args.dataset == 'arcene':
        zeta_array = arcene_zeta
    elif args.dataset == 'basehock':
        zeta_array = basehock_zeta
    elif args.dataset == 'coil':
        zeta_array = coil_zeta
    elif args.dataset == 'gisette':
        zeta_array = gisette_zeta
    elif args.dataset == 'isolet':
        zeta_array = isolet_zeta
    elif args.dataset == 'pcmac':
        zeta_array = pcmac_zeta
    elif args.dataset == 'relathe':
        zeta_array = relathe_zeta

    path_0 = 'C:\\Users\\dell\\Desktop\\DualSTG\\LDPLog\\{}_ldp_{}.csv'.format(args.dataset, zeta_array[0])
    path_1 = 'C:\\Users\\dell\\Desktop\\DualSTG\\LDPLog\\{}_ldp_{}.csv'.format(args.dataset, zeta_array[1])
    path_2 = 'C:\\Users\\dell\\Desktop\\DualSTG\\LDPLog\\{}_ldp_{}.csv'.format(args.dataset, zeta_array[2])
    path_3 = 'C:\\Users\\dell\\Desktop\\DualSTG\\LDPLog\\{}_ldp_{}.csv'.format(args.dataset, zeta_array[3])
    path_4 = 'C:\\Users\\dell\\Desktop\\DualSTG\\LDPLog\\{}_ldp_{}.csv'.format(args.dataset, zeta_array[4])
    path_5 = 'C:\\Users\\dell\\Desktop\\DualSTG\\LDPLog\\{}_ldp_{}.csv'.format(args.dataset, zeta_array[5])
    path_6 = 'C:\\Users\\dell\\Desktop\\DualSTG\\LDPLog\\{}_ldp_{}.csv'.format(args.dataset, zeta_array[6])
    path_7 = 'C:\\Users\\dell\\Desktop\\DualSTG\\LDPLog\\{}_ldp_{}.csv'.format(args.dataset, zeta_array[7])
    path_8 = 'C:\\Users\\dell\\Desktop\\DualSTG\\LDPLog\\{}_ldp_{}.csv'.format(args.dataset, zeta_array[8])
    path_9 = 'C:\\Users\\dell\\Desktop\\DualSTG\\LDPLog\\{}_ldp_{}.csv'.format(args.dataset, zeta_array[9])

    if args.dataset == 'coil':
        path_10 = 'C:\\Users\\dell\\Desktop\\DualSTG\\LDPLog\\{}_ldp_{}.csv'.format(args.dataset, zeta_array[10])
        data_10 = pd.read_csv(path_10) 

    data_0 = pd.read_csv(path_0) 
    data_1 = pd.read_csv(path_1) 
    data_2 = pd.read_csv(path_2) 
    data_3 = pd.read_csv(path_3) 
    data_4 = pd.read_csv(path_4) 
    data_5 = pd.read_csv(path_5) 
    data_6 = pd.read_csv(path_6) 
    data_7 = pd.read_csv(path_7) 
    data_8 = pd.read_csv(path_8) 
    data_9 = pd.read_csv(path_9) 
 

    # if args.dataset == "arcene" or args.dataset == "gisette":
    #     total_epoch = 40
    # elif args.dataset == "basehock":
    #     total_epoch = 80
    # elif args.dataset == "coil" or args.dataset == "isolet":
    #     total_epoch = 200
    # elif args.dataset == "pcmac":
    #     total_epoch = 30
    # elif args.dataset == "relathe":
    #     total_epoch = 60

    total_epoch = 500
    print('total epoch', total_epoch)
    end = total_epoch - 10 + 1

    x = np.arange(total_epoch)

    train_loss_0, test_acc_0 = data_0['train_loss'], data_0['test_acc']
    train_loss_1, test_acc_1 = data_1['train_loss'], data_1['test_acc']
    train_loss_2, test_acc_2 = data_2['train_loss'], data_2['test_acc']
    train_loss_3, test_acc_3 = data_3['train_loss'], data_3['test_acc']
    train_loss_4, test_acc_4 = data_4['train_loss'], data_4['test_acc']
    train_loss_5, test_acc_5 = data_5['train_loss'], data_5['test_acc']
    train_loss_6, test_acc_6 = data_6['train_loss'], data_6['test_acc']
    train_loss_7, test_acc_7 = data_7['train_loss'], data_7['test_acc']
    train_loss_8, test_acc_8 = data_8['train_loss'], data_8['test_acc']
    train_loss_9, test_acc_9 = data_9['train_loss'], data_9['test_acc']

    plt.figure(1)
    plt.plot(x[:end], moving_average(train_loss_0), label = 'zetac = {}'.format(zeta_array[0]))
    plt.plot(x[:end], moving_average(train_loss_1), label = 'zetac = {}'.format(zeta_array[1]))
    plt.plot(x[:end], moving_average(train_loss_2), label = 'zetac = {}'.format(zeta_array[2]))
    plt.plot(x[:end], moving_average(train_loss_3), label = 'zetac = {}'.format(zeta_array[3]))
    plt.plot(x[:end], moving_average(train_loss_4), label = 'zetac = {}'.format(zeta_array[4]))
    plt.plot(x[:end], moving_average(train_loss_5), label = 'zetac = {}'.format(zeta_array[5]))
    plt.plot(x[:end], moving_average(train_loss_6), label = 'zetac = {}'.format(zeta_array[6]))
    plt.plot(x[:end], moving_average(train_loss_7), label = 'zetac = {}'.format(zeta_array[7]))
    plt.plot(x[:end], moving_average(train_loss_8), label = 'zetac = {}'.format(zeta_array[8]))
    plt.plot(x[:end], moving_average(train_loss_9), label = 'zetac = {}'.format(zeta_array[9]))

    plt.xlabel('epoch')
    plt.ylabel('train loss')
    plt.title('{}'.format(args.dataset))
    plt.legend()
    plt.show()


    plt.figure(2)
    plt.plot(x[:end], moving_average(test_acc_0), label = 'zetac = {}'.format(zeta_array[0]))
    plt.plot(x[:end], moving_average(test_acc_1), label = 'zetac = {}'.format(zeta_array[1]))
    plt.plot(x[:end], moving_average(test_acc_2), label = 'zetac = {}'.format(zeta_array[2]))
    plt.plot(x[:end], moving_average(test_acc_3), label = 'zetac = {}'.format(zeta_array[3]))
    plt.plot(x[:end], moving_average(test_acc_4), label = 'zetac = {}'.format(zeta_array[4]))
    plt.plot(x[:end], moving_average(test_acc_5), label = 'zetac = {}'.format(zeta_array[5]))
    plt.plot(x[:end], moving_average(test_acc_6), label = 'zetac = {}'.format(zeta_array[6]))
    plt.plot(x[:end], moving_average(test_acc_7), label = 'zetac = {}'.format(zeta_array[7]))
    plt.plot(x[:end], moving_average(test_acc_8), label = 'zetac = {}'.format(zeta_array[8]))
    plt.plot(x[:end], moving_average(test_acc_9), label = 'zetac = {}'.format(zeta_array[9]))

    plt.xlabel('epoch')
    plt.ylabel('test acc')
    plt.title('{}'.format(args.dataset))
    plt.legend()
    plt.show()

