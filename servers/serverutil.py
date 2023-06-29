import copy
import math
import numpy as np
import torch

from torch import optim
from model.Models import Mnist_CNN, Cifar_CNN, Mnist_2NN


def client_init(myClients, model_name, lr, loss_fun, bs, epoch, set_len, algo):

    if model_name == 'cifar_cnn':
        for i, client in myClients.clients_set.items():
            client.Net = Cifar_CNN()
            client.eNet = Cifar_CNN()
            client.Net = client.Net.to(client.dev)
            client.Net = client.eNet.to(client.dev)
            client.opti = optim.SGD(client.Net.parameters(), lr=lr)
            client.opti_e = optim.SGD(client.eNet.parameters(), lr=lr)
            client.lr = lr
            client.bs = bs
            client.epoch = epoch
            client.loss_fun = loss_fun
            client.paras = [None] * set_len
            if algo == "FedCo":
                for j in range(set_len):
                    client.net.append(Cifar_CNN())
            client.fre = [-1] * (len(myClients.clients_set) + 1)
            client.order = [-1] * set_len
    elif model_name == 'fmnist_cnn':
        for i, client in myClients.clients_set.items():
            client.Net = Mnist_CNN()
            client.eNet = Mnist_CNN()
            client.Net = client.Net.to(client.dev)
            client.Net = client.eNet.to(client.dev)
            client.opti = optim.SGD(client.Net.parameters(), lr=lr)
            client.opti_e = optim.SGD(client.eNet.parameters(), lr=lr)
            client.lr = lr
            client.bs = bs
            client.epoch = epoch
            client.loss_fun = loss_fun
            client.paras = [None] * set_len
            if algo == "FedCo":
                for j in range(set_len):
                    client.net.append(Cifar_CNN())
            client.fre = [-1] * (len(myClients.clients_set) + 1)
            client.order = [-1] * set_len

    elif model_name == 'fmnist_2nn':
        for i, client in myClients.clients_set.items():
            client.Net = Mnist_2NN()
            client.eNet = Mnist_2NN()
            client.Net = client.Net.to(client.dev)
            client.Net = client.eNet.to(client.dev)
            client.opti = optim.SGD(client.Net.parameters(), lr=lr)
            client.opti_e = optim.SGD(client.eNet.parameters(), lr=lr)
            client.lr = lr
            client.bs = bs
            client.epoch = epoch
            client.loss_fun = loss_fun
            client.paras = [None] * set_len
            if algo == "FedCo":
                for j in range(set_len):
                    client.net.append(Mnist_2NN())
            client.fre = [-1] * (len(myClients.clients_set) + 1)
            client.order = [-1] * set_len


def select_client(num_of_clients, fraction):
    num_in_comm = int(max(num_of_clients * fraction, 1))
    order = np.random.permutation(num_of_clients)
    clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]
    return clients_in_comm


def send_parameter(myClients, client, parameters, isOne):
    """Sending a single global model"""
    if isOne:
        for i, parameter in parameters.items():
            if myClients.clients_set[client].para is None:
                myClients.clients_set[client].para = copy.deepcopy(parameter)
    else:
        for idx, (i, parameter) in enumerate(parameters.items()):
            if idx >= len(myClients.clients_set[client].paras):
                break
            myClients.clients_set[client].paras[idx] = copy.deepcopy(parameter)
            myClients.clients_set[client].order[idx] = i

            if myClients.clients_set[client].para is None and idx == len(parameters) - 1:
                myClients.clients_set[client].para = copy.deepcopy(parameter)


def update_info(Clients_Model, Clients_New_Model, info, info_new):
    for name, val in Clients_New_Model.items():
        Clients_Model[name] = copy.deepcopy(val)
    for name, val in info_new.items():
        info[name] = val
    return Clients_Model, info


def update_last_clients(clients_in_comm, last_clients, n):
    for client in clients_in_comm:
        if len(last_clients) < n:
            if client in last_clients:
                index = last_clients.index(client)
                del last_clients[index]
            last_clients.append(client)
        else:
            if client not in last_clients:
                index = 0
            else:
                index = last_clients.index(client)
            del last_clients[index]
            last_clients.append(client)
    return last_clients


def select_last_model(g_para, last_clients, base_client, k, Clients_Model, rounds, n):
    para_set = []
    para_order = []
    if rounds != n and Clients_Model[base_client] is not None:
        orders = np.random.permutation(len(last_clients))
        for i in orders:
            client = last_clients[i]
            if client != base_client and Clients_Model[client] is not None:
                para_set.append(Clients_Model[client])
                if len(client) == 7:
                    para_order.append(int(client[-1]))
                else:
                    para_order.append(int(client[-2:]))
            if len(para_set) == k:
                break
    para_set.append(g_para)
    para_order.append(len(Clients_Model))
    order_para = {key: val for key, val in zip(para_order, para_set)}
    return order_para


def select_model(info, Clients_Model, g_para, order, client, k, random=False):
    """
       @para
        info : dict None
        order : list
        Clients_Model : dict None
    """
    para_set = []
    para_order = []
    Set = []
    s = 0
    for item in Clients_Model:
        if item is not None:
            s += 1
    # 第一轮或者客户机从未参加
    if s == 0 or Clients_Model[client] is None:
        pass

    #  random
    elif random is True:

        orders = np.random.permutation(len(Clients_Model))
        for i in orders:
            if Clients_Model["client{}".format(i)] is not None and "client{}".format(i) != client:
                Set.append("client{}".format(i))
            if len(Set) == k:
                break
    else:
        index = -1
        for i in range(len(order)):
            if order[i] == client:
                index = i
                break
        left = index - 1
        right = index + 1
        while len(Set) < k:
            if left < 0 or info[order[left]] is None:
                Set.append(order[right])
                right += 1
            elif right >= len(order) or info[order[right]] is None:
                Set.append(order[left])
                left -= 1
            elif math.fabs(info[order[left]] - info[client]) > math.fabs(info[order[right]] - info[client]):
                Set.append(order[right])
                right += 1
            else:
                Set.append(order[left])
                left -= 1

    if len(Set) != 0:
        para_set = [Clients_Model[c] for c in Set]
        for c in Set:
            if len(c) == 7:
                para_order.append(int(c[-1]))
            else:
                para_order.append(int(c[-2:]))
    para_set.append(g_para)
    para_order.append(len(Clients_Model))
    order_para = {key: val for key, val in zip(para_order, para_set)}
    return order_para


def cal_midis(info):
    n = len(info)
    dis = dict()
    base = None
    for i in range(n):
        if info['client{}'.format(i)] is not None:
            base = 'client{}'.format(i)

    for i in range(n):
        if info['client{}'.format(i)] is not None:
            # sim = cos_sim(info[i], info[base])
            sim = torch.linalg.norm(torch.subtract(info['client{}'.format(i)], info[base]))
            dis['client{}'.format(i)] = sim
        else:
            dis['client{}'.format(i)] = -1
    return dis


def caldis(info):
    n = len(info)
    dis = dict()
    base = None
    for i in range(n):
        if info['client{}'.format(i)] is not None:
            base = 'client{}'.format(i)

    for i in range(n):
        if info['client{}'.format(i)] is not None:
            # output = 0
            # for j in range(len(info['client{}'.format(i)])):
            #     x = info['client{}'.format(i)][j] - info[base][j]
            #     output += LA.vector_norm(x, ord=2)
            # dis['client{}'.format(i)] = output
            output1 = []
            output2 = []
            for j in range(len(info['client{}'.format(i)])):
                x = torch.max(info['client{}'.format(i)][j])
                y = torch.max(info[base][j])
                output1.append(x)
                output2.append(y)
            sim = cos_sim(torch.tensor(output1), torch.tensor(output2))
            dis['client{}'.format(i)] = sim
        else:
            dis['client{}'.format(i)] = None
    return dis


def client_sort(info, by_loss=True):
    """客户端排序"""
    order = []
    ms = dict()
    for k, v in info.items():
        ms[k] = v

    if by_loss:
        for i in range(len(ms)):
            if ms["client{}".format(i)] is None:
                ms["client{}".format(i)] = -1
        i = sorted(ms.items(), key=lambda d1: d1[1])
        order = [k for k, v in i]

    return order


def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim
