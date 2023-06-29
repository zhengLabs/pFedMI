import copy
import os
import argparse
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from model.Models import Mnist_2NN, Mnist_CNN, Cifar_CNN
from serverutil import client_sort, client_init, send_parameter
from serverutil import update_info, select_client, cal_midis, update_last_clients, select_model
from users.ClientGroup import ClientsGroup

"""para"""
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='3', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01)
parser.add_argument('-E', '--epoch', type=int, default=5)
parser.add_argument('-B', '--batchsize', type=int, default=10)
parser.add_argument('-iid', '--IID', type=int, default=0)

parser.add_argument('-mn', '--model_name', type=str, default='fmnist_cnn')
parser.add_argument('-dn', '--data_name', type=str, default='fmnist')

parser.add_argument('-nc', '--num_of_clients', type=int, default=15)
parser.add_argument('-cf', '--fraction', type=float, default=1)

parser.add_argument('-ncomm', '--num_comm', type=int, default=50)

parser.add_argument('-l', '--last_size', type=int, default=15)
parser.add_argument('-s', '--ssize', type=int, default=5)  # send_size

parser.add_argument('-r', '--random', type=bool, default=False)

parser.add_argument('-w', '--wand', type=int, default=1)

parser.add_argument('-algo', '--algorithm', type=str, default='FedCo')

"""
'FedCo' is our algorithm , we can use it in Fed(_m, _R)
"""


def Fed(_m, _R):
    global last_clients_in_comm
    Clients_Model = {'client{}'.format(i): None for i in range(args['num_of_clients'])}
    Clients_New_Model = dict()
    info_ = None
    info = {'client{}'.format(i): None for i in range(args['num_of_clients'])}
    info_new = dict()
    random = args['random']
    last_clients = []

    #   fmnist    cifar
    # 15    400        333
    # 50    120        100
    dn = -1
    if args['data_name'] == 'cifar' and args['num_of_clients'] == 15:
        dn = _R * 333
    if args['data_name'] == 'cifar' and args['num_of_clients'] == 50:
        dn = _R * 100
    if args['data_name'] == 'fmnist' and args['num_of_clients'] == 15:
        dn = _R * 400
    if args['data_name'] == 'fmnist' and args['num_of_clients'] == 50:
        dn = _R * 120

    client_init(myClients, args['model_name'], args["learning_rate"], loss_func,
                args['batchsize'], args['epoch'], _m, args['algorithm'])

    for i in range(args['num_comm']):
        print("communicate round {}".format(i + 1))

        clients_in_comm = select_client(args['num_of_clients'], args['fraction'])

        sum_parameters = None
        test_acc_client = 0
        train_loss_client = 0
        per_train_data_num = 0
        client_order = None

        if i != 0:
            Clients_Model, info = update_info(Clients_Model, Clients_New_Model, info, info_new)
            Clients_New_Model = dict()
            info_new = dict()
            last_clients = update_last_clients(last_clients_in_comm, last_clients, args['last_size'])
            if random is False:
                info_ = cal_midis(info)
                client_order = client_sort(info_)

        for client in clients_in_comm:
            per_train_data_num += myClients.clients_set[client].train_ds_num
            # print(client, "train_num:{}".format(myClients.clients_set[client].train_ds_num),
            #       "test_num :{}".format(myClients.clients_set[client].test_ds_num))

        for client in tqdm(clients_in_comm):

            # para_dict = select_last_model(global_parameters, last_clients, client_order, client, args['ssize'],
            # Clients_Model, i, n)
            para_dict = select_model(info_, Clients_Model, global_parameters, client_order, client, args['ssize'],
                                     random)
            # assert len(para_dict) == 1 or len(para_dict) == 6, "l {} len={}".format(len(last_clients), len(para_dict))

            send_parameter(myClients, client, para_dict, False)

            local_para, acc, loss, info1 = myClients.clients_set[client].local_train(args['algorithm'], dn, not random)

            Clients_New_Model[client] = copy.deepcopy(local_para)
            info_new[client] = info1
            train_loss_client += loss * myClients.clients_set[client].train_ds_num
            test_acc_client += acc

            if sum_parameters is None:
                sum_parameters = {}
                for key in local_para:
                    sum_parameters[key] = local_para[key].clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] += local_para[var].clone()

        last_clients_in_comm = copy.deepcopy(clients_in_comm)
        "weight aggregate"
        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / len(clients_in_comm))

        test_acc_client = test_acc_client / per_train_data_num
        train_loss_client = train_loss_client / per_train_data_num

        print('train_loss: {}'.format(train_loss_client))
        print('client_accuracy: {}'.format(test_acc_client))

        if args['wand'] == 1:
            wandb.log({"client_accuracy": test_acc_client, "round": i})
            wandb.log({"train_loss": train_loss_client, "round": i})


if __name__ == "__main__":
    """convert to list"""
    args = parser.parse_args()
    args = args.__dict__
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = None

    if args['model_name'] == 'fmnist_2nn':
        net = Mnist_2NN()
    elif args['model_name'] == 'fmnist_cnn':
        net = Mnist_CNN()
    elif args['model_name'] == 'cifar_cnn':
        net = Cifar_CNN()
    net = net.to(dev)

    loss_func = F.cross_entropy

    myClients = ClientsGroup(args['data_name'], args['IID'], args['num_of_clients'], dev)
    testDataLoader = myClients.test_data_loader

    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    m = 5  # the size of personalized model collection
    R = 0.4  # the size of personalized model collection
    if args['wand'] == 1:
        wandb.init(
            project=str(args['data_name']) + str(args['num_of_clients']),
            name=str(args['algorithm']) + '_s=' + str(m) + '_R=' + str(R))
        wandb.config = {
            "learning_rate": 0.01,
            "num_of_clients": args['num_of_clients'],
            "fraction": args['fraction']
        }
        Fed(m, R)
