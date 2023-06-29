import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from train import client

from getData import GetDataSet

np.random.seed(1)


class ClientsGroup(object):
    def ___init__(self, dataSetName, isIID, numOfClients, dev):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}
        self.train_data_num = 0
        self.test_data_loader = None
        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        DataSet = GetDataSet(self.data_set_name, self.is_iid)
        self.train_data_num = DataSet.train_data_size
        test_data = torch.tensor(DataSet.test_data)
        test_label = torch.argmax(torch.tensor(DataSet.test_label), dim=1)
        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)

        train_data = DataSet.train_data
        train_label = DataSet.train_label

        class_num = 6
        train_shard_size = DataSet.train_data_size // self.num_of_clients // class_num
        train_shards_id = np.random.permutation(DataSet.train_data_size // train_shard_size)

        print(train_shards_id)
        net_cls_counts = {}

        for i in range(self.num_of_clients):
            # train data
            train_shard_ids = []
            for j in range(class_num):
                train_shard_ids.append(train_shards_id[i * class_num + j])

            train_data_shards = []
            train_label_shards = []
            for j in range(class_num):
                train_data_shards.append(train_data[
                                         train_shard_ids[j] * train_shard_size: train_shard_ids[
                                                                                    j] * train_shard_size + train_shard_size])

                train_label_shards.append(train_label[
                                          train_shard_ids[j] * train_shard_size: train_shard_ids[
                                                                                     j] * train_shard_size + train_shard_size])

            train_data_ = (elem for elem in train_data_shards)
            train_label_ = (elem for elem in train_label_shards)
            train_local_data = np.vstack(train_data_)
            train_local_label = np.vstack(train_label_)
            train_local_label = np.argmax(train_local_label, axis=1)

            unq, unq_cnt = np.unique(train_local_label, return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts['client{}'.format(i)] = tmp

            train_loader = TensorDataset(torch.tensor(train_local_data), torch.tensor(train_local_label))
            someone = client(train_loader, self.dev)
            self.clients_set['client{}'.format(i)] = someone
            self.clients_set['client{}'.format(i)].train_ds_num = len(train_local_label)

        client_data, client_test_num = self.get_test_dataloader(test_label,
                                                                TensorDataset(test_data, test_label),
                                                                net_cls_counts)
        for i in range(self.num_of_clients):
            self.clients_set['client{}'.format(i)].test_ds = client_data[i]
            self.clients_set['client{}'.format(i)].test_ds_num = client_test_num[i]
        print(net_cls_counts)

    def get_test_dataloader(self, y_test, test_ds, traindata_cls_counts, num_class=10):
        # return test_dl
        # save by label
        test_data = [[] for i in range(num_class)]
        for i in range(len(y_test)):
            test_data[y_test[i]].append(test_ds[i])
        # client test data
        client_num = len(traindata_cls_counts)
        client_data = [[] for i in range(client_num)]
        client_test_num = dict()

        alpha = 0.2
        for i in range(client_num):
            for label, num in traindata_cls_counts['client{}'.format(i)].items():
                test_num = int(alpha * num)
                # test_num = len(test_data[label])
                # test_data have enough data
                order = np.random.permutation(len(test_data[label]))
                # select by order
                client_data[i].extend([test_data[label][k] for k in order[0:test_num]])
            client_test_num[i] = len(client_data[i])

        return client_data, client_test_num


if __name__ == "__main__":
    MyClients = ClientsGroup('fmnist', False, 15, 0)
    print(MyClients)
    # print(MyClients.clients_set['client10'].train_ds[0:100])
    # print(MyClients.clients_set['client11'].train_ds[400:500])
