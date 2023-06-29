import copy
import numpy as np
import torch
from torch.utils.data import DataLoader


class client(object):
    def __init__(self, trainDataSet, dev, testDataSet=None):
        self.mu = 0.2
        self.max_f = 150

        self.train_ds = trainDataSet
        self.test_ds = testDataSet
        self.train_ds_num = 0
        self.test_ds_num = 0
        self.train_dl = None
        self.test_dl = None
        self.epoch = 0
        self.bs = 0
        self.lr = 0
        self.opti = None
        self.loss_fun = None
        self.dev = dev
        self.eNet = None
        self.Net = None
        self.parti = 0
        self.fre = None
        self.last_para = None
        self.para = None
        self.paras = []
        self.order = []
        self.net = []

    def local_train(self, algo, dn, isC=True):
        self.max_f = dn
        info = None
        if algo == 'FedCo':
            loss, info = self.FedCo(isC)
        else:
            pass

        test_acc = self.test()

        return self.para, test_acc, loss, info

    def FedCo(self, isC):
        if self.parti == 0:
            self.parti = 1
            train_loss, info = self.localUpdate(isC)
            return train_loss, info

        W, isBest = self.calW_byft()
        temp_para = None
        if np.sum(W) != 0:
            for i in range(len(self.paras)):
                if temp_para is None:
                    temp_para = copy.deepcopy(self.paras[i])
                    for item in self.paras[i]:
                        temp_para[item] = W[i] * self.paras[i][item].clone()
                else:
                    for item in self.paras[i]:
                        temp_para[item] += W[i] * self.paras[i][item].clone()
            self.para = copy.deepcopy(temp_para)
        train_loss, info = self.localUpdate(isC)
        return train_loss, info

    def calW_byft(self):
        n = len(self.fre)
        m = len(self.order)
        w_0 = 0

        mi, sym, isBest = self.calMI()
        update_w = []
        old_w = []
        new_w = []
        for i in range(m):
            new_w.append(mi[i] * sym[i])
            if self.fre[self.order[i]] != -1:
                old_w.append(self.fre[self.order[i]])
            else:
                old_w.append(0)
        old_w = np.divide(old_w, np.sum(old_w) if np.sum(old_w) > 0.0 else 1)
        new_w = np.divide(new_w, np.sum(new_w) if np.sum(new_w) > 0.0 else 1)

        mul = m / n
        for i in range(m):
            if new_w[i] == 0:
                pass

            elif self.fre[self.order[i]] == -1:
                self.fre[self.order[i]] = w_0 + new_w[i] * mul

            else:
                self.fre[self.order[i]] = self.fre[self.order[i]] + new_w[i] * mul

            w = old_w[i] + self.mu * new_w[i]
            update_w.append(w)

        for i in range(len(update_w)):
            if update_w[i] != 0:
                update_w[i] = np.exp(update_w[i])
        update_w = np.divide(update_w, np.sum(update_w) if np.sum(update_w) > 0.0 else 1)

        sums = 0.0
        for item in self.fre:
            if item != -1:
                sums += item

        for i in range(len(self.fre)):
            if self.fre[i] > 0:
                self.fre[i] /= sums if sums > 0.0 else 1

        return update_w, isBest

    def calMI(self):
        mi = []
        sym = []
        self.Net.load_state_dict(self.para, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=self.bs, shuffle=True)
        fm1 = None
        fm2_list = []
        loss1 = 0.0
        loss2_list = []
        local_flag = False
        for par in self.paras:
            loss2 = 0.0
            fm2 = None
            f = 0
            with torch.no_grad():
                self.eNet.load_state_dict(par, strict=True)
                for data, label in self.train_dl:
                    if f == self.max_f:
                        break
                    f += 1
                    data, label = data.to(self.dev), label.to(self.dev)

                    if local_flag is False:
                        l, _, preds1 = self.Net(data)
                        loss1 += self.loss_fun(preds1, label).item()
                        if fm1 is None:
                            fm1 = self.calsim_feature_map(_)
                        else:
                            fm1 = np.add(fm1, self.calsim_feature_map(_))
                    l, __, preds2 = self.eNet(data)

                    if fm2 is None:
                        fm2 = self.calsim_feature_map(__)
                    else:
                        fm2 = np.add(fm2, self.calsim_feature_map(__))

                    loss2 += self.loss_fun(preds2, label).item()
            local_flag = True
            fm2_list.append(fm2)
            loss2_list.append(loss2)

        if np.min(loss2_list) > loss1:
            isBest = -1
        else:
            isBest = 0
            for i in range(len(fm2_list)):
                # sim = self.cos_sim(fm1, fm2_list[i])
                sim = torch.norm(fm1 - fm2_list[i])
                sim = -np.log2(1 - sim * sim) / 2
                mi.append(sim)
                if loss2_list[i] < loss1:
                    sym.append(1)
                else:
                    sym.append(0)
        return mi, sym, isBest

    def calsim_feature_map(self, tensor):
        l = None
        for i in range(len(tensor)):  # 10
            temp = []
            for j in range(len(tensor[0])):  # 64
                count = torch.count_nonzero(tensor[i][j])
                num = len(tensor[i][j]) * len(tensor[i][j][0])
                temp.append(torch.divide(count, num))
            temp = torch.tensor(temp)
            if l is None:
                l = temp
            else:
                l = torch.add(l, temp)
        return l
        return W

    def localUpdate(self, isC):
        self.Net.load_state_dict(self.para, strict=True)
        self.opti = torch.optim.SGD(self.Net.parameters(), lr=self.lr)
        self.train_dl = DataLoader(self.train_ds, batch_size=self.bs, shuffle=True)
        vector = None
        train_loss = 0

        for epoch in range(self.epoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                self.opti.zero_grad()
                info, _, pre = self.Net(data)

                if epoch == self.epoch - 1 and isC is True:
                    if vector is None:
                        vector = self.calsim_feature_map(_)
                    else:
                        vector = np.add(vector, self.calsim_feature_map(_))
                loss = self.loss_fun(pre, label)
                train_loss += loss.item()
                loss.backward()
                self.opti.step()
        if isC:
            for i in range(len(vector)):
                vector[i] = np.divide(vector[i], self.train_ds_num)
        self.para = copy.deepcopy(self.Net.state_dict())
        return train_loss / self.epoch, vector

    def test(self, para=None):
        self.test_dl = DataLoader(self.test_ds, batch_size=self.bs, shuffle=True)
        sum_accu = 0
        num = 0
        with torch.no_grad():
            for data, label in self.test_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                l, _, pre = self.Net(data)
                pre = torch.argmax(pre, dim=1)
                sum_accu += (pre == label).float().mean()
                num += 1
        return (sum_accu / num) * self.train_ds_num
