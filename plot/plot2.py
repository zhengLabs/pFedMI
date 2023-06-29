import csv
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import rcParams

config = {
    "font.family": "serif",
    # "font.size": 24,
    "mathtext.fontset": "stix",
    "font.serif": ["SimHei"],
}
rcParams.update(config)


def get_data(fliename, i, rounds):
    train_accuracy = []
    with open("{}.csv".format(fliename), "r", encoding="utf-8") as f:
        f_read = csv.reader(f)
        num = 0
        for v in f_read:

            if len(v) != 0 and num != 0:
                train_accuracy.append(float(v[i]))
            num += 1
            if num == rounds + 1:
                break
    return train_accuracy


def plot_MI(file, order, name, title, save_name, rounds, index='Testing  Accuracy'):
    """
    Verify the validity of mutual information
    """
    test_accuracy = get_data(file, order[0], rounds)
    test_accuracy1 = get_data(file, order[1], rounds)
    test_accuracy2 = get_data(file, order[2], rounds)
    plt.figure()

    plt.plot(range(len(test_accuracy)), test_accuracy, marker=".", markersize=3,
             label=u'' + name[0], linewidth=1.0, color='r')
    plt.plot(range(len(test_accuracy1)), test_accuracy1, marker=".", markersize=3,
             label=u'' + name[1], linewidth=1.0, color='b')
    plt.plot(range(len(test_accuracy2)), test_accuracy2, marker=".", markersize=3,
             label=u'' + name[2], linewidth=1.0, color='purple')

    plt.tick_params(axis='both', which='major', direction='in', width=1, labelsize=10)  # 刻度width=2

    ax = plt.gca()

    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.xaxis.set_major_locator(MultipleLocator(rounds / 10))

    # ax.set_ylim(0.64, 0.76)
    ax.set_ylim(0.92, 0.97)
    ax.set_xlim(0, rounds)

    ax.spines['bottom'].set_linewidth(0.6)  # 设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(0.6)  # 设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(0.6)  # 设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(0.6)

    plt.legend(loc='lower right', fontsize=10)

    plt.text(0.02, 0.94, s='$\mathrm{(b)}$', fontsize=10, transform=ax.transAxes)

    ax.set_ylabel(index, fontsize=10)
    ax.set_xlabel('Communication Rounds', fontsize=10)
    plt.title(title, fontsize=10)

    plt.rcParams['savefig.dpi'] = 1000
    plt.rcParams['figure.dpi'] = 1000
    plt.savefig('zn' + save_name, format='jpg', transparent=True)
    plt.show()


def plot(file, order, name, title, save_name, rounds, index='Testing  Accuracy'):
    """
    Compare the difference weighting schemes(Loss and Avg)
    """
    test_accuracy = get_data(file, order[0], rounds)
    test_accuracy1 = get_data(file, order[1], rounds)
    plt.figure()

    plt.plot(range(len(test_accuracy)), test_accuracy, marker=".", markersize=3, markevery=3,
             label=u'' + name[0], linewidth=1.0, color='r')
    plt.plot(range(len(test_accuracy1)), test_accuracy1, marker=".", markersize=3, markevery=3,
             label=u'' + name[1], linewidth=1.0, color='b')

    plt.tick_params(axis='both', which='major', direction='in', width=1, labelsize=10)  # 刻度width=2

    ax = plt.gca()

    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.xaxis.set_major_locator(MultipleLocator(rounds / 10))

    plt.ylim(0.64, 0.76)
    plt.xlim(0, rounds)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['right'].set_linewidth(0.6)
    ax.spines['top'].set_linewidth(0.6)

    plt.legend(loc='lower right', fontsize=10)
    plt.text(0.02, 0.94, s='$\mathrm{(a)}$', fontsize=10, transform=ax.transAxes)

    plt.ylabel(index, fontsize=10)
    plt.xlabel('Communication Rounds', fontsize=13)  # 'Communication Rounds'
    plt.title(title, fontsize=10)

    # 保存
    plt.rcParams['savefig.dpi'] = 1000
    plt.rcParams['figure.dpi'] = 1000
    plt.savefig('zn' + save_name, format='jpg', transparent=True)
    plt.show()


if __name__ == '__main__':
    rounds = 50
    # file = '../exp/fmnist15_w'
    # order = [1, 2, 0]
    # name = ['$\mathrm{MI}$', '$\mathrm{Fomo}$', '$\mathrm{Avg}$']
    # title = r"$\mathrm{FMNIST}$数据集, $\mathrm{15}$个客户机"
    # save_name = "fmnist15_w.jpg"

    # file = '../exp/cifar15_w'
    # order = [1, 0, 2]
    # name = ['$\mathrm{MI}$', '$\mathrm{Fomo}$', '$\mathrm{Avg}$']
    # title = r"$\mathrm{CIFAR}$-10数据集, $\mathrm{15}$个客户机"
    # save_name = "cifar15_w.jpg"
    # plot_MI(file, order, name, title, save_name, rounds,'平均测试准确率')

    # order = [1, 0]
    # name = ['$\mathrm{Avg}$', '$\mathrm{Loss}$']

    # CIFAR
    # rounds = 100
    # file = '../exp/avg_w/cifar50_avg_w'
    # title = r"$\mathrm{CIFAR}$-10数据集, $\mathrm{50}$个客户机"
    # save_name = "cifar50_avg_w.jpg"

    # rounds = 50
    # file = '../exp/avg_w/cifar15_avg_w'
    # title = r"$\mathrm{CIFAR}$-10数据集, $\mathrm{15}$个客户机"
    # save_name = "cifar15_avg_w.jpg"
    # FMNIST

    # rounds = 50
    # file = '../exp/avg_w/fmnist15_avg_w'
    # title = r"$\mathrm{FMNIST}$数据集, $\mathrm{15}$个客户机"
    # save_name = "fmnist15_avg_w.jpg"
    #
    # rounds = 100
    # file = '../exp/avg_w/fmnist50_avg_w'
    # title = r"$\mathrm{FMNIST}$数据集, $\mathrm{50}$个客户机"
    # save_name = "fmnist50_avg_w.jpg"
    # plot(file, order, name, title, save_name, rounds, '平均测试准确率')
