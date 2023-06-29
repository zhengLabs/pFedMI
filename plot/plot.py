import csv

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator


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


def plot_s_val(file, order, name, title, save_name, rounds, index='Testing Accuracy'):
    """
    S: the size of personalized model collection
    val : the size of personalized model collection
    """
    # Get results
    test_accuracy = get_data(file, order[0], rounds)
    test_accuracy1 = get_data(file, order[1], rounds)
    test_accuracy2 = get_data(file, order[2], rounds)
    test_accuracy3 = get_data(file, order[3], rounds)
    test_accuracy4 = get_data(file, order[4], rounds)
    test_accuracy5 = get_data(file, order[5], rounds)
    plt.figure()

    # plot
    plt.plot(range(len(test_accuracy)), test_accuracy, marker=".", markersize=3,
             label=u'' + name[0], linewidth=1.0, color='purple')
    plt.plot(range(len(test_accuracy1)), test_accuracy1, marker=".", markersize=3,
             label=u'' + name[1], linewidth=1.0, color='cornflowerblue')
    plt.plot(range(len(test_accuracy2)), test_accuracy2, marker=".", markersize=3,
             label=u'' + name[2], linewidth=1.0, color='b')
    plt.plot(range(len(test_accuracy3)), test_accuracy3, marker=".", markersize=3,
             label=u'' + name[3], linewidth=1.0, color='k')
    plt.plot(range(len(test_accuracy4)), test_accuracy4, marker=".", markersize=3,
             label=u'' + name[4], linewidth=1.0, color='r')
    plt.plot(range(len(test_accuracy5)), test_accuracy5, marker=".", markersize=3,
             label=u'' + name[5], linewidth=1.0, color='c')

    plt.tick_params(axis='both', which='major', direction='in', width=1.5, labelsize=12)  # 刻度width=2
    ax = plt.gca()

    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.xaxis.set_major_locator(MultipleLocator(rounds / 10))

    # Coordinate axis settings
    ax.set_ylim(0.62, 0.78)  # Adjust accordingly for different datasets
    ax.set_xlim(0, rounds)

    ax.spines['bottom'].set_linewidth(1.5)  # 设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1.5)  # 设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(1.5)  # 设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(1.5)

    # legend
    plt.legend(loc='lower right', fontsize=10)

    # title
    plt.ylabel(index, fontsize=11)
    plt.xlabel('Communication Rounds', fontsize=11)
    plt.title(title, fontsize=11)

    plt.rcParams['savefig.dpi'] = 1000
    plt.rcParams['figure.dpi'] = 1000
    plt.savefig(save_name)
    plt.show()


if __name__ == "__main__":
    rounds = 50

    # file = '../exp/fmnist15_s'  # M = 9 1 3 7 10 5
    # order = [1, 2, 5, 3, 0, 4]
    # name = ['|S|=1', '|S|=3', '|S|=5', '|S|=7', '|S|=9', '|S|=10']
    # title = "FMNIST, 15 clients"
    # save_name = "fmnist15_s"

    # file = '../exp/cifar15_s'  # M = 9 1 3 10 7 5
    # order = [1, 2, 5, 4, 0, 3]
    # name = ['|S|=1', '|S|=3', '|S|=5', '|S|=7', '|S|=9', '|S|=10']
    # title = "CIFAR-10, 15 clients"
    # save_name = "cifar15_s"
    # plot_s_val(file, order, name, title, save_name, rounds)

    # file = '../exp/fmnist15_val_data'  # 0.3 0.5 0.05 0.4 0.1 0.2
    # order = [2, 4, 5, 0, 3, 1]  # 图的顺序 0.1 0.3 —— 0.5 0.05
    # name = ['Val Split Ratio:0.05', 'Val Split Ratio:0.1', 'Val Split Ratio:0.2',
    #         'Val Split Ratio:0.3', 'Val Split Ratio:0.4', 'Val Split Ratio:0.5']
    # title = "FMNIST, 15 clients"
    # save_name = "fmnist15val"
    #
    # file = 'exp/cifar15_val_data'  # 0.5 0.4 0.2 0.1 0.3 0.05
    # order = [5, 3, 2, 4, 1, 0]
    # name = ['Val Split Ratio:0.05', 'Val Split Ratio:0.1', 'Val Split Ratio:0.2',
    #         'Val Split Ratio:0.3', 'Val Split Ratio:0.4', 'Val Split Ratio:0.5']
    # title = "CIFAR-10, 15 clients"
    # save_name = "cifat15val"
    # plot_s_val(file, order, name, title, save_name, rounds)
