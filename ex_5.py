import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
from torchvision import datasets, transforms
from random import shuffle
from time import time
from matplotlib import pyplot as plt
from torchvision.models import resnet18
from sklearn.metrics import confusion_matrix
import numpy as np

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def split_train(train_set, train_part=0.8, batch_size=1, num_workers=0):
    """
    Splits the data set into 2 data loader using SubsetRandomSampler
    by counting how many examples should be for each class
    :param train_set: Dataset to split
    :param train_part: Fraction representing train loader part (e.g 0.8 = train is 80% from dataset)
    :param batch_size: Batch size for loaders
    :param num_workers: num workers
    :return: train dataloader, validation dataloader
    """
    size = len(train_set)
    train_size = int(size * train_part)
    class_size = int(train_size / 10.0)
    indices = range(size)
    shuffle(indices)

    train_indices = []
    counters = {i: 0 for i in xrange(10)}
    for i in indices:
        if counters[int(train_set[i][1])] < class_size:
            train_indices.append(i)
            counters[int(train_set[i][1])] += 1
    dev_indices = [i for i in indices if i not in train_indices]
    # train_indices, dev_indices = indices[:train_size], indices[train_size:]
    train_sampler, dev_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(dev_indices)
    return DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers), \
           DataLoader(train_set, batch_size=batch_size, sampler=dev_sampler, num_workers=num_workers)


def loss_and_accuracy_on(net, criterion, dev_loader, return_results=False, device=None):
    """
    Predicts and computes average loss per example and accuracy
    :param net: Neural Net object
    :param criterion: Loss function module
    :param dev_loader: Data loader to predict on
    :param return_results: If true returns the predictions as a list
    :param device: If net should run on device other than CPU, then device is set. Otherwise None.
    :return: avg. loss, accuracy [, results]
    """
    total_loss = good = 0.0
    if return_results:
        results = []

    net.eval()  # evaluation mode (for dropout)
    for x, y in dev_loader:

        if device is not None:
            x, y = x.to(device), y.to(device)

        out = net.forward(x)
        total_loss += criterion(out, y).item()
        pred = torch.argmax(out, dim=1)
        good += (pred == y).sum().item()
        if return_results:
            results.extend(pred.tolist())

    if return_results:
        return total_loss / len(dev_loader.sampler), good / len(dev_loader.sampler), results
    return total_loss / len(dev_loader.sampler), good / len(dev_loader.sampler)


def train_on(net, criterion, optimizer, train_loader, dev_loader, epochs=10, device=None):
    """
    Trains on train data and computes avg. loss and accuracy for train and dev each epoch and prints them
    :param net: Neural Net object
    :param criterion: Loss function module
    :param optimizer: Pytorch Optimizer
    :param train_loader: Data loader for train set
    :param dev_loader: Data loader for validation set
    :param epochs: Number of epochs to train net on
    :param device: If net should run on device other than CPU, then device is set. Otherwise None.
    :return: list of avg. train loss at each epoch, list of avg. validation loss at each epoch
    """
    print "+-------+------------+-----------+----------+---------+------------+"
    print "| epoch | train loss | train acc | dev loss | dev acc | epoch time |"
    print "+-------+------------+-----------+----------+----------------------+"

    all_train_loss, all_dev_loss = [], []
    for i in xrange(epochs):
        total = total_loss = good = 0.0
        start = time()
        net.train()  # training mode
        for j, (x, y) in enumerate(train_loader):

            if device is not None:
                x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = net.forward(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += x.shape[0]
            correct = (torch.argmax(out, dim=1) == y).sum().item()
            good += correct

        #             if j % 5 == 4:
        #                 print "{} / {}".format(j, len(train_loader))

        dev_loss, dev_acc = loss_and_accuracy_on(net, criterion, dev_loader, device=device)
        size = len(train_loader.sampler)
        train_loss = total_loss / size

        print "| {:^5} | {:010f} | {:8.4f}% | {:7f} | {:6.3f}% | {:08f}s |".format(
            i, train_loss, good / size * 100.0, dev_loss, dev_acc * 100.0, time() - start)

        all_train_loss.append(train_loss)
        all_dev_loss.append(dev_loss)
    print "+-------+------------+-----------+----------+---------+------------+\n"
    return all_train_loss, all_dev_loss


def run_my_cnn(use_batch_norm=False, use_dropout=False):
    """
    Trains my CNN on CIFAR10, produces a loss graph and a confusion matrix
    :param use_batch_norm: If True uses batches normalization layers after every convolution layer or linear layer
    :param use_dropout: If True uses dropout layers after every pool layer or after activation functions in the fully
        connected part
    :return: None
    """
    # parameters
    lr = 0.01
    epochs = 50
    batch = 64
    workers = 2
    do_prob = 0.05

    # load data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    train_loader, dev_loader = split_train(train_set, batch_size=batch)
    test_set = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch, num_workers=workers)

    # create net
    class ViewModule(nn.Module):
        def __init__(self):
            super(ViewModule, self).__init__()

        def forward(self, x):
            features = reduce(lambda x, y: x * y, x.shape[1:])
            return x.view(-1, features)

    modules = [nn.Conv2d(3, 6, 5, stride=1), nn.ReLU(), nn.MaxPool2d(2),
               nn.Conv2d(6, 16, 5, stride=1), nn.ReLU(), nn.MaxPool2d(2),
               ViewModule(),
               nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
               nn.Linear(120, 84), nn.ReLU(),
               nn.Linear(84, 10), nn.LogSoftmax(dim=1)]

    if use_batch_norm:
        modules.insert(1, nn.BatchNorm2d(6))
        modules.insert(5, nn.BatchNorm2d(16))
        modules.insert(10, nn.BatchNorm1d(120))
        modules.insert(14, nn.BatchNorm1d(84))
        if use_dropout:
            modules.insert(4, nn.Dropout2d(do_prob))
            modules.insert(9, nn.Dropout2d(do_prob))
            modules.insert(13, nn.Dropout(do_prob))
            modules.insert(16, nn.Dropout(do_prob))
    elif use_dropout:
        modules.insert(3, nn.Dropout2d(do_prob))
        modules.insert(7, nn.Dropout2d(do_prob))
        modules.insert(11, nn.Dropout(do_prob))
        modules.insert(14, nn.Dropout(do_prob))

    net = nn.Sequential(*modules)

    # cuda
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        net.to(device)

    criterion = nn.NLLLoss(size_average=False)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    train_loss, dev_loss = train_on(net, criterion, optimizer, train_loader, dev_loader, epochs=epochs, device=device)

    # test
    test_loss, test_acc, test_pred = loss_and_accuracy_on(net, criterion, test_loader, return_results=True,
                                                          device=device)
    print "\nLoss on test is {}, and accuracy is {}%".format(test_loss, test_acc * 100)
    cm = confusion_matrix(test_set.test_labels, test_pred, labels=range(10))

    # write to test.pred or a verion of it
    # write_test_pred(net, "my_")
    with open("my_test.pred", "w") as f:
        f.writelines(map(lambda y: str(int(y)) + "\n", test_pred))

    plot_graph(dev_loss, train_loss)

    #     plot_confusion_matrix(cm, "CNN Confusion Matrix")
    print "\n Conf Matrix"
    print np.array(cm)


def plot_graph(dev_loss, train_loss):
    """
    Plots the average batch loss per epoch graph for train and validation sets
    :param dev_loss: List containing the average batch loss each epoch for validation set
    :param train_loss: Likewise, but of the training set
    :return: None
    """
    f = plt.figure()
    plt.plot(range(len(train_loss)), train_loss, "r", label="Training Set")
    plt.plot(range(len(dev_loss)), dev_loss, "b--", label="Validation Set")
    # plt.axis([0, 9, 0.0, 0.7])
    plt.xlabel("Epoch Number")
    plt.ylabel("Average Loss per Example")
    plt.legend()
    plt.show()
    f.savefig("fig.png")


def run_transfer():
    """
    Runs ResNet18 as a feature extractor and trains a linear layer to recieve the ResNet18's output and classify
        CIFAR10 dataset
    :return: None
    """
    # parameters
    lr = 0.001
    epochs = 1
    batch = 64
    workers = 1

    # load data
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    train_loader, dev_loader = split_train(train_set, batch_size=batch)
    test_set = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch, num_workers=workers)

    net = resnet18(pretrained=True)
    for p in net.parameters():
        p.require_grads = False
    net.fc = nn.Linear(net.fc.in_features, 10)  # replace last linear layer

    # cuda
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        net.to(device)

    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.Adam(net.fc.parameters(), lr)

    train_loss, dev_loss = train_on(net, criterion, optimizer, train_loader, dev_loader, epochs=epochs, device=device)

    # test
    test_loss, test_acc, test_pred = loss_and_accuracy_on(net, criterion, test_loader, return_results=True,
                                                          device=device)
    print "\nLoss on test is {}, and accuracy is {}%".format(test_loss, test_acc * 100)
    cm = confusion_matrix(test_set.test_labels, test_pred, labels=range(10))

    # write to test.pred or a verion of it
    # write_test_pred(net, "transfer_")
    with open("transfer_test.pred", "w") as f:
        f.writelines(map(lambda y: str(int(y)) + "\n", test_pred))

    plot_graph(dev_loss, train_loss)

    print "\n Conf Matrix"
    print np.array(cm)


def plot_confusion_matrix(cm, title):
    """
    Plots the confusion matrix
    :param cm: Confusion Matrix - can be a list of lists or numpy matrix
    :param title: Title of figure
    :return: None
    """
    f = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    f.savefig("cm.png")


if __name__ == '__main__':
    run_my_cnn(use_batch_norm=True, use_dropout=True)
    run_transfer()
