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
import pickle


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
    return DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers),\
           DataLoader(train_set, batch_size=batch_size, sampler=dev_sampler, num_workers=num_workers)


def loss_and_accuracy_on(net, criterion, dev_loader, return_results=False):
    """
    Predicts and computes average loss per example and accuracy
    :param net: Neural Net object
    :param criterion: Loss function module
    :param dev_loader: Data loader to predict on
    :param return_results: If true returns the predictions as a list
    :return: avg. loss, accuracy [, results]
    """
    total_loss = good = 0.0
    if return_results:
        results = []

    net.eval()  # evaluation mode (for dropout)
    for x, y in dev_loader:
        out = net.forward(x)
        total_loss += criterion(out, y).item()
        pred = torch.argmax(out, dim=1)
        good += (pred == y).sum().item()
        if return_results:
            results.extend(pred.tolist())

    if return_results:
        return total_loss / len(dev_loader.sampler), good / len(dev_loader.sampler), results
    return total_loss / len(dev_loader.sampler), good / len(dev_loader.sampler)


def train_on(net, criterion, optimizer, train_loader, dev_loader, epochs=10):
    """
    Trains on train data and computes avg. loss and accuracy for train and dev each epoch and prints them
    :param net: Neural Net object
    :param criterion: Loss function module
    :param optimizer: Pytorch Optimizer
    :param train_loader: Data loader for train set
    :param dev_loader: Data loader for validation set
    :param epochs: Number of epochs to train net on
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
            optimizer.zero_grad()
            out = net.forward(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += x.shape[0]
            correct = (torch.argmax(out, dim=1) == y).sum().item()
            good += correct

            if j % 5 == 4:
                print "{} / {}".format(j, len(train_loader))

        dev_loss, dev_acc = loss_and_accuracy_on(net, criterion, dev_loader)
        size = len(train_loader.sampler)
        train_loss = total_loss / size

        print "| {:^5} | {:010f} | {:8.4f}% | {:7f} | {:6.3f}% | {:08f}s |".format(
            i, train_loss, good / size * 100.0, dev_loss,
               dev_acc * 100.0, time() - start)

        all_train_loss.append(train_loss)
        all_dev_loss.append(dev_loss)
    print "+-------+------------+-----------+----------+---------+------------+\n"
    return all_train_loss, all_dev_loss


def write_test_pred(net, prefix=""):
    with open(prefix + "test.pred", "w") as f:
        with open('test.pickle') as tp:
            test = pickle.load(tp)
        for data in test:
            image = torch.Tensor(data)
            output = net(image)
            pred = torch.argmax(output)
            f.write(str(pred) + "\n")


def run_my_cnn(use_batch_norm=False):
    # parameters
    lr = 0.001
    epochs = 15
    batch = 64
    workers = 2

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
            features = reduce(lambda x, y: x*y, x.shape[1:])
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


    net = nn.Sequential(*modules)
    criterion = nn.NLLLoss(size_average=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train_loss, dev_loss = train_on(net, criterion, optimizer, train_loader, dev_loader, epochs=epochs)

    # test
    test_loss, test_acc, test_pred = loss_and_accuracy_on(net, criterion, test_loader, return_results=True)
    print "\nLoss on test is {}, and accuracy is {}%".format(test_loss, test_acc * 100)
    print confusion_matrix(test_set.test_labels, test_pred, labels=range(10))

    # write to test.pred or a verion of it
    # write_test_pred(net, "my_")
    with open("transfer_test.pred") as f:
        f.writelines(map(lambda y: str(int(y))+"\n", test_pred))

    plot_graph(dev_loss, train_loss)


def plot_graph(dev_loss, train_loss):
    plt.close()
    plt.plot(range(len(train_loss)), train_loss, "r", label="Training Set")
    plt.plot(range(len(dev_loss)), dev_loss, "b--", label="Validation Set")
    # plt.axis([0, 9, 0.0, 0.7])
    plt.xlabel("Epoch Number")
    plt.ylabel("Average Loss per Example")
    plt.legend()
    plt.show()


def run_transfer():
    # parameters
    lr = 0.01
    epochs = 1
    batch = 500
    workers = 2

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

    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.SGD(net.fc.parameters(), lr)

    train_loss, dev_loss = train_on(net, criterion, optimizer, train_loader, dev_loader, epochs=epochs)

    # test
    test_loss, test_acc, test_pred = loss_and_accuracy_on(net, criterion, test_loader, return_results=True)
    print "\nLoss on test is {}, and accuracy is {}%".format(test_loss, test_acc * 100)
    print confusion_matrix(test_set.test_labels, test_pred, labels=range(10))

    # write to test.pred or a verion of it
    # write_test_pred(net, "transfer_")
    with open("transfer_test.pred") as f:
        f.writelines(map(lambda y: str(int(y))+"\n", test_pred))

    plot_graph(dev_loss, train_loss)

if __name__ == '__main__':
    # run_my_cnn(use_batch_norm=True)
    run_transfer()
