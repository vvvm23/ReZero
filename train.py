import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import matplotlib.pyplot as plt

import tqdm
import time

from model import RZLinear, RZConv2d

TASK = 'cifar10'
TRY_CUDA = True
BATCH_SIZE = 512
ALPHA = 1e-4

NB_EPOCHS = 1000
EARLY_STOP = True
EARLY_STOP_THRESHOLD = 5
EARLY_STOP_EPS = 5e-3

MODEL_SAVING = True
DEBUG = False

IN_CHANNELS = 32
NB_DOWN_RZ = 24
NB_STATIC_RZ = 24

info = lambda s: print(f"\33[92m> {s}\33[0m")
error = lambda s: print(f"\33[31m! {s}\33[0m")
debug = lambda s: print(f"\33[93m? {s}\33[0m") if DEBUG else None

def get_device():
    if TRY_CUDA == False:
        info("CUDA disabled by hyperparameters.")
        return torch.device('cpu')
    if torch.cuda.is_available():
        info("CUDA is available.")
        return torch.device('cuda')
    error("CUDA is unavailable but selected in hyperparameters.")
    error("Falling back to default device.")
    return torch.device('cpu')

def create_down_block(in_channels, out_channels, rz_layers):
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
    layers.append(nn.MaxPool2d(2))
    for _ in range(rz_layers):
        layers.append(RZConv2d(out_channels, out_channels, 3, padding=1))
    return nn.Sequential(*layers)

def create_model(task):
    layers = []
    if task == 'mnist':
        layers.append(create_down_block(1, IN_CHANNELS, NB_DOWN_RZ))
        layers.append(create_down_block(IN_CHANNELS, IN_CHANNELS*2, NB_DOWN_RZ))
        for _ in range(NB_STATIC_RZ):
            layers.append(RZConv2d(IN_CHANNELS*2, IN_CHANNELS*2, 3, padding=1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(IN_CHANNELS*2*7*7, 256))
        layers.append(nn.Linear(256, 10))
        layers.append(nn.Softmax(dim=-1))
    elif task == 'cifar10':
        layers.append(create_down_block(3, IN_CHANNELS, NB_DOWN_RZ))
        layers.append(create_down_block(IN_CHANNELS, IN_CHANNELS*2, NB_DOWN_RZ))
        layers.append(create_down_block(IN_CHANNELS*2, IN_CHANNELS*4, NB_DOWN_RZ))
        layers.append(create_down_block(IN_CHANNELS*4, IN_CHANNELS*8, NB_DOWN_RZ))
        for _ in range(NB_STATIC_RZ):
            layers.append(RZConv2d(IN_CHANNELS*8, IN_CHANNELS*8, 3, padding=1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(IN_CHANNELS*8*2*2, 128))
        layers.append(nn.Linear(128, 10))
        layers.append(nn.Softmax(dim=-1))
    else:
        error(f"Unrecognised task {task}!")
        error("Exiting..\n")
        exit()

    return nn.Sequential(*layers)

def load_dataset(task, batch_size):
    if task == 'mnist':
        train_dataset = torchvision.datasets.MNIST('data', train=True, download=True, transform=torchvision.transforms.ToTensor())
        test_dataset = torchvision.datasets.MNIST('data', train=False, download=True, transform=torchvision.transforms.ToTensor())
    elif task == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=torchvision.transforms.ToTensor())
        test_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=torchvision.transforms.ToTensor())
    else:
        error(f"Unrecognised task {task}!")
        error("Exiting..\n")
        exit()

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    return train_loader, test_loader, train_dataset, test_dataset

# Trains the given model on the data loaded by loader for one epoch, given an optimizer and criterion
def train(model, loader, optim, crit, device):
    cumulative_loss = 0.0

    model.train()

    for x, y in tqdm.tqdm(loader):
        optim.zero_grad()
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = crit(pred, y)
        cumulative_loss += loss.item()
        loss.backward()
        optim.step()

    return cumulative_loss / len(loader)

# Evaluates the given model by loss and accuracy given a test dataloader, optimizer(?) and criterion
def evaluate(model, loader, optim, crit, device):
    cumulative_loss = 0.0
    correct_pred = 0

    model.eval()

    for x, y in loader:
        optim.zero_grad()
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = crit(pred, y)
        cumulative_loss += loss.item()

        _, pred = torch.max(pred.data, -1)
        correct_pred += (pred == y).sum().item()

    return cumulative_loss / len(loader), correct_pred * 100.0 / (len(loader) * BATCH_SIZE)

if __name__ == "__main__":
    device = get_device()
    train_loader, test_loader, _, _ = load_dataset(TASK, BATCH_SIZE)
    model = create_model(TASK).to(device)
    info(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

    optim = torch.optim.Adam(model.parameters(), lr=ALPHA)
    crit = nn.CrossEntropyLoss()

    best_loss = 999.9
    loss_fail_count = 0

    for ei in range(NB_EPOCHS):
        info(f"Epoch {ei+1}/{NB_EPOCHS}")
        train_loss = train(model, train_loader, optim, crit, device)
        eval_loss, eval_accuracy = evaluate(model, test_loader, optim, crit, device)
        info(f"Training Loss: {train_loss}")
        info(f"Evaluation Loss: {eval_loss}")
        info(f"Evaluation Accuracy: {eval_accuracy:.2f}%\n")

        if eval_loss <= best_loss - EARLY_STOP_EPS:
            debug("New best loss, resetting counter")
            best_loss = eval_loss
            loss_fail_count = 0
        else:
            debug("Best loss not beat, increasing counter")
            loss_fail_count += 1

        if EARLY_STOP and loss_fail_count >= EARLY_STOP_THRESHOLD:
            info("Early stopping threshold reached.")
            break
