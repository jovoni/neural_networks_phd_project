
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloaders
import pandas as pd
from models import twoLNN, fiveLNN
from utils import use_gpu_if_possible
import argparse

MODELS = {
    "twoLNN": twoLNN,
    "fiveLNN": fiveLNN
}

def train_parity(model_name, K, batch_size=128, nepochs=20, lr=.01):
    # get device
    device = use_gpu_if_possible()

    print(f"Using device : {device}")

    # Load model
    model = MODELS[model_name](K).to(device)
    loss_function = nn.BCELoss()
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    # Init results table
    results = pd.DataFrame()

    # Start training
    for e in range(nepochs):
        print(f"epoch: {e+1}")

        # Prep data
        dataloaders = get_dataloaders("parity", K, batch_size)

        running_loss = 0.0
        train_acc = 0
        for _, batch in enumerate(dataloaders['train']):
            X, Y = batch
            
            X = X.to(device)
            Y = Y.to(device)

            out = torch.sigmoid(model(X)).reshape(-1)

            loss = loss_function(out, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_acc += (out.round() == Y).sum()

        # Test Evalutaion
        test_acc = 0
        for X, Y in dataloaders['test']:
            X = X.to(device)
            Y = Y.to(device)

            out = torch.sigmoid(model(X)).reshape(-1)
            test_acc += (out.round() == Y).sum()

        train_acc = train_acc / len(dataloaders['train'].dataset)
        test_acc = test_acc / len(dataloaders['test'].dataset)

        new_results = pd.DataFrame({"epoch": e, "train_acc": train_acc.item(), "test_acc": test_acc.item()}, index=[0])
        results = pd.concat([results, new_results])

        print(f"\tTrain acc = {train_acc:.4f}")
        print(f"\tTest acc  = {test_acc:.4f}")
        
    # save results
    name = f'parity_{model_name}_{K}_{lr}_{batch_size}_{nepochs}'

    results_path = "results/" + name + ".csv"
    results.to_csv(results_path, sep=",")

    # save model
    model_path = "models/" + name + ".pt"
    torch.save(model.state_dict(), model_path)
    
    print("DONE") 


def train_classification(model_name, K, batch_size=128, nepochs=20, lr=.01):
    # get device
    device = use_gpu_if_possible()
    device = "cpu"

    print(f"Using device : {device}")

    # Load model freezing certain parameters
    trained_model_path = f'models/parity_{model_name}_{K}_{lr}_{batch_size}_{nepochs}.pt'
    model = MODELS[model_name](K).to(device)
    model.load_state_dict(torch.load(trained_model_path))
    for name, param in model.named_parameters():
        if name.startswith("first"):
            param.requires_grad=False
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Init results table
    results = pd.DataFrame()

    # Start training
    for e in range(nepochs):
        print(f"epoch: {e+1}")

        # Prep data
        dataloaders = get_dataloaders("classification", K, batch_size)

        running_loss = 0.0
        train_acc = 0
        for _, batch in enumerate(dataloaders['train']):
            X, Y = batch            
            X = X.to(device)
            Y = Y.to(device).long()
            out = model.classify(X)

            loss = loss_function(out, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            out = torch.argmax(out, dim=1).reshape(-1)
            train_acc += (out.round() == Y).sum()

        # Test Evalutaion
        test_acc = 0
        for X, Y in dataloaders['test']:
            X = X.to(device)
            Y = Y.to(device).long()

            out = torch.argmax(model.classify(X), dim=1).reshape(-1)
            test_acc += (out.round() == Y).sum()

        train_acc = train_acc / len(dataloaders['train'].dataset)
        test_acc = test_acc / len(dataloaders['test'].dataset)

        new_results = pd.DataFrame({"epoch": e, "train_acc": train_acc.item(), "test_acc": test_acc.item()}, index=[0])
        results = pd.concat([results, new_results])

        print(f"\tTrain acc = {train_acc:.4f}")
        print(f"\tTest acc  = {test_acc:.4f}")
        
    # save results
    name = f'classification_{model_name}_{K}_{lr}_{batch_size}_{nepochs}'

    results_path = "results/" + name + ".csv"
    results.to_csv(results_path, sep=",")

    # save model
    model_path = "models/" + name + ".pt"
    torch.save(model.state_dict(), model_path)
    
    print("DONE") 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument( "-model", "--model_name", default="twoLNN", type=str, help="model name")
    parser.add_argument( "-task", "--task_name", default="parity", type=str, help="task name")
    parser.add_argument( "-k", "--k_value", default=1, type=int, help="value of K")
    parser.add_argument( "-lr", "--lr_value", default=.01, type=float, help="value of learning rate")

    args = parser.parse_args()
    
    k = args.k_value
    lr = args.lr_value
    task_name = args.task_name
    model_name = args.model_name

    if task_name == "parity":
        train_parity(model_name=model_name, K=k, lr=lr)
    else:
        train_classification(model_name=model_name, K=k, lr=lr)