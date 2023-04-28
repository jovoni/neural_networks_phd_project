
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloaders
import pandas as pd
from twoLNN import twoLNN
from utils import use_gpu_if_possible
import argparse

MODELS = {
    "twoLNN": twoLNN
}

def train(model_name, task_name, K, batch_size=128, nepochs=20):
    # Prep data
    dataloaders = get_dataloaders(K, batch_size)

    # get device
    device = use_gpu_if_possible()

    print(f"Using device : {device}")

    # Load model
    model = MODELS[model_name](K).to(device)
    loss_function = nn.BCELoss()
    optimizer = optim.Adadelta(model.parameters(), lr=.01)

    # Init results table
    results = pd.DataFrame()

    # Start training
    for e in range(nepochs):
        running_loss = 0.0
        running_acc = 0.0
        for step, batch in enumerate(dataloaders['train']):
            X, Y = batch
            
            X = X.to(device)
            Y = Y.to(device)

            out = torch.sigmoid(model.forward(X)).reshape(-1)

            loss = loss_function(out, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += (out.round() == Y).sum() / len(Y)

            if step % 20 == 19: 
                print(f'{e + 1}, {step + 1}, loss = {running_loss / 20}, acc = {running_acc / 20}')
                running_acc = 0.0
                running_loss = 0.0

        # Train evaluation
        train_acc = 0
        for X, Y in dataloaders['train']:
            X = X.to(device)
            Y = Y.to(device)

            out = torch.sigmoid(model.forward(X)).reshape(-1)
            train_acc += (out.round() == Y).sum()

        # Test Evalutaion
        test_acc = 0
        for X, Y in dataloaders['test']:
            X = X.to(device)
            Y = Y.to(device)

            out = torch.sigmoid(model.forward(X)).reshape(-1)
            test_acc += (out.round() == Y).sum()

        test_acc = test_acc / len(dataloaders['test'])
        train_acc = train_acc / len(dataloaders['train'])

        new_results = pd.DataFrame({"epoch": e, "train_acc": train_acc, "test_acc": test_acc}, index=[0])
        results = pd.concat([results, new_results])

        print(f"TRAIN ACC = {train_acc:.4f}")
        print(f"TEST ACC  = {test_acc:.4f}")
        
    results_name = f'{task_name}_{model_name}_{K}_{batch_size}.csv'
    results_path = "results/" + results_name
    results.to_csv(results_path, sep=",")
    
    print("DONE") 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument( "-model", "--model_name", default="twoLNN", type=str, help="model name")
    parser.add_argument( "-task", "--task_name", default="parity", type=str, help="task name")
    parser.add_argument( "-k", "--k_value", default=1, type=int, help="value of K")

    args = parser.parse_args()
    
    k = args.k_value
    task_name = args.task_name
    model_name = args.model_name

    train(model_name=model_name, task_name=task_name, K=k)