
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloaders
import pandas as pd
from twoLNN import twoLNN

MODELS = {
    "twoLNN": twoLNN
}

def train(model_name, task_name, K, batch_size=128, nepochs=20):
    # Prep data
    dataloaders = get_dataloaders(K, batch_size)

    # Load model
    model = MODELS[model_name](K)
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
        train_acc = list()
        for X, Y in dataloaders['train']:
            out = torch.sigmoid(model.forward(X)).reshape(-1)
            acc = (out.round() == Y).sum() / len(Y) 
            train_acc.append(acc)

        # Test Evalutaion
        test_acc = list()
        for X, Y in dataloaders['test']:
            out = torch.sigmoid(model.forward(X)).reshape(-1)
            acc = (out.round() == Y).sum() / len(Y) 
            test_acc.append(acc)

        mean_train_acc = torch.mean(torch.tensor(train_acc)).item()
        mean_test_acc = torch.mean(torch.tensor(test_acc)).item()

        new_results = pd.DataFrame({"epoch": e, "train_acc": mean_train_acc, "test_acc": mean_test_acc}, index=[0])
        results = pd.concat([results, new_results])

        print(f"TRAIN ACC = {mean_train_acc:.4f}")
        print(f"TEST ACC  = {mean_test_acc:.4f}")
        
    results_name = f'{task_name}_{model_name}_{K}_{batch_size}.csv'
    results_path = "results/" + results_name
    results.to_csv(results_path, sep=",")
    
    print("DONE") 

if __name__ == "__main__":

    Ks = [1,3]

    for k in Ks:
        train(model_name="twoLNN", task_name="parity", K=k)