
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloaders
from twoLNN import twoLNN

if __name__ == "__main__":
    # parameters
    K = 1
    mode = "train"

    nepochs = 20
    steps = 500
    batch_size = 128

    # Prep data
    dataloaders = get_dataloaders(K, batch_size)

    # Load model
    model = twoLNN(K)
    loss_function = nn.BCELoss()
    optimizer = optim.Adadelta(model.parameters(), lr=.01)

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

        test_acc = list()
        for X, Y in dataloaders['test']:
            out = torch.sigmoid(model.forward(X)).reshape(-1)
            acc = (out.round() == Y).sum() / len(Y) 
            test_acc.append(acc)

        print(f"TEST ACC = {torch.mean(torch.tensor(test_acc))}")
        
    print("DONE") 

            

    
    

    