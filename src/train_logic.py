import torch


def train_loop(model, train_loader, optimizer, criterion, device):
    """Core training loop framework"""
    model.train()
    total_loss = 0.0
    for traffic, image, target in train_loader:
        traffic, image, target = traffic.to(device), image.to(device), target.to(device)
        pred = model(traffic, image)
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def eval_loop(model, test_loader, criterion, device):
    """Core evaluation loop framework"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for traffic, image, target in test_loader:
            traffic, image, target = traffic.to(device), image.to(device), target.to(device)
            pred = model(traffic, image)
            total_loss += criterion(pred, target).item()
    return total_loss / len(test_loader)