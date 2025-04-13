import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from dataset import get_dataset, TransformedImagesDataset
from architecture import MyCNN

def get_device():

    device = torch.device("mps")
    print(f"Using device: {device}")
    return device

def training_loop(
        model: nn.Module,
        train_data: torch.utils.data.Dataset,
        eval_data: torch.utils.data.Dataset,
        num_epochs: int,
        batch_size: int,
        learning_rate: float = 1e-3,
        patience: int = 10,
        show_progress: bool = False
):

    device = get_device()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)

    train_losses = []
    eval_losses = []
    train_accuracies = []
    eval_accuracies = []
    best_eval_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        network.train()
        epoch_train_loss = 0.0
        correct_train_predictions = 0
        total_train_predictions = 0

        if show_progress:
            train_loader = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

        for inputs, targets, _, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train_predictions += (predicted == targets).sum().item()
            total_train_predictions += targets.size(0)

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracy = correct_train_predictions / total_train_predictions
        train_accuracies.append(train_accuracy)

        network.eval()
        epoch_eval_loss = 0.0
        correct_eval_predictions = 0
        total_eval_predictions = 0

        with torch.no_grad():
            for inputs, targets, _, _ in eval_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = network(inputs)
                loss = criterion(outputs, targets)
                epoch_eval_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_eval_predictions += (predicted == targets).sum().item()
                total_eval_predictions += targets.size(0)

        avg_eval_loss = epoch_eval_loss / len(eval_loader)
        eval_losses.append(avg_eval_loss)
        eval_accuracy = correct_eval_predictions / total_eval_predictions
        eval_accuracies.append(eval_accuracy)

        scheduler.step(avg_eval_loss)

        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            torch.save(network.state_dict(), "model.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

        print(f"Epoch: {epoch + 1} --- Train loss: {avg_train_loss:7.4f} --- Train accuracy: {train_accuracy:7.4f} --- Eval loss: {avg_eval_loss:7.4f} --- Eval accuracy: {eval_accuracy:7.4f}")

    return train_losses, eval_losses, train_accuracies, eval_accuracies


if __name__ == "__main__":
    torch.random.manual_seed(1234)
    image_dir = '/Users/yehor_larcenko/Desktop/training_data'
    train_data, eval_data = get_dataset(image_dir=image_dir)
    transformed_train_data = TransformedImagesDataset(train_data)
    network = MyCNN(num_classes=20)
    train_losses, eval_losses, train_accuracies, eval_accuracies = training_loop(
        network, transformed_train_data, eval_data, num_epochs=30, batch_size=64, learning_rate=0.001, show_progress=True)
    for epoch, (tl, el) in enumerate(zip(train_losses, eval_losses)):
        print(f"Epoch: {epoch + 1} --- Train loss: {tl:7.4f} --- Eval loss: {el:7.4f}")