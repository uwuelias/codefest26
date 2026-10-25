import copy
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CACHE = Path("cache")
WEIGHTS = Path("weights")
SEED = 42
EPOCHS = 100
BATCH_SIZE = 32
LR = 0.001
PATIENCE = 5

# Load embeddings from build.py
def load_data(section):
    X = np.load(CACHE / f"X_{section}.npy", allow_pickle=False)
    y = np.load(CACHE / f"y_{section}.npy", allow_pickle=False)
    #set allow_pickle to False for security reasons

    #Check for Proper format
    if X.ndim != 2 or X.shape[1] != 768:
        raise ValueError(
            F"{section}: Incorrect shape. Expected (N, 768), got {X.shape}"
        )
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError(
            f"{section}: invalid label shape"          
        )
    if not np.isfinite(X).all():
        raise ValueError(
            f"{section}: contains non-finite embeddings"
        )
    if not np.isin(y, [0,1]).all():
        raise ValueError(
            f"{section}: invalid label values"
        )
    if len(np.unique(y)) != 2:
        raise ValueError(
            f"{section}: labels must contain both 0 and 1"
        )

    return X.astype(np.float32), y.astype(np.float32)

#Logistic Regression 
def make_model():
    #spits out logit
    return nn.Linear(768, 1)

def evaluate(model, X, y, loss_fn):
    model.eval()
    with torch.inference_mode():
        logits = model(X).squeeze(1)
        loss = loss_fn(logits, y).item()

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()
        accuracy = (
            (predictions ==y)
            .float()
            .mean()
            .item()
        )

        real_mask = (y == 0)
        fake_mask = (y == 1)

        real_recall = (
            (predictions[real_mask] == 0)
            .float()
            .mean().item()
        )

        fake_recall = (
            (predictions[fake_mask] == 1)
            .float()
            .mean().item()
        )

        balanced_accuracy = (
            real_recall + fake_recall
        ) / 2

        return loss, accuracy, balanced_accuracy

#training
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    X_train, y_train = load_data("training")
    X_val, y_val = load_data("validation")
    X_test, y_test = load_data("testing")

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)

    # Standardize the features using z = (x - mean) / std
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convert numpy arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # initialize logistic regression
    model = make_model()

    # Turns logit to probability using sigmoid function
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay = 1e-2)

    best_loss = float('inf')
    best_weights = None
    stale_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            logits = model(X_batch).squeeze(1)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += (loss.item() * len(X_batch))
        train_loss = (total_loss / len(train_dataset))
        val_loss, val_accuracy, val_balanced_accuracy = evaluate(model, X_val, y_val, loss_fn)

        print(
            f"Epoch {epoch}/{EPOCHS}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.2%}, "
            f"Val Balanced Accuracy: {val_balanced_accuracy:.2%}"
        )

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print("\nEarly stop.")
                break

    if best_weights is None:
        raise RuntimeError("no valid checkpoint.")
    model.load_state_dict(best_weights)

    test_loss, test_accuracy, test_balanced_accuracy = evaluate(
    model,
    X_test,
    y_test,
    loss_fn
    )

    print("\nFinal test results:")
    print("--------------------------")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"Test Balanced Accuracy: {test_balanced_accuracy:.2%}")

    # save model and scalar
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS / "classifier.pt")
    joblib.dump(scaler, WEIGHTS / "scaler.joblib")

    print("training done")

if __name__ == "__main__":
    main()
