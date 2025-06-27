import torch
import time
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference(linear_model, test_seq):
    linear_model.eval() 
    input_seq_tensor = torch.tensor(test_seq, dtype=torch.float32).to(device)
    # print(input_seq_tensor.shape)
    with torch.no_grad():
        pred_seq = linear_model(input_seq_tensor, None, None, None)  # Model inference
    pred_seq = pred_seq.cpu().numpy()
    return pred_seq

def autoregressive_inference(linear_model, input_seq, stride, test_len):
    pred_seq = []
    internal_input_seq = input_seq.copy()
    n_infer = test_len//stride
    print(test_len, stride, n_infer)
    for i in range(n_infer):
        out_seq = inference(linear_model, internal_input_seq)
        # print(internal_input_seq.shape, out_seq.shape)
        # (1, 96, 7), (96, 7)
        internal_input_seq = np.concatenate([internal_input_seq[:, stride:], out_seq[:, :stride]], axis=1)
        pred_seq.append(out_seq[:, :stride])

    pred_seq = np.concatenate(pred_seq, axis=1)
    return pred_seq


def train_and_inference(Model, train_seqs, test_seqs, seq_len, pred_len, stride, epochs, learning_rate, decom_len=25, train_ratio=0.80):    
    TT, C = test_seqs.shape
    window_dataset = np.lib.stride_tricks.sliding_window_view(train_seqs, (pred_len + pred_len, C))[:,0,...]
    # window_dataset = np.random.permutation(window_dataset)
    print(window_dataset.shape)

    N = len(window_dataset)

    train_len = int(N * train_ratio)
    train_window_dataset = window_dataset[:train_len]
    val_window_dataset = window_dataset[train_len:]
    print(train_window_dataset.shape, val_window_dataset.shape)

    train_window_dataset = torch.tensor(train_window_dataset, dtype=torch.float32)
    val_window_dataset = torch.tensor(val_window_dataset, dtype=torch.float32)

    train_dataset = TensorDataset(train_window_dataset)
    val_dataset = TensorDataset(val_window_dataset)

    # Create DataLoader
    batch_size = min([val_window_dataset.shape[0], 16])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    linear_model = Model(seq_len, pred_len, decom_len).to(device)
    criterion = torch.nn.MSELoss()  # Define loss function
    optimizer = torch.optim.Adam(linear_model.parameters(), lr=learning_rate)  # Define optimizer

    patience = 2 # 3
    best_loss = 10000
    for epoch in range(epochs):
        linear_model.train()  # Set model to training mode
        train_loss = 0.0

        for batch in train_dataloader:
            input_seq = batch[0][:, :seq_len].to(device)  # Move input sequence to GPU
            target_seq = batch[0][:, -pred_len:].to(device)  # Move target sequence to GPU

            optimizer.zero_grad()  # Initialize optimizer
            # print(input_seq.shape, target_seq.shape)
            pred_seq = linear_model(input_seq, None, None, None)  # Model inference
            # print(pred_seq.shape, target_seq.shape)
            # raise Exc-eption()
            loss = criterion(pred_seq, target_seq)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        # Validation loop
        linear_model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                input_seq = batch[0][:, :seq_len].to(device)  # Move input sequence to GPU
                target_seq = batch[0][:, -pred_len:].to(device)  # Move target sequence to GPU

                pred_seq = linear_model(input_seq, None, None, None)  # Model inference
                loss = criterion(pred_seq, target_seq)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if best_loss > val_loss:
            best_loss = val_loss
        else:
            patience = patience - 1
        
        if patience < 0:
            print(f'Early Stopping at {epoch+1}')
            break
        
    pred_seq = autoregressive_inference(linear_model, test_seqs[np.newaxis,:seq_len], stride=stride, test_len=len(test_seqs))

    return pred_seq