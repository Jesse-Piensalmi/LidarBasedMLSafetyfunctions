import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np

class GlobalStats:
    def __init__(self):
        self.min_x = float('inf')
        self.max_x = float('-inf')
        self.min_y = float('inf')
        self.max_y = float('-inf')

global_stats = GlobalStats()


# Define the neural network
class SafetyNN(nn.Module):
    def __init__(self, input_dim=75, hidden_dim=64, output_dim=4):
        super(SafetyNN, self).__init__()
        
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

        # Separate heads for `h` and gradients
        self.fc_h = nn.Sequential(
            nn.Linear((hidden_dim * 2)+1, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1)  # Single output for `h`
        )
        self.fc_gradients = nn.Sequential(
            nn.Linear((hidden_dim * 2)+1, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 3)  # Outputs for `dh/dx`, `dh/dy`, `dh/dz`
        )

    def forward(self, x, alpha):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)  # Flatten feature maps to [batch_size, hidden_dim * 2]
        x = torch.cat([x, alpha], dim=1)
        # Separate predictions
        h = self.fc_h(x)
        gradients = self.fc_gradients(x)
        return torch.cat([h, gradients], dim=1)  # Combine `h` and gradients

# Process data into fixed-size vectors
def process_data(pos_x, pos_y, edge_detections, max_edges=36):
    """
    Create a vector of size 37:
    - First element: robot position (combined pos_x and pos_y as scalar)
    - Next 36 elements: edge detections (padded with inf if fewer than 36 points)
    """
    # Combine robot positions into a single value (optional: use [pos_x, pos_y] if both are needed)
    robot_position = np.array([0.0,0.0])

    # Shuffle edge detections
    if edge_detections is not None:
        localized_edges = edge_detections - np.array([pos_x,pos_y])
        global_stats.min_x = min(global_stats.min_x, np.min(localized_edges[:, 0]))
        global_stats.max_x = max(global_stats.max_x, np.max(localized_edges[:, 0]))
        global_stats.min_y = min(global_stats.min_y, np.min(localized_edges[:, 1]))
        global_stats.max_y = max(global_stats.max_y, np.max(localized_edges[:, 1]))


        padding = np.array([[1, 1]] * (max_edges - len(edge_detections)))

        # Stack the original detections and the padding
        padded_edges = np.vstack([localized_edges, padding])
        #random.shuffle(padded_edges)
    else:
        padded_edges = np.array([[1, 1]]* (max_edges))

    # Trim to max_edges in case there are more than max_edges
    padded_edges = padded_edges[:max_edges]

    # Combine robot position and padded edges into a single vector
    return padded_edges

def process_targets(targets):
    """
    Process targets into a uniform structure suitable for PyTorch tensors.
    """
    processed_targets = []
    for target in targets:
        # Ensure target[0] is a float
        value = float(target[0])  # Convert np.int64 or np.float64 to float
        # Flatten the array and convert to list
        array_values = target[1].flatten().tolist()
        # Combine into a single list
        processed_targets.append([value] + array_values)
    
    # Convert to a NumPy array or PyTorch tensor
    return torch.tensor(processed_targets, dtype=torch.float32)

# Load datasets from pickle files in a folder
def load_datasets_from_folder(folder_path, max_edges=36):
    inputs = []
    alphas = []
    targets = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)  # Load the dictionary
                stored_data = data['stored_data']

                # Extract relevant fields
                pos_x = stored_data['pos_x_0']
                pos_y = stored_data['pos_y_0']
                edge_detections = stored_data['data_X_0']
                safety_values = stored_data['h_gp_0']
                dh_values = stored_data['dh_dt0']
                if '0' in filename:
                    index = filename.find('0')
                    alpha = float('0.'+filename[index+1:index+3])
                else:
                    alpha = 0.15
                # Process and combine data
                for px, py, edges, safety, dh in zip(pos_x, pos_y, edge_detections, safety_values, dh_values):
                        if dh is not None:
                            input_vector = process_data(px, py, edges, max_edges)
                            #flat = np.array(input_vector.reshape(-1))
                            #flat = np.insert(flat,2,alpha)
                            inputs.append(input_vector)
                            alphas.append([alpha])
                            dh_localized = dh-np.array([px,py,0.0])
                            targets.append([safety, dh_localized])
                        else:
                            if safety is not None:
                                input_vector = process_data(px, py, edges, max_edges)
                                #flat = np.array(input_vector.reshape(-1))
                                #flat = np.insert(flat,2,alpha)
                                #inputs.append(input_vector)
                                #alphas.append([alpha])
                                #targets.append([safety, np.array([[0.0,0.0,0.0]])])
    # Flatten inputs
    #flattened = np.array([input.reshape(-1) for input in inputs])
    #flattened = np.insert(flattened,2,alpha)
    # Normalize inputs 
    normalized_data = (inputs - np.array([-0.5, -0.5])) / (np.array([0.5 - (-0.5), 0.5 - (-0.5)]))
    # Convert to PyTorch tensors
    flattened = np.array([input.reshape(-1) for input in normalized_data])
    inputs = torch.tensor(flattened, dtype=torch.float32)
    targets = process_targets(targets)
    alphas = torch.tensor(alphas)

    return inputs, targets,torch.tensor(alphas)

# Prepare the dataset
def prepare_dataset(inputs, alphas, targets):
    dataset = TensorDataset(inputs, alphas,targets)
    return dataset

# Initialize weights
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Validation loop
def validate(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # Disable gradient computation for validation
        for inputs, alphas, targets in dataloader:
            #inputs = inputs.unsqueeze(-1)
            #inputs = torch.permute(inputs, (0, 2, 1))
            alpha = alphas 
            edge_points = inputs  # Remove the third value (shape: [batch_size, 74])
            edge_points = edge_points.unsqueeze(1)  # Add channel dimension for Conv1D (shape: [batch_size, 1, 74])
            inputs = inputs.unsqueeze(-1)
            
            inputs = torch.permute(inputs,(0,2,1))
            outputs = model(edge_points,alpha)

            # Create a mask for valid values in the target
            mask = ~torch.isnan(targets)
            mask = mask.float()

            # Compute validation loss
            loss = masked_mse_loss(outputs, targets, mask)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def masked_mse_loss(output, target, mask):
    # Apply the mask to ignore padded output during loss calculation
    masked_output = output[np.bool(mask)]
    masked_target = target[np.bool(mask)]
    return torch.mean((masked_output - masked_target) ** 2)

def mask_gradients(model, mask):
    for param in model.parameters():
        if param.grad is not None:
            param.grad *= mask

def composite_loss(output, target):
    h_loss = nn.MSELoss()(output[:, 0], target[:, 0])  # `h` loss
    grad_loss = nn.MSELoss()(output[:, 1:], target[:, 1:])  # Gradient loss
    return h_loss + 1.0 * grad_loss  # Adjust the weight as needed

# Training loop
def train_safety_nn(model, dataloader, criterion, optimizer, num_epochs, val_dataloader, scheduler):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, alphas,targets in dataloader:
            optimizer.zero_grad()
            alpha = alphas 
            edge_points = inputs  # Remove the third value (shape: [batch_size, 74])
            edge_points = edge_points.unsqueeze(1)  # Add channel dimension for Conv1D (shape: [batch_size, 1, 74])
            inputs = inputs.unsqueeze(-1)
            
            inputs = torch.permute(inputs,(0,2,1))
            outputs = model(edge_points,alpha)
            # Mask the outputs for padding values (outputs)
            output_mask = ~torch.isnan(outputs).any(dim=-1)  # Mask for NaN values in output
            target_mask = ~torch.isnan(targets).any(dim=-1)  # Mask for NaN values in target

            # Combine the output and target masks (only consider non-NaN entries in both)
            mask = output_mask & target_mask

            # Expand mask to match the output's shape (batch_size, num_features)
            mask = mask.unsqueeze(-1).expand_as(outputs)  # Shape [32, 4] for batch size 32 and 4 features

            # Compute the loss with the mask
            loss = composite_loss(outputs, targets)
            
            if ~torch.isnan(loss):
                loss.backward()

                # Mask gradients
                #mask_gradients(model, mask)

                optimizer.step()
                total_loss += loss.item()
        scheduler.step()
        if epoch%10==0:
            model_path = 'safety_nn_model_full_norm_complicated'+str(epoch)+'.pth'
            torch.save(model.state_dict(),model_path)    
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {validate(model,val_dataloader,criterion=criterion):.4f}")

# Main script
def main():
    # Folder path containing pickle datasets
    train_data_path = 'control_lib/SimData/SimData'
    val_data_path = 'control_lib/SimData/Val_data'

    # Load data from all pickle files in the folder
    inputs, targets, alphas = load_datasets_from_folder(train_data_path)
    print("Global max x: "+str(global_stats.max_x))
    print("Global min x: "+str(global_stats.min_x))
    print("Global max y: "+str(global_stats.max_y))
    print("Global min y: "+str(global_stats.min_y))
    val_inputs,val_targets, alphas_val = load_datasets_from_folder(val_data_path)
    # Prepare dataset and dataloader
    dataset = prepare_dataset(inputs, alphas, targets)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_dataset = prepare_dataset(val_inputs, alphas_val, val_targets)
    val_dataloader = DataLoader(val_dataset,batch_size=32,shuffle=True)
    # Define model, loss function, and optimizer
    input_dim = 72  # Fixed-size input vector
    hidden_dim = 64
    output_dim = 4  # Output for [safety_value, dh]

    model = SafetyNN(input_dim, hidden_dim, output_dim)
    initialize_weights(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Train the model
    num_epochs = 50
    train_safety_nn(model, dataloader, criterion, optimizer, num_epochs, val_dataloader, scheduler)

    # Save the trained model
    torch.save(model.state_dict(), 'safety_complicated_nn_model.pth')
    print("Model training complete and saved.")
    print("Global max x: "+str(global_stats.max_x))
    print("Global min x: "+str(global_stats.min_x))
    print("Global max y: "+str(global_stats.max_y))
    print("Global min y: "+str(global_stats.min_y))

if __name__ == "__main__":
    main()