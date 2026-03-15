import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    """
    The standard simple regression model for m-height estimation.
    """
    def __init__(self, n, k):
        super(SimpleMLP, self).__init__()
        # Calculate input size directly inside the model
        input_size = k * (n - k)
        
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class ColumnCNN(nn.Module):
    """
    A CNN that sweeps across the columns of the P matrix.
    Treats the k rows as 'input channels' and the n-k columns as the 'spatial/sequence' dimension.
    """
    def __init__(self, n, k):
        super(ColumnCNN, self).__init__()
        self.n = n
        self.k = k
        self.cols = n - k
        
        # 1D Convolution across the columns
        # in_channels = k (because each column has k row elements)
        # out_channels = feature maps learned by the kernel
        self.conv1 = nn.Conv1d(in_channels=k, out_channels=16, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, padding=1)
        
        # Because the number of columns (n-k) changes based on the input parameters (ranging from 3 to 5),
        # an AdaptiveAvgPool1d forces the output to a fixed size (e.g., 2) before hitting the linear layers.
        self.pool = nn.AdaptiveAvgPool1d(2)
        
        # 32 channels * 2 pooled features = 64 input features for the fully connected layer
        self.fc1 = nn.Linear(32 * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        # x arrives flattened with shape (batch_size, k * (n-k))
        # Reshape to (batch_size, channels=k, sequence_length=cols)
        x = x.view(-1, self.k, self.cols)
        
        # Pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Pool and flatten for the dense layers
        x = self.pool(x)
        x = x.view(x.size(0), -1) 
        
        # Pass through linear layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Remember: Your MHeightModelWrapper will automatically apply the 1.0 + F.relu() 
        # to this raw output to guarantee it satisfies the >= 1 constraint.
        return self.fc3(x)
    
class TheoryMLP(nn.Module):
    """
    An MLP that takes BOTH the flattened P matrix and the integer m as inputs.
    """
    def __init__(self, n, k):
        super(TheoryMLP, self).__init__()
        # Input size: elements in P matrix + 1 feature for 'm'
        input_size = (k * (n - k)) + 1
        
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x, m_val):
        # Ensure m_val is a column vector with the same batch size as x
        if m_val.dim() == 1:
            m_val = m_val.unsqueeze(1)
            
        # Concatenate the flattened matrix with m
        combined = torch.cat((x, m_val), dim=1)
        
        out = F.relu(self.fc1(combined))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        return self.fc4(out)
    
class TheoryColumnCNN(nn.Module):
    """
    Combines the Column-wise CNN with the integer 'm' input 
    to allow for theory-informed mathematical constraints.
    """
    def __init__(self, n, k):
        super(TheoryColumnCNN, self).__init__()
        self.n = n
        self.k = k
        self.cols = n - k
        
        # Convolution across the n-k columns
        self.conv1 = nn.Conv1d(in_channels=k, out_channels=16, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(2)
        
        # 32 channels * 2 pooled features = 64 CNN features
        # + 1 feature for the 'm' parameter = 65 total input features for the dense layer
        self.fc1 = nn.Linear(64 + 1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x, m_val):
        # 1. Process the P matrix through the CNN
        cnn_x = x.view(-1, self.k, self.cols)
        cnn_x = F.relu(self.conv1(cnn_x))
        cnn_x = F.relu(self.conv2(cnn_x))
        cnn_x = self.pool(cnn_x)
        cnn_x = cnn_x.view(cnn_x.size(0), -1) # Flatten to [batch_size, 64]
        
        # 2. Append the 'm' value to the CNN features
        if m_val.dim() == 1:
            m_val = m_val.unsqueeze(1)
        combined = torch.cat((cnn_x, m_val), dim=1)
        
        # 3. Process through fully connected layers
        out = F.relu(self.fc1(combined))
        out = F.relu(self.fc2(out))
        return self.fc3(out)

class ColumnTransformer(nn.Module):
    """
    Treats the columns of the P matrix as a sequence of continuous tokens.
    Uses Self-Attention to find relationships between columns, completely 
    ignoring their spatial order (Permutation Invariant).
    """
    def __init__(self, n, k):
        super(ColumnTransformer, self).__init__()
        self.n = n
        self.k = k
        self.cols = n - k
        self.d_model = 64
        
        # 1. Project the k-dimensional column vector into the Transformer's hidden dimension
        self.input_projection = nn.Linear(k, self.d_model)
        
        # 2. Bidirectional Transformer Encoder Layer
        # nhead=4 means 4 attention heads looking at different column relationships
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=4, 
            dim_feedforward=128, 
            batch_first=True,
            activation='relu'
        )
        # Stack 2 layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 3. Final Regression Head
        self.fc1 = nn.Linear(self.d_model, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x arrives flattened: shape (batch_size, k * cols)
        # Reshape to treat columns as sequence items: (batch_size, sequence_length=cols, features=k)
        # Note: We transpose so columns are the tokens.
        x = x.view(-1, self.k, self.cols).transpose(1, 2)
        
        # Project tokens to d_model
        x = self.input_projection(x)
        
        # Pass through Transformer (Self-Attention)
        # Shape remains (batch_size, cols, d_model)
        x = self.transformer(x)
        
        # Global Average Pooling across the sequence dimension (the columns)
        # This makes the entire architecture permutation invariant!
        x = x.mean(dim=1) 
        
        # Final fully connected layers
        x = F.relu(self.fc1(x))
        return self.fc2(x)