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
        # self.pool = nn.AdaptiveAvgPool1d(2)
        self.pool = nn.AdaptiveMaxPool1d(2)
        
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
        # self.pool = nn.AdaptiveAvgPool1d(2)
        self.pool = nn.AdaptiveMaxPool1d(2)
        
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

class Conv1dResBlock(nn.Module):
    """
    A standard 1D Residual Block. 
    Keeps channel count and sequence length identical so the skip connection can be added.
    """
    def __init__(self, channels):
        super(Conv1dResBlock, self).__init__()
        # kernel_size=3 and padding=1 ensures the sequence length (number of columns) doesn't shrink
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        # The magic happens here: add the original input before the final activation
        out += residual
        return F.relu(out)

class ResColumnCNN(nn.Module):
    """
    Upgrades the ColumnCNN by integrating deep Residual Blocks.
    Excellent for learning highly non-linear functions without vanishing gradients.
    """
    def __init__(self, n, k):
        super(ResColumnCNN, self).__init__()
        self.n = n
        self.k = k
        self.cols = n - k

        self.width = 512
        
        # 1. Initial projection layer
        # Projects the varying 'k' input channels into a stable 32 feature channels
        self.init_conv = nn.Conv1d(in_channels=k, out_channels=self.width, kernel_size=3, padding=1)
        
        # 2. Residual Blocks
        # We can stack these safely because the skip connections prevent gradient loss
        self.res_block1 = Conv1dResBlock(self.width)
        self.res_block2 = Conv1dResBlock(self.width)
        self.res_block3 = Conv1dResBlock(self.width)
        
        # 3. Pooling and FC Layers (same as the original ColumnCNN)
        self.pool = nn.AdaptiveAvgPool1d(2)
        
        # 32 channels * 2 pooled features = 64 input features
        self.fc1 = nn.Linear(self.width * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        # Reshape flattened input to (batch_size, channels=k, sequence_length=cols)
        x = x.view(-1, self.k, self.cols)
        
        # Initial projection
        x = F.relu(self.init_conv(x))
        
        # Pass through the ResBlocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # Pool to a fixed size and flatten
        x = self.pool(x)
        x = x.view(x.size(0), -1) 
        
        # Fully connected regression head
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return self.fc3(x)

class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        # Skip connection: Add original x to the output
        residual = x
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return F.relu(out + residual)

class ResMLP(nn.Module):
    def __init__(self, n, k):
        super(ResMLP, self).__init__()
        input_size = k * (n - k)
        hidden_dim = 128
        
        self.input_layer = nn.Linear(input_size, hidden_dim)
        # Stack 3 Residual Blocks
        self.res_blocks = nn.Sequential(
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim)
        )
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.res_blocks(x)
        return self.fc_out(x)

class TannerGNN(nn.Module):
    """
    Treats the matrix P as an adjacency matrix for a bipartite graph.
    Passes messages between row nodes and column nodes.
    """
    def __init__(self, n, k, embed_dim=32):
        super(TannerGNN, self).__init__()
        self.k = k
        self.cols = n - k
        
        # Linear layers to process the messages
        self.row_to_col = nn.Linear(k, embed_dim)
        self.col_to_row = nn.Linear(self.cols, embed_dim)
        
        # Final regression head
        self.fc1 = nn.Linear(embed_dim * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch_size, k * cols)
        P = x.view(-1, self.k, self.cols)
        
        # Step 1: Column nodes receive messages from Row nodes (P^T * rows)
        # Since we don't have explicit node features, we process the P matrix itself
        # P transposed is (batch_size, cols, k)
        col_messages = F.relu(self.row_to_col(P.transpose(1, 2)))
        
        # Step 2: Row nodes receive messages from Column nodes
        row_messages = F.relu(self.col_to_row(P))
        
        # Step 3: Global pooling (mean over the nodes)
        col_pooled = col_messages.mean(dim=1) # (batch_size, embed_dim)
        row_pooled = row_messages.mean(dim=1) # (batch_size, embed_dim)
        
        # Combine the graph representations
        graph_embed = torch.cat([row_pooled, col_pooled], dim=1)
        
        out = F.relu(self.fc1(graph_embed))
        return self.fc2(out)
    
class NeuralSortNet(nn.Module):
    """
    Projects the matrix into simulated codeword features, physically sorts them 
    in descending order, and learns the m-height from the sorted distribution.
    """
    def __init__(self, n, k, num_features=16):
        super(NeuralSortNet, self).__init__()
        self.k = k
        self.cols = n - k
        input_size = k * (n - k)
        self.num_features = num_features
        
        # Project matrix into a set of values (simulating a codeword)
        self.projector = nn.Linear(input_size, num_features)
        
        # The MLP now only looks at strictly ordered, sorted data
        self.fc1 = nn.Linear(num_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        # Generate simulated codeword features
        raw_features = self.projector(x)
        
        # The algorithmic alignment step: Sort the features!
        # torch.sort()[0] returns the values, which maintain gradient history
        sorted_features, _ = torch.sort(raw_features, dim=1, descending=True)
        
        # Predict based on the sorted distribution
        out = F.relu(self.fc1(sorted_features))
        out = F.relu(self.fc2(out))
        return self.fc3(out)
    
class UnrolledDEQ(nn.Module):
    """
    Approximates an Implicit layer by running a recurrent unrolled loop.
    Simulates finding a fixed-point "worst-case" equilibrium before predicting.
    """
    def __init__(self, n, k, hidden_dim=64, iterations=5):
        super(UnrolledDEQ, self).__init__()
        input_size = k * (n - k)
        self.iterations = iterations
        
        # Feature extractor
        self.embed_x = nn.Linear(input_size, hidden_dim)
        
        # The "Equilibrium" layer (Weight-tied recurrent step)
        self.W_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_x = nn.Linear(hidden_dim, hidden_dim)
        
        # Readout
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Initial projection
        x_embed = self.embed_x(x)
        
        # Initialize hidden state z at zero
        z = torch.zeros_like(x_embed)
        
        # Unroll the fixed-point iteration loop
        # z_{t+1} = ReLU(W_z * z_t + W_x * x)
        for _ in range(self.iterations):
            z = F.relu(self.W_z(z) + self.W_x(x_embed))
            
        # Predict from the final equilibrium state
        return self.fc_out(z)

class MathFeatureMLP(nn.Module):
    """
    Calculates explicit linear algebra features from P on the fly, 
    concatenates them, and feeds them into a wider MLP.
    """
    def __init__(self, n, k):
        super(MathFeatureMLP, self).__init__()
        self.n = n
        self.k = k
        self.cols = n - k
        
        # Calculate the exact number of features we are generating
        p_elements = k * self.cols
        gram_col_elements = self.cols * self.cols  # P^T P
        gram_row_elements = k * k                  # P P^T
        svd_elements = min(k, self.cols)           # Singular values
        
        # The new massive input dimension
        total_input_size = p_elements + gram_col_elements + gram_row_elements + svd_elements
        
        self.fc1 = nn.Linear(total_input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        # 1. Reconstruct the P matrix
        P = x.view(-1, self.k, self.cols)
        
        # 2. Calculate Gram Matrices (Linear independence and angles)
        # P^T * P (Column relationships)
        P_t = P.transpose(1, 2)
        gram_cols = torch.bmm(P_t, P).view(x.size(0), -1) 
        
        # P * P^T (Row relationships)
        gram_rows = torch.bmm(P, P_t).view(x.size(0), -1)
        
        # 3. Calculate Singular Values (The absolute stretching limits of the matrix)
        # We use a try/except block because SVD can occasionally fail to converge 
        # on perfectly singular, highly degenerate matrices during backprop.
        try:
            svd_vals = torch.linalg.svdvals(P)
        except torch._C._LinAlgError:
            # Fallback to zeros if SVD fails for a specific bizarre batch
            svd_vals = torch.zeros(x.size(0), min(self.k, self.cols), device=x.device)
            
        # 4. Concatenate all mathematical features together
        enhanced_features = torch.cat([x, gram_cols, gram_rows, svd_vals], dim=1)
        
        # 5. Pass the math-enriched data through the network
        out = F.relu(self.fc1(enhanced_features))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        return self.fc4(out)