import deepspeed
from torch.utils.data import DataLoader, TensorDataset
from model import SimpleNet

import numpy as np


# Initialize model
model = SimpleNet()

# Parameters for DeepSpeed
params = list(model.parameters())
cmd_args = {
    'deepspeed_config': 'ds_config.json'
}

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                     model=model,
                                                     model_parameters=params)

# Dummy dataset
x = torch.from_numpy(np.random.randn(1000, 784)).float()
y = torch.from_numpy(np.random.randint(0, 10, (1000,))).long()
dataset = TensorDataset(x, y)
data_loader = DataLoader(dataset, batch_size=32)

# Training loop
for epoch in range(5):
    for batch in data_loader:
        inputs, labels = batch

        # Forward pass
        outputs = model_engine(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        # Backward pass and optimization
        model_engine.backward(loss)
        model_engine.step()
