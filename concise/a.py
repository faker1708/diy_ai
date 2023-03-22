import torch
from torch import nn


net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256).cuda(),
                    nn.ReLU(),
                    nn.Linear(256, 10).cuda()
                    
                    )

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)


batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

