import torch

CE = torch.nn.CrossEntropyLoss()

print(CE(torch.tensor([.5, 0.2, 0.3]), torch.tensor([0.5, 0.2, 0.3])))

