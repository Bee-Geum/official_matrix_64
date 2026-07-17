import torch

class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)

def Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
    return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

def test_Adam():
    results = {}

    # Test Case 1: Default parameters
    model1 = ModelNew().to('cuda')
    optimizer1 = Adam(model1.parameters())
    results["test_case_1"] = optimizer1.defaults

    # Test Case 2: Custom learning rate
    model2 = ModelNew().to('cuda')
    optimizer2 = Adam(model2.parameters(), lr=0.01)
    results["test_case_2"] = optimizer2.defaults

    # Test Case 3: Custom betas
    model3 = ModelNew().to('cuda')
    optimizer3 = Adam(model3.parameters(), betas=(0.85, 0.95))
    results["test_case_3"] = optimizer3.defaults

    # Test Case 4: Custom weight decay
    model4 = ModelNew().to('cuda')
    optimizer4 = Adam(model4.parameters(), weight_decay=0.01)
    results["test_case_4"] = optimizer4.defaults

    return results

test_results = test_Adam()
