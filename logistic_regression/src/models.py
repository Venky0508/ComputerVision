from torch import nn


class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super(LogisticRegression, self).__init__()
        # raise NotImplementedError("Your code here. Hint: 1 line in the answer key.")
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # raise NotImplementedError("Your code here. Hint: 1-2 lines in the answer key.")
        # return self.linear(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)
