""" Model classes defined here! """

import torch
import torch.nn.functional as F

class FeedForward(torch.nn.Module):
    def __init__(self, hidden_dim):
        """
        In the constructor we instantiate two nn.Linear modules and 
        assign them as member variables.
        """
        super(FeedForward, self).__init__()
        self.linear1 = torch.nn.Linear(28*28, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, 8)

    def forward(self, x):
        """
        Compute the forward pass of our model, which outputs logits.
        """
        # TODO: Implement this!
        # raise NotImplementedError()
        x = self.linear1(x) #computes the dot product and adds bias
        x = F.relu(x)
        x = self.linear2(x) #computes dot product and adds bias
        #h2 = a2.exp()/a2.exp().sum(-1).unsqueeze(-1)

        return x

class SimpleConvNN(torch.nn.Module):
    def __init__(self, n1_chan, n1_kern, n2_kern):
        super(SimpleConvNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, n1_chan, kernel_size=n1_kern)
        self.conv2 = torch.nn.Conv2d(n1_chan, 8, kernel_size=n2_kern, stride=2)

    def forward(self, x):
        # TODO: Implement this!
        # raise NotImplementedError()
        x = x.view(-1,1,28,28)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        poolKernelsize = x.size()[2]
        x = F.max_pool2d(x,poolKernelsize)
        x = x.view(-1,8)
        return x

class BestNN(torch.nn.Module):
    # TODO: You can change the parameters to the init method if you need to
    # take hyperparameters from the command line args!
    def __init__(self, n1_chan, n2_chan, n1_kern, n2_kern, n3_kern):
        super(BestNN, self).__init__()
        # TODO: Implement this!
        #raise NotImplementedError()
        self.conv1 = torch.nn.Conv2d(1, n1_chan, kernel_size=n1_kern)
        self.conv2 = torch.nn.Conv2d(n1_chan, n2_chan, kernel_size=n2_kern)
        self.conv3 = torch.nn.Conv2d(n2_chan, 8, kernel_size=n3_kern, stride=2)

    def forward(self, x):
        # TODO: Implement this!
        #raise NotImplementedError()
        x = x.view(-1,1,28,28)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        poolKernelsize = x.size()[2]
        x = F.max_pool2d(x,poolKernelsize)
        x = x.view(-1,8)
        return x
