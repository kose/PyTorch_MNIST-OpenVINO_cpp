## mode: python ##
## #coding: utf-8 ##

import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init

class NET(nn.Module):
    def __init__(self, ec_dim):
        super(NET, self).__init__()

        self.ln1 = nn.Linear(28 * 28, ec_dim)
        self.ln2 = nn.Linear(ec_dim, ec_dim)
        self.ln3 = nn.Linear(ec_dim, 10)

        self._initialize_weights()


    def _initialize_weights(self):
        nn.init.normal_(self.ln1.bias, 0.0, 1.0)
        nn.init.normal_(self.ln2.bias, 0.0, 1.0)
        nn.init.normal_(self.ln3.bias, 0.0, 1.0)

        
    def __call__(self, x):
        h = F.relu(self.ln1(x))
        h = F.relu(self.ln2(h))
        return self.ln3(h)


    def forward(self, x):
        h = F.relu(self.ln1(x))
        h = F.relu(self.ln2(h))
        h = self.ln3(h)
        return F.log_softmax(h, dim=1)


    def loss_function(self, x, target):

        output = self.forward(x)
        loss = F.nll_loss(output, target)
            
        return loss

# import pdb; pdb.set_trace()

### Local Variables: ###
### truncate-lines:t ###
### End: ###
