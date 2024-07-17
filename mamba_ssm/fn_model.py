
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeModel(nn.Module):
    def __init__(self, model, input_dim, genlen, device, dtype):
        super(TimeModel, self).__init__()
        self.genlen = genlen
        self.input_dim = input_dim
        self.outputFN = nn.Sequential(
            nn.Linear(input_dim[1]*input_dim[2], genlen, bias=False, device=device, dtype=dtype),  
            #nn.Linear(genlen, 1, bias=False, device=device, dtype=dtype)  # Transform to [batch_size, 1]
        )
    
    def forward(self, embed_out):
        # Apply the output function
        output = self.outputFN(embed_out.view(self.input_dim[0],self.input_dim[1]*self.input_dim[2]))
        # Ensure the output has the shape [batch_size, genlen, 1]
        output = output.view(self.genlen) #since batch_size is 1, just get rid of it
        return output

'''

def fn(input_ids):
    embed_in = model.get_input_embeddings()(input_ids)
    timeOut = model(embed_in).last_hidden_state
    outputModel = TimeModel(input_dim=timeOut.shape, genlen=args.genlen, device=device, dtype=dtype)
    return outputModel(timeOut)
'''