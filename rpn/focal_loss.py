import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Focal loss for training
class FocalLoss(nn.Module):
  def __init__(self, alpha=0.05, gamma=0, device = device):
      super(FocalLoss, self).__init__()
      self.weight = torch.FloatTensor([alpha, 1-alpha]).to(device)
      self.nllLoss = F.nll_loss
      self.gamma = gamma

  def forward(self, input, target):
      softmax = F.softmax(input, dim=1)
      log_logits = torch.log(softmax)
      fix_weights = (1 - softmax) ** self.gamma
      logits = fix_weights * log_logits
      return self.nllLoss(logits, target,weight = self.weight,ignore_index=-1)