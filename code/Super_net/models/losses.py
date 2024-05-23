import torch
import torch.nn as nn


class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification.
    Refer to https://github.com/JiahuiYu/slimmable_networks/blob/master/utils/loss_ops.py
    """
    def forward(self, output, target, temp=1):
        output_log_prob = torch.nn.functional.log_softmax(output/temp, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob) 
        cross_entropy_loss = cross_entropy_loss.mean()
        return cross_entropy_loss
    
class KLDivLossSoft(torch.nn.modules.loss._Loss):
    """The kullback-lebiler divergence loss measure
    """
    def forward(self, output, target, temp=1):
        output_log_prob = torch.nn.functional.log_softmax(output/temp, dim=1)
        kl_loss = nn.KLDivLoss(reduction="batchmean") 
        soft_loss = kl_loss(output_log_prob, target)
        return soft_loss
    
class Custom_CrossEntropy_PSKD(nn.Module):
	def __init__(self):
		super(Custom_CrossEntropy_PSKD, self).__init__()
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, output, targets):
		"""
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
		log_probs = self.logsoftmax(output)
		loss = (- targets * log_probs).mean(0).sum()
		return loss   