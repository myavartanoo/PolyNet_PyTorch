import torch.nn.functional as F
import torch.nn as nn
import torch





CELoss = nn.CrossEntropyLoss()




def total_loss(target, loss_weights,logits):
    '''
    logits:
        dim: (10,) - 10 classes in dataset
    * loss shuold be calculated on gpu not to be bottleneck
    '''
    loss = loss_weights['class']*loss_class(target['class_num'],logits)

    return loss,\
            {'loss_cls':loss.detach().item()
             }


def loss_class(labels, logits):
    '''
    logits: 
        (batch, 10) - Output from net. Containing softmax values in [0,1]
    labels: 
        (batch, 1) - target['class_num'] : class numbers in 0,1,2,...,9.

    * CrossEntropyLoss contains auto one-hot encoding for target(labels). see func CrossEntropyLoss in docs
    '''
    loss_class = CELoss(logits, labels.long())

    if not torch.isfinite(loss_class): print("problem on loss_class\n", "labels\n",labels, "\nlogits\n",logits); raise    
    return loss_class

if __name__ == '__main__':
    print('loss')
