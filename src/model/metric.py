import torch


def accuracy(logits, target):
    '''
    logits: 
        (batch, 10) - Output from net. Containing softmax values in [0,1]
    labels: 
        (batch) - target['class_num'] : class numbers in 0,1,2,...,9.
    '''

    with torch.no_grad():
        pred = torch.argmax(logits, dim=1)
        assert pred.shape == target['class_num'].shape
        for i in range(pred.shape[0]):
           if pred[i]!=target['class_num'][i]:
              line = "\n Object:  "  + str(target['dir'][i].cpu().detach().numpy()[0]+1)+'/'+str(target['dir'][i].cpu().detach().numpy()[1]+1)+',   '+str(torch.argmax(pred[i]).cpu().detach().numpy()+1)+'/'+ str(target['class_num'][i].cpu().detach().numpy()+1)

              with open('result.txt', 'a') as f:
                        f.write(line)
        correct = torch.sum(pred == target['class_num']).item()
    return correct/target['class_num'].shape[0]

