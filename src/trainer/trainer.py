import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import time

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.train_metrics = MetricTracker('Total_loss',  *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('Total_loss',  *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (target, directory) in enumerate(self.data_loader):
            for key in target:
                if target[key].size==0: print(directory); raise
                target[key] = target[key].to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(target['input'],target['adj1'],target['adj2'],target['adj3'],target['adj4'],target['c1'],target['c2'],target['c3'],target['c4'],target['ver_num'], 'train') # data: images, output: (params, R, logits)
            loss, loss_valdict = self.criterion(target, self.config['loss_weights'], logits)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('Total_loss', loss.item())

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(logits, target))

            # sampled_point = surfaceSampling(coeff, R, Lambda, logits, target, self.config['PIval_filter'], self.allpoints, self.data_loader.point_limit[0]*3) # on points
            # chamferL1, chamfer_accuracy, chamfer_complete = chamfer_distance_naive(sampled_point, target['cordinate'])
            # self.train_metrics.update('ChamferL1', chamferL1)
            # self.train_metrics.update('Chamfer_accuracy', chamfer_accuracy)
            # self.train_metrics.update('Chamfer_complete', chamfer_complete)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} TotalLoss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()
                ))
                # self.writer.add_image('input', make_grid(data.detach().cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

            
            del target
            del loss

        log = self.train_metrics.result()

        if self.do_validation:
            s = time.time()
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            t = time.time()
            print(t-s)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.writer.close()
        
        return log


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (target, directory) in enumerate(self.valid_data_loader):
                for key in target:
                    if target[key].size==0: print(directory); raise
                    target[key] = target[key].to(self.device)

                logits = self.model(target['input'],target['adj1'],target['adj2'],target['adj3'],target['adj4'],target['c1'],target['c2'],target['c3'],target['c4'],target['ver_num'], 'test') # data: images, output: (params, R, logits)
                loss, loss_valdict = self.criterion(target, self.config['loss_weights'],logits)
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'test')            
                self.valid_metrics.update('Total_loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(logits, target))


                self.writer.close()
                del target
                del loss


        self.writer.close()
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

