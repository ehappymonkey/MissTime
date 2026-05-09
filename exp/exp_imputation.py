from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from datetime import datetime
import math
warnings.filterwarnings('ignore')

from models.saits import SAITS

class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)

    def _build_model(self):
        if self.args.model == 'saits':
            saits_config = {
                "input_with_mask": True,
                "MIT": True,  # 训练时设为 True
                "param_sharing_strategy": "inner_group",
                "device": self.device, # 或者是 'cuda'
                "diagonal_attention_mask": True
            }
            model = SAITS(configs=self.args, n_groups=2, n_group_inner_layers=1, d_time=self.args.seq_len, d_feature=self.args.enc_in, d_model=self.args.d_model, d_inner=self.args.d_ff, n_head=self.args.n_heads,d_k=64, d_v=64, dropout=self.args.dropout, **saits_config)
        else:
            model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        if self.args.model == 'RAFT' or self.args.model == 'TSRAG' :
            _, train_loader_unshuffled = self._get_data(flag='train', shuffle_flag=False)
            model.prepare_retrieval(train_loader_unshuffled)
        
        if self.args.rag_type == 'latent_rag':
            _, train_loader_contra = self._get_data(flag='train', shuffle_flag=True, batch_size=self.args.contrastive_batch)
            _, train_loader_unshuffled = self._get_data(flag='train', shuffle_flag=False)
            _, vali_loader_contra = self._get_data(flag='train', shuffle_flag=True, batch_size=self.args.contrastive_batch)
            # TimesNet
            model.prepare_retrieval(train_loader_contra, vali_loader_contra, train_loader_unshuffled) # 前两个用于训练Encoder，后一个建立raw_data和embedding。


        return model

    def _get_data(self, flag, shuffle_flag=None, batch_size = None):
        data_set, data_loader = data_provider(self.args, flag, shuffle_flag, batch_size)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_mask, batch_x_full) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_x_full = batch_x_full.to(self.device)
                batch_mask = batch_mask.to(self.device)

                
                if self.args.model == 'saits':
                    B,L = batch_x.shape[0], batch_x.shape[1]
                    batch_mask_expanded = batch_mask.unsqueeze(0).unsqueeze(0).expand(B, L, -1).to(self.device)
                    missing_mask = batch_mask_expanded
                    indicating_mask = 1.0 - missing_mask
                    inputs = {
                    "X": batch_x,
                    "missing_mask": missing_mask,
                    "indicating_mask": indicating_mask,
                    "X_holdout": batch_x_full 
                    }
                    outputs = self.model(inputs, stage='val')['imputed_data']
                    # loss = outputs['reconstruction_loss'] + outputs['imputation_loss'] # 相当于直接在full data上做个loss嘛。
                
                elif self.args.model == 'TimeFilter':
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs, l_moe = self.model(batch_x)
                    else:
                        outputs, l_moe = self.model(batch_x)
                
                elif self.args.model == 'TimeMixer' or self.args.model == 'iTransformer' or self.args.model == 'GinAR' or self.args.model == 'MSGNet':
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark, mask=None)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark, mask=None)
    
                else:
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark, batch_mask, batch_x_full, mode='valid', mask=None)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                # add support for MS
                batch_x = batch_x[:, :, f_dim:]
                pred = outputs.detach()
                true = batch_x_full.detach()

                # loss = criterion(pred[mask == 0], true[mask == 0])
                loss = criterion(
                    pred[:, :, batch_mask == 0], 
                    true[:, :, batch_mask == 0]
                ) # 验证时候计算missing通道上的损失。
                # loss = criterion(outputs, batch_x_full)
                if not math.isnan(loss):
                    total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_mask, batch_x_full) in enumerate(train_loader): 
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_x_full = batch_x_full.to(self.device)

                
                if self.args.model == 'saits':
                    B,L = batch_x.shape[0], batch_x.shape[1]
                    batch_mask_expanded = batch_mask.unsqueeze(0).unsqueeze(0).expand(B, L, -1).to(self.device)
                    missing_mask = batch_mask_expanded
                    indicating_mask = 1.0 - missing_mask
                    inputs = {
                    "X": batch_x,
                    "missing_mask": missing_mask,
                    "indicating_mask": indicating_mask,
                    "X_holdout": batch_x_full 
                    }
                    outputs = self.model(inputs, stage='train') 
                    loss = outputs['reconstruction_loss'] + self.args.weight*outputs['imputation_loss'] # 相当于直接在full data上做个loss嘛。
                

                elif self.args.model == 'TimeFilter':
                    batch_x = batch_x_full 
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs, l_moe = self.model(batch_x, is_training=True)
                    else:
                        outputs, l_moe = self.model(batch_x, is_training=True)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y) + l_moe


                elif self.args.model == 'TimeMixer' or self.args.model == 'iTransformer' or self.args.model == 'GinAR' or self.args.model == 'MSGNet':
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark, mask=None)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark, mask=None)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    batch_x = batch_x[:, :, f_dim:]
                    loss = criterion(outputs, batch_x_full) # 训练时候无missing channels，只能计算所有通道的损失！
                    train_loss.append(loss.item())

    
                else: # TimesNet, RAFT, TSRAG
                    outputs  = self.model(batch_x, batch_x_mark, None, batch_y_mark, batch_mask, batch_x_full, mode='train', mask=None)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    # add support for MS
                    batch_x = batch_x[:, :, f_dim:]
                    loss = criterion(outputs, batch_x_full) # 训练时候无missing channels，只能计算所有通道的损失！
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)

            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        batch_masks = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_mask, batch_x_full) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_x_full = batch_x_full.to(self.device)
                batch_mask = batch_mask.to(self.device)

                if self.args.model == 'saits':
                    B,L = batch_x.shape[0], batch_x.shape[1]
                    batch_mask_expanded = batch_mask.unsqueeze(0).unsqueeze(0).expand(B, L, -1).to(self.device)
                    missing_mask = batch_mask_expanded
                    indicating_mask = 1.0 - missing_mask
                    inputs = {
                    "X": batch_x,
                    "missing_mask": missing_mask,
                    "indicating_mask": indicating_mask,
                    "X_holdout": batch_x_full 
                    }
                    outputs = self.model(inputs, stage='test')['imputed_data']
                    # loss = outputs['reconstruction_loss'] + outputs['imputation_loss'] # 相当于直接在full data上做个loss嘛。
                  
                elif self.args.model == 'TimeFilter':
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs, l_moe = self.model(batch_x)
                    else:
                        outputs, l_moe = self.model(batch_x)

                elif self.args.model == 'TimeMixer' or self.args.model == 'iTransformer' or self.args.model == 'GinAR' or self.args.model == 'MSGNet':
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark, mask=None)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark, mask=None)
    
                else: 
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark, batch_mask, batch_x_full, mode='test', mask=None)
                    # outputs是模型的输出。

                # eval
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                # add support for MS 
                batch_x = batch_x[:, :, f_dim:]
                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                # true = batch_x.detach().cpu().numpy() # 这个true是否应该改为batch_x_full?
                true = batch_x_full.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                batch_mask_np = batch_mask.detach().cpu().numpy()
                batch_mask_np = np.tile(batch_mask_np, (batch_x.shape[0], 1))
                batch_masks.append(batch_mask_np)

                # if i % 20 == 0:
                #     filled = true[0, :, -1].copy()
                #     filled = filled * mask[0, :, -1].detach().cpu().numpy() + \
                #              pred[0, :, -1] * (1 - mask[0, :, -1].detach().cpu().numpy())
                #     visual(true[0, :, -1], filled, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        batch_masks = np.concatenate(batch_masks, 0)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        masks_expanded = np.expand_dims(batch_masks, axis=1)
        masks_expanded = np.repeat(masks_expanded, preds.shape[1], axis=1)
        eval_preds = preds[masks_expanded == 0]
        eval_trues = trues[masks_expanded == 0]
        mae, mse, rmse, mape, mspe = metric(eval_preds, eval_trues) # 测试时候计算缺失通道上的损失。是不是应该计算所有的呢？
        mae, mse, rmse, mape, mspe = metric(preds, trues) 
        print('mse:{}, mae:{}'.format(mse, mae))
        file_name = "result_imputation.txt"
        f = open(os.path.join(folder_path,file_name), 'a')
        current_time = datetime.now().strftime("%m-%d %H:%M")  # 格式: 06-30 14:25
        f.write(f"{self.args.model}_{self.args.rag_type}_{self.args.retrieve_encoder} missing: ({self.args.mask_ratio}) ({current_time})\n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()
        print('results saved to', os.path.join(folder_path,file_name))


        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return