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
import copy
from utils.dtw_metric import dtw,accelerated_dtw
from datetime import datetime
from tqdm import tqdm 
warnings.filterwarnings('ignore')

from models.saits import SAITS
from models.GinAR import GinAR

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):

        if self.args.model == 'saits':
            saits_config = {
                "input_with_mask": True,
                "MIT": True,
                "param_sharing_strategy": "inner_group",
                "device": self.device,
                "diagonal_attention_mask": True
            }
            model = SAITS(configs=self.args, n_groups=2, n_group_inner_layers=1, d_time=self.args.seq_len, d_feature=self.args.enc_in, d_model=self.args.d_model, d_inner=self.args.d_ff, n_head=self.args.n_heads,d_k=64, d_v=64, dropout=self.args.dropout, **saits_config)
        else:
            model = self.model_dict[self.args.model].Model(self.args).float()
            model_size = sum(p.numel() * p.element_size() for p in model.parameters())
            print(f'Model size: {model_size / (1024 ** 2):.2f} MB')

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        if self.args.model == 'RAFT' or self.args.model == 'TSRAG' :
            _, train_loader_unshuffled = self._get_data(flag='train', shuffle_flag=False)
            model.prepare_retrieval(train_loader_unshuffled)

        if self.args.rag_type in['latent_rag', 'feature_rag']:
            _, train_loader_contra = self._get_data(flag='train', shuffle_flag=True, batch_size=self.args.contrastive_batch)
            _, train_loader_unshuffled = self._get_data(flag='train', shuffle_flag=False)
            _, vali_loader_contra = self._get_data(flag='train', shuffle_flag=True, batch_size=self.args.contrastive_batch)
            # TimesNet
            model.prepare_retrieval(train_loader_contra, vali_loader_contra, train_loader_unshuffled)


        # if self.args.rag_type == 'feature_rag' or self.args.rag_type == 'latent_rag':
        #     train_data, train_loader = self._get_data(flag='train')

        #         _, vali_loader = self._get_data(flag='val')
        #         model.prepare_dataset(train_data, train_loader, vali_loader)
        #     else:
        #         model.prepare_dataset(train_data, task_name='forecasting')

        return model

    def _get_data(self, flag, shuffle_flag=None, batch_size=None):
        data_set, data_loader  = data_provider(self.args, flag, shuffle_flag, batch_size)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.model == 'Duet':
            criterion = nn.HuberLoss(delta=0.5)
        else: 
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_mask, batch_x_full) in enumerate(tqdm(vali_loader, desc='Validation')):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_mask = batch_mask.to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if batch_x_full is not None:
                    batch_x_full = batch_x_full.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
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
                    outputs = self.model(inputs, stage='val')['output'] 

                elif self.args.model == 'Duet':
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                elif self.args.model == 'TimeFilter':
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs, l_moe = self.model(batch_x)
                    else:
                        outputs, l_moe = self.model(batch_x)

                elif self.args.model == 'TimeMixer' or self.args.model == 'iTransformer' or self.args.model == 'GinAR' or self.args.model == 'MSGNet':
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else: # TimesNet
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_mask, batch_x_full, mode='valid')
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_mask, batch_x_full, mode='valid')
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                # print("batch_y min:", batch_y.min().item())
                # print("batch_y max:", batch_y.max().item())
                # print("batch_y dtype:", batch_y.dtype)

                loss = criterion(pred, true)

                total_loss.append(loss)
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

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        best_valid_loss = float('inf')
        best_model = None
            
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_mask, batch_x_full) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_x_full = batch_x_full.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

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
                    loss_impute = outputs['reconstruction_loss'] + outputs['imputation_loss']
                    outputs = outputs['output']
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss_forecast = criterion(outputs, batch_y)
                    loss = self.args.weight*loss_impute+loss_forecast
                elif self.args.model == 'GinAR' or self.args.model == 'MSGNet':

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)

                elif self.args.model == 'Duet':
                    batch_x = batch_x_full 
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs, L_imp = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs, L_imp = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y) + L_imp

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

                elif self.args.model == 'TimeMixer' or self.args.model == 'iTransformer':
                    batch_x = batch_x_full 
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                elif self.args.model == 'RAFT':
                    batch_x = batch_x_full
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_mask, batch_x_full, mode='train')
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_mask, batch_x_full, mode='train')
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y.half())
                else: # TimesNet, RAFT, TSRAG
                    batch_x = batch_x_full
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_mask, batch_x_full, mode='train')
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_mask, batch_x_full, mode='train')
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            adjust_learning_rate(model_optim, epoch + 1, self.args)

            if any(keyword in self.args.model_id for keyword in ['traffic', 'ECL', 'PEMS']): # and self.args.rag_type == 'latent_rag': # do not do validation for large datasets
                best_model = copy.deepcopy(self.model)
            else:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                if vali_loss < best_valid_loss:
                    best_model = copy.deepcopy(self.model)
                    best_valid_loss = vali_loss
            # if not self.args.rag_type == 'latent_rag':
            #     test_loss = self.vali(test_data, test_loader, criterion)
            #     print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #         epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            
            # We do not use early stopping
            # early_stopping(vali_loss, self.model, path)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break

        best_model_path = path + '/' + 'checkpoint.pth'
        torch.save(best_model.state_dict(), best_model_path)
#         self.model.load_state_dict(torch.load(best_model_path))

#         return self.model
        return best_model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_mask, batch_x_full) in enumerate(tqdm(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_mask = batch_mask.to(batch_x.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if batch_x_full is not None:
                    batch_x_full = batch_x_full.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
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
                    outputs = self.model(inputs, stage='test')['output']
                elif self.args.model == 'Duet':
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                elif self.args.model == 'TimeFilter':
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs, l_moe = self.model(batch_x)
                    else:
                        outputs, l_moe = self.model(batch_x)
                
                elif self.args.model == 'TimeMixer' or self.args.model == 'iTransformer' or self.args.model == 'GinAR' or self.args.model == 'MSGNet':
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    import time
                    start = time.time()
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_mask, batch_x_full, mode='test')
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_mask, batch_x_full, mode='test')
                    end = time.time()
                    # print(f'RAG time: {self.model.time_rag:.2f} seconds')
                    # print(f'Inference time: {self.model.time_inf:.2f} seconds')
                    # print(f'Encoding time for retrieval: {self.model.time_encoding:.2f} seconds')
                    # print(f'Training time: {self.model.time_training:.2f} seconds')
                    # # print('Total time:', end-start)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    print('do reverse transform...')
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = -999


        # results_save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        file_name = "result_long_term_forecast.txt"
        f = open(os.path.join(folder_path,file_name), 'a')
        # f.write(setting + "  \n")
        current_time = datetime.now().strftime("%m-%d %H:%M")
        f.write(f"{self.args.model}_{self.args.rag_type}_{self.args.retrieve_encoder}_{self.args.contrastive_loss}_missing: ({self.args.mask_ratio}) ({current_time})\n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()
        print('results saved to', os.path.join(folder_path,file_name))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
