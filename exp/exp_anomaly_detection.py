from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')
from models.saits import SAITS
from models.GinAR import GinAR

class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

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
        
        elif self.args.model == 'GinAR':
            model = GinAR(self.args, seq_len=self.args.seq_len, enc_in=self.args.enc_in, pred_len=self.args.pred_len)
        
        else:
            model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        if self.args.rag_type == 'latent_rag':
            _, train_loader_contra = self._get_data(flag='train', shuffle_flag=True, batch_size=self.args.contrastive_batch)
            _, train_loader_unshuffled = self._get_data(flag='train', shuffle_flag=False)
            _, vali_loader_contra = self._get_data(flag='train', shuffle_flag=True, batch_size=self.args.contrastive_batch)
            # TimesNet
            model.prepare_retrieval(train_loader_contra, vali_loader_contra, train_loader_unshuffled)


        
        # elif self.args.rag_type == 'feature_rag' or self.args.rag_type == 'latent_rag':
        #     train_data, train_loader = self._get_data(flag='train')
        #     model.prepare_dataset(train_data, task_name='anomaly_detection')
        return model

    def _get_data(self, flag, shuffle_flag=None, batch_size=None):
        data_set, data_loader  = data_provider(self.args, flag, shuffle_flag, batch_size)
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
            for i, (batch_x, batch_x_full, batch_y, batch_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_full = batch_x_full.float().to(self.device) 

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
                elif self.args.model == 'GinAR':
                    outputs = self.model(batch_x)
                else:
                    outputs, _ = self.model(batch_x, None, None, None, batch_mask, batch_x_full, mode='valid')
                

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach()
                true = batch_x.detach()

                loss = criterion(pred, true)
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
            for i, (batch_x, batch_x_full, batch_y, batch_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
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
                    loss_impute = outputs['reconstruction_loss'] + outputs['imputation_loss']
                    imputed_data, reconstructed_data = outputs['imputed_data'], outputs['output']
                    loss_recon = criterion(reconstructed_data, imputed_data)
                    loss = loss_impute + loss_recon
                elif self.args.model == 'GinAR':
                    outputs = self.model(batch_x)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    print(outputs)
                    loss = criterion(outputs, batch_x)
                else:
                    outputs, _ = self.model(batch_x, None, None, None, batch_mask, batch_x_full, mode='train')
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    loss = criterion(outputs, batch_x)
                
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
            # vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)

            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            # early_stopping(vali_loss, self.model, path)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_x_full, batch_y, batch_mask) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
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
                    outputs = self.model(inputs, stage='test')
                    imputed_data, recon_data = outputs['imputed_data'], outputs['output']
                    score = torch.mean(self.anomaly_criterion(imputed_data, recon_data), dim=-1)
                elif self.args.model == 'GinAR':
                    outputs = self.model(batch_x)
                    score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                else:
                    outputs, _ = self.model(batch_x, None, None, None, batch_mask, batch_x_full, mode='test')
                    # criterion
                    score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)


        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_x_full, batch_y, batch_mask) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_x_full = batch_x_full.float().to(self.device)
            # reconstruction
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
                outputs = self.model(inputs, stage='test')
                imputed_data, recon_data = outputs['imputed_data'], outputs['output']
                score = torch.mean(self.anomaly_criterion(imputed_data, recon_data), dim=-1)


            else:
                outputs, x_recon_feat = self.model(batch_x, None, None, None, batch_mask, batch_x_full, mode='test')
                score = torch.mean(self.anomaly_criterion(x_recon_feat, outputs), dim=-1)
                


            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

        file_name = "result_anomaly_detection.txt"
        f = open(os.path.join(folder_path,file_name), 'a')
        current_time = datetime.now().strftime("%m-%d %H:%M")
        f.write(f"{self.args.model}_{self.args.rag_type}_{self.args.retrieve_encoder} missing: ({self.args.mask_ratio}) ({current_time})\n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        f.write('\n')
        f.write('\n')
        f.close()
        print('results saved to', os.path.join(folder_path,file_name))

        return