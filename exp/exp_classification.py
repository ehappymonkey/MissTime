from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
from sklearn.metrics import f1_score, roc_auc_score
from datetime import datetime
from tqdm import tqdm 

warnings.filterwarnings('ignore')
from models.saits import SAITS


class Exp_Classification(Exp_Basic):
    _printed_shape = False 
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        # model init
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


        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_full, label, batch_mask, padding_mask) in enumerate(tqdm(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                if batch_x_full is not None:
                    batch_x_full = batch_x_full.to(self.device)
                # print(batch_x[0])
                # padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

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
                else:
                    outputs = self.model(batch_x, None, None, None, batch_mask, batch_x_full, mode='valid')


                pred = outputs.detach()
                loss = criterion(pred, label.long().squeeze())
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TRAIN') # 验证数据不mask。只有测试数据缺失。
        test_data, test_loader = self._get_data(flag='TEST')

        # print(test_data[0][0])
        if not Exp_Classification._printed_shape:
            print(f"Sample X shape: {train_data[0][0].shape}  (seq_len, channels)")
            all_labels = [train_data[i][1].item() for i in range(len(train_data))]
            num_classes = len(set(all_labels))
            print(f"Number of classes: {num_classes}")
            Exp_Classification._printed_shape = True

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

            for i, (batch_x, batch_x_full, label, batch_mask, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                if batch_x_full is not None:
                    batch_x_full = batch_x_full.to(self.device)
                # padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

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
                    loss_impute = outputs['reconstruction_loss'] + outputs['imputation_loss'] # 相当于直接在full data上做个loss嘛。
                    output_logits = outputs['output']
                    loss_classify = criterion(output_logits, label.long().squeeze(-1))
                    loss = loss_impute + loss_classify
                else:
                    outputs = self.model(batch_x, None, None, None, batch_mask, batch_x_full, mode='train')
                    loss = criterion(outputs, label.long().squeeze(-1))
                
                # else:
                #     outputs = self.model(batch_x, padding_mask, None, None)

                
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            if not self.args.rag_type == 'latent_rag':
                test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)
                print(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                    .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
                
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
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
            for i, (batch_x, batch_x_full, label, batch_mask, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                if batch_x_full is not None:
                    batch_x_full = batch_x_full.to(self.device)
                # padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

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
                else:
                    outputs = self.model(batch_x, None, None, None, batch_mask, batch_x_full, mode='test')

               
                # else:
                #     outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)


    

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        print("Preds:", predictions)
        print("Labels", trues)
        accuracy = cal_accuracy(predictions, trues)

        num_classes = probs.shape[1]
        if num_classes == 2:
            f1 = f1_score(trues, predictions, average='macro')
        else:
            f1 = f1_score(trues, predictions, average='macro') 
        print(f"F1 Score: {f1:.4f}")

        probs = probs.cpu().numpy()
        if num_classes == 2:
            # 二分类：用正类概率
            auc = roc_auc_score(trues, probs[:, 1])
        else:
            # 多分类：用 OvR 策略 + macro 平均
            auc = roc_auc_score(trues, probs, multi_class='ovr', average='macro')
        print(f"AUC: {auc:.4f}")

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('accuracy:{}'.format(accuracy))
        file_name='result_classification.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        # f.write(setting + "  \n")
        current_time = datetime.now().strftime("%m-%d %H:%M")  # 格式: 06-30 14:25
        f.write(f"{self.args.model}_{self.args.rag_type}_{self.args.retrieve_encoder} missing: ({self.args.mask_ratio}) ({current_time})\n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('AUC:{}'.format(auc))
        f.write('F1:{}'.format(f1))
        f.write('\n')
        f.write('\n')
        f.close()
        print('results saved to', os.path.join(folder_path,file_name))


        return