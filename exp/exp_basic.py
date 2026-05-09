import os
import torch
from models import RAFT_DLinear, Dlinear, DLinear, RMTS_feature, TimesNet3, TimeXer, TimeMixer2, MSGNet, TimeMixer, iTransformer, duet, TimeFilter, TSRAG_TimesNet


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'RAFT': RAFT_DLinear,
            'Dlinear': Dlinear,
            'DLinear': DLinear,
            'RMTS_feature': RMTS_feature,
            'TimesNet': TimesNet3,
            'TimeXer': TimeXer,
            'MSGNet': MSGNet,
            'TimeMixer': TimeMixer,
            'TimeMixer2': TimeMixer2, # equip with rag
            'iTransformer': iTransformer,
            'Duet': duet, 
            # 'GinAR': MSGNet,
            'TimeFilter': TimeFilter,
            'TSRAG': TSRAG_TimesNet,
            # 'Moirai': Moirai,
        }

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(
            #     self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
