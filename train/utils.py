import pandas as pd
import torch
import json
from easydict import EasyDict as edict
import os
import numpy as np
import random
from torch import distributed as dist

class Recorder:
	def __init__(self, name):
		self.metrics = {}
		self.name = name
	def record(self, metrics_name, index, metrics):
		if metrics_name not in self.metrics:
			self.metrics[metrics_name] = ([index,], [metrics,])
		else:
			self.metrics[metrics_name][0].append(index)
			self.metrics[metrics_name][1].append(metrics)
	def save(self, metrics_name, dataset, TPR, FPR):
		save_dir = '../res/convergence/{}_{}_TPR_{}_FPR_{}.csv'.format(self.name, dataset, TPR, FPR)
		data = pd.DataFrame(index=self.metrics[metrics_name][0])
		data[metrics_name] = self.metrics[metrics_name][1]
		data.to_csv(save_dir)
	def clear(self):
		self.metrics = {}
	def save_test(self, metrics_name, dataset, metrics,TPR, FPR):
		data = pd.read_csv('../res/res_{}_TPR_{}_FPR_{}.csv'.format(metrics_name, TPR, FPR), index_col=0).copy()
		data[dataset][self.name] = metrics
		data.to_csv('../res/res_{}_TPR_{}_FPR_{}.csv'.format(metrics_name, TPR, FPR), index=True)
	def save_model(self, method, model, dataset, metrics_name, metrics, hyperparam):
		if not os.path.exists('../trained_models/{}/{}_{}'.format(dataset, metrics_name, metrics)):
			os.makedirs('../trained_models/{}/{}_{}'.format(dataset, metrics_name, metrics))
		torch.save(model, '../trained_models/{}/{}_{}/{}.pth'.format(dataset, metrics_name, metrics, method))
		with open('../trained_models/{}/{}_{}/{}.json'.format(dataset, metrics_name, metrics, method), 'w') as f:
			json.dump(hyperparam, f)
	def read_model(self, method, dataset, metrics_name, metrics):
		model = torch.load('../trained_models/{}/{}_{}/{}.pth'.format(dataset, metrics_name, metrics, method))
		return model
			

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def load_param(method, dataset, TPR, FPR):
    if TPR == 1:
        metric = 'OPAUC'
    else:
        metric = 'TPAUC'
    with open('../trained_models/{}/{}_{}/{}.json'.format(dataset, metric, FPR, method), 'r') as f:
        args = json.load(f)
        return args

def load_json(dataset=None, model=None):
	with open('../configs/base_config.json', 'r') as f:
		args = json.load(f)
		args = edict(args)
	if dataset is not None:
		with open('../configs/datasets/%s.json'%dataset, 'r') as f:
			args.dataset.update(edict(json.load(f)))
	if model is not None:
		with open('../configs/models/%s.json'%model, 'r') as f:
			args.model.update(edict(json.load(f)))
	return args