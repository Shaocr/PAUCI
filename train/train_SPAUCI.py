import sys
import os
import copy
sys.path.append(os.pardir)
import numpy as np
from losses.SPAUCI import SPAUCI
from optimizer.MinMax import MinMax
import torch
import torch.nn as nn
from dataloaders import get_datasets
from dataloaders import get_data_loaders
from models import generate_net
from metrics.partial_AUROC import p2AUC
from utils import Recorder, load_json, set_seed
import json
set_seed(11)

method='SPAUCI'

# hyper parameters
hyper_param = {
	'mini-batch':    1024,
	'alpha':         1.0,
	'beta':          0.3,
	'weight_decay':  1e-5,
	'init_lr': 		 0.001
}

if hyper_param['alpha'] == 1:
	metrics = 'OPAUC'
else:
	metrics = 'TPAUC'

rec = Recorder(method) # record the metrics during training
sigmoid = nn.Sigmoid() # Limit the output score between 0 and 1


for dataset in ['cifar-10-long-tail-1', 'cifar-10-long-tail-2', 'cifar-10-long-tail-3',
				'cifar-100-long-tail-1', 'cifar-100-long-tail-2', 'cifar-100-long-tail-3',
				'tiny-imagenet-200-1', 'tiny-imagenet-200-2', 'tiny-imagenet-200-3']:
	print(dataset)
	# load data and dataloader
	args = load_json(dataset)
	train_set, val_set, test_set = get_datasets(args.dataset)
	train_loader, val_loader, test_loader, data_num = get_data_loaders(
	  train_set,
	  val_set,
	  test_set,
	  hyper_param['mini-batch'],
	  hyper_param['mini-batch']
	)
	
    # load model (train model from the scratch, using model: resnet18)
	# args = load_json(dataset, 'resnet18')
	# args.model['pretrained'] = None
	# model = generate_net(args.model).cuda(0)
	# model = nn.DataParallel(model, device_ids=range(4))

	# load pre-trained model
	model = torch.load('../pretrained_models_back/{}.pth'.format(dataset)).cuda()
	model = nn.DataParallel(model, device_ids=range(4))

	# load hyper parameters from json file
	hparams = json.load(open('params.json', 'r'))[metrics][dataset]

	# define loss and optimizer
	criterion = SPAUCI(hyper_param['alpha'], hyper_param['beta'])
	optimizer = MinMax([
            {'params': model.parameters(), 'name':'net'},
            {'params': [criterion.a, criterion.b], 'clip':(0, 1), 'name':'ab'},
            {'params': criterion.s_n, 'clip':(0, 5), 'name':'sn'},
            {'params': criterion.s_p, 'clip':(-4, 1), 'name':'sp'},
			{'params': criterion.lam_b, 'clip':(0, 1e9), 'name':'lamn'},
            {'params': criterion.lam_a, 'clip':(0, 1e9), 'name':'lamp'},
            {'params': criterion.g, 'clip':(-1, 1), 'name':'g'}], 
            weight_decay=hyper_param['weight_decay'], hparams=hparams)
            
	best_model = model.state_dict()
	best_perf = 0
	all_counter = 0

	# train 50 epoch
	for epoch in range(50):
		all_pauc = 0
		counter = 0
		model.train()
		for i, (img, lbl, idx) in enumerate(train_loader):
			optimizer.zero_grad()
			img = img.cuda(0)
			lbl = lbl.cuda(0).float()
			out = sigmoid(model(img))
			loss = criterion(out, lbl)
			loss.backward()
			optimizer.step(pre=True, t=all_counter)
	
			optimizer.zero_grad()
			out = sigmoid(model(img))
			loss = criterion(out, lbl)
			loss.backward()
			optimizer.step(pre=False, t=all_counter)
			label = lbl.cpu().detach().numpy().reshape((-1, ))
			pred = out.cpu().detach().numpy().reshape((-1, ))
			all_pauc += p2AUC(label, pred, hyper_param['alpha'], hyper_param['beta'])
			counter += 1
			all_counter += 1
		# record instances' prediction and label of val set
		model.eval()
		val_pred = np.array([])
		val_label = np.array([])
		for i, (img, lbl, idx) in enumerate(train_loader):
			img = img.cuda(0)
			lbl = lbl.cuda(0).float()
			out = sigmoid(model(img))
			label = lbl.cpu().detach().numpy().reshape((-1, ))
			pred = out.cpu().detach().numpy().reshape((-1, ))
			val_pred = np.hstack([val_pred, pred])
			val_label = np.hstack([val_label, label])
		pauc = p2AUC(val_label, val_pred, hyper_param['alpha'], hyper_param['beta'])
		print('epoch:{} val pauc:{}'.format(epoch, pauc))
		rec.record(metrics, epoch, pauc)
		if pauc > best_perf:
			best_perf = pauc
			best_model = copy.deepcopy(model.state_dict())

			
	# calculate parial auc on testset 
	rec.save(metrics, dataset, hyper_param['alpha'], hyper_param['beta'])
	rec.clear()
	
	# record instances' prediction and label of test set
	model.load_state_dict(best_model)
	model.eval()
	test_pred = np.array([])
	test_label = np.array([])
	for i, (img, lbl, idx) in enumerate(test_loader):
		img = img.cuda(0)
		lbl = lbl.cuda(0)
		out = sigmoid(model(img))
		label = lbl.cpu().detach().numpy().reshape((-1, ))
		pred = out.cpu().detach().numpy().reshape((-1, ))
		test_pred = np.hstack([test_pred, pred])
		test_label = np.hstack([test_label, label])
	pauc = p2AUC(test_label, test_pred, hyper_param['alpha'], hyper_param['beta'])
	print('test pauc:{}'.format(pauc))
	rec.save_test(metrics, dataset, pauc, hyper_param['alpha'], hyper_param['beta'])
	rec.save_model(method, model, dataset, metrics, hyper_param['beta'], hyper_param)
