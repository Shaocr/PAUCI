# 基于局部AUROC优化的端到端长尾识别框架

## 1. 算法描述
针对一些特定分类场景下（如医学诊断、金融风险账户检测等）如何更好的对难样本进行挖掘和优化，提出一种基于逐样本的局部AUROC优化的端到端长尾识别框架，使得模型同时关注ROC曲线下的高真阳性率（TPR）和低假阳性率（FPR）部分的面积，从而保证模型具有更好的泛化能力，显著提升长尾识别任务的性能。

## 2. 环境依赖及安装
该框架所需的环境依赖如下：
- easydict==1.9
- lmdb==0.9.24
- numpy==1.19.2
- pytorch==1.8.1
- scikit-image==0.18.1
- scikit-learn==0.24.1
- torchaudio==0.8.1
- torchvision==0.2.2
- tqdm==4.59.0

建议使用anaconda或pip配置环境。例如：
```
pip install easydict==1.9
pip install lmdb==0.9.24
pip install numpy==1.19.2
pip install pytorch==1.8.1
pip install scikit-image==0.18.1
pip install scikit-learn==0.24.1
pip install torchaudio==0.8.1
pip install torchvision==0.2.2
pip install tqdm==4.59.0
```

## 3. 运行示例

### 模型训练
模型训练需预先在params文件夹下对应数据集的json文件中配置相应的"class2id", 正样本（少数类）为1，其余默认为0，并运行如下命令：
```
python3 train/train_SPAUCI.py
```

本框架主要提供以下三个数据集，所有数据均已公开，具体如下：
- CIFAR-10-LT: 可通过[此链接下载](https://github.com/statusrank/XCurve/tree/master/example/data)
- CIFAR-100-LT: 可通过[此链接下载](https://github.com/statusrank/XCurve/tree/master/example/data)
- Tiny-ImageNet-200-LT: 可通过[此链接下载](https://drive.google.com/file/d/1WYoQrDIDK-E2aK8Rj_Vph_MBXIDjusHs/view)

可选两种训练损失函数，One-way Partial AUC Loss “PAUCI(alpha=1, beta=0.5)”和Two-way Partial AUC Loss “PAUCI(alpha=0.5, beta=0.5)”

此外，该算法已整合至XCurve通用框架中且兼容pytorch训练模式。
1. 执行如下命令安装XCurve
```
git clone https://github.com/statusrank/XCurve.git
python setup.py install
```
2. 通过如下方式定制训练：
```python3
# 基于局部AUROC优化的端到端长尾识别损失
from XCurve.AUROC.losses import PAUCI
from XCurve.AUROC.optimizer.ASGDA import ASGDA

model = {"your pytorch model"}
device = "cuda" if torch.cuda.is_available() else "cpu"

# 超参数
hparams = {
	"k": 1,
	"c1": 3,
	"c2": 3,
	"lam": 0.02,
	"nu": 0.02,
	"m": 500,
	"device": device,
}

# 训练损失
criterion = PAUCI(alpha=0.5, beta=0.5)
optimizer = ASGDA([
		{'params': model.parameters(), 'name':'net'},
		{'params': [criterion.a, criterion.b], 'clip':(0, 1), 'name':'ab'},
		{'params': criterion.s_n, 'clip':(0, 5), 'name':'sn'},
		{'params': criterion.s_p, 'clip':(-4, 1), 'name':'sp'},
		{'params': criterion.lam_b, 'clip':(0, 1e9), 'name':'lamn'},
		{'params': criterion.lam_a, 'clip':(0, 1e9), 'name':'lamp'},
		{'params': criterion.g, 'clip':(-1, 1), 'name':'g'}], 
		hparams=hparams)

# create Dataset (train_set, val_set, test_set) and dataloader (trainloader)
# You can construct your own dataset/dataloader 
# but must ensure that there at least one sample for every class in each mini-batch 

train_set, val_set, test_set = your_datasets(...)
trainloader, valloader, testloader = your_dataloaders(...)

epoch = 1
# forward of model
for x, target in trainloader:
		optimizer.zero_grad()
    x, target  = x.cuda(), target.cuda()
    pred = model(x)
		loss = criterion(pred, target)
		loss.backward()
		optimizer.step(pre=True, t=epoch)

		optimizer.zero_grad()
    pred = model(x)
		loss = criterion(pred, target)
		loss.backward()
		optimizer.step(pre=False, t=epoch)
    
    epoch += 1
```

## 4. 论文/专利成果
> Huiyang Shao, Qianqian Xu, Zhiyong Yang, Shilong Bao, Qingming Huang. [Asymptotically Unbiased Instance-wise Regularized Partial AUC Optimization: Theory and Algorithm.](https://arxiv.org/pdf/2210.03967.pdf) NIPS 2022
