# STAnoGAN
《一种左心室超声心动图视频的半监督异常检测方法》

运行环境要求：
1. Ubuntu16.04
2. Pytorch >= 0.4.1


模型训练：
1. python -m visdom.server
2. 训练：python train_*.py


模型测试：
1. python -m visdom.server
2. python validate_*.py

计算AUC结果：
python evaluate/AUC_*.py
