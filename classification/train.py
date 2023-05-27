import time 
import argparse

import torch 
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from models import * 
from dataset import BaselineDataset, CutOutDataset, CutMixDataset, MixUpDataset 


## 1. Hyper - Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--batchsize", type=int, default=100)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--mode", type=int, default=0)
args = parser.parse_args()

if args.gpu >= 0 and type(args.gpu) == int:
    device = torch.device(f"cuda:{args.gpu}")
else:
    device = torch.device("cpu")


## 2. Dataset & DataLoader
test_data = BaselineDataset(train=False)
if args.mode == 0:
    train_data = BaselineDataset(train=True)
elif args.mode == 1:
    train_data = CutOutDataset()
elif args.mode == 2:
    train_data = MixUpDataset()
elif args.mode == 3:
    train_data = CutMixDataset()

TrainLoader = DataLoader(train_data, batch_size=args.batchsize, shuffle=True)
TestLoader = DataLoader(test_data, batch_size=args.batchsize, shuffle=False)


## 3. Loss part
def CrossEntropy(target, prediction):
    prob = prediction.softmax(dim=-1)
    log_prob = torch.log(prob + 1e-6)
    entropy =  target * log_prob
    loss = - entropy.sum(-1).mean()
    return loss 


## 4. training 
model = ResNet18().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

total_train_step = 0
total_test_step = 0
writer = SummaryWriter("./results/logdir")

start_time = time.time()
for i in range(args.epoch):
    print("————————Epoch: {}————————".format(i+1))

    # start training
    model.train()
    for data in TrainLoader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = CrossEntropy(targets, outputs)
        print(loss)

        # optimizer
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time-start_time)
            print("Steps:{}, Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # testing
    model.eval()
    total_test_loss = 0
    total_accuracy = 0  
    with torch.no_grad():  
        for data in TestLoader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = CrossEntropy(targets, outputs)

            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets.argmax(1)).sum()
            total_accuracy = total_accuracy + accuracy

    print("Total loss:{}".format(total_test_loss))
    print("Accuracy:{}".format(total_accuracy/len(test_data)))

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/len(test_data), total_test_step)
    total_test_step = total_test_step + 1

    torch.save(model, f"results/checkpoints/cifar_{args.mode}_{i}.pth")
    print("Done!")


writer.close()









    