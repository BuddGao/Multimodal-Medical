import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import os
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from models.resnet_simclr import ResNetSimCLR
#from resnet_simclr import ResNetSimCLR

from load_data import load_data

parser = argparse.ArgumentParser(description='PyTorch SimCLR Pre-training')
parser.add_argument('--data', type=str, default='QIN', help='dataset name')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', help='model architecture')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay', dest='weight_decay')


parser.add_argument('-j', '--workers', default=12, type=int, metavar='N', help='number of data loading workers (default: 12)')
parser.add_argument('--out_dim', default=1024, type=int, help='feature dimension')
parser.add_argument('--temperature', default=0.07, type=float, help='softmax temperature')
parser.add_argument('--gpu_index', default=0, type=int, help='Gpu index')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true', help='Use 16-bit precision')

class SimCLRDataTransform(object):
    def __init__(self, size):
        s = 1
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

class PretrainDataset(Dataset):
    def __init__(self, images):
        self.images = images
        self.transform = SimCLRDataTransform(size=96)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img_tensor = torch.from_numpy(img)
        
        x1, x2 = self.transform(img_tensor)
        return x1, x2

# InfoNCE Loss 
def info_nce_loss(features, args):
    batch_size = args.batch_size
    
 
    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(args.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

   
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(args.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

  
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(args.device)

    logits = logits / args.temperature
    return logits, labels

def main():
    args = parser.parse_args()
    
    
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu_index}')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')

    print(f"Running on device: {args.device}")

 
    print("Loading data...")
    train_images, _, _, _ = load_data(args)
    
    if train_images is None or len(train_images) == 0:
        print("Error: No images loaded.")
        return

    dataset = PretrainDataset(train_images)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.workers, drop_last=True, pin_memory=True)

    
    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
    
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    scaler = GradScaler(enabled=args.fp16_precision)

    
    save_dir = f'/content/drive/MyDrive/MML/QIN/Pretrain_model_emb{args.out_dim}'
    os.makedirs(save_dir, exist_ok=True)

   
    print("Start pre-training...")
    model.train()
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for x1, x2 in progress_bar:
            x1, x2 = x1.to(args.device), x2.to(args.device)
            
            
            images = torch.cat((x1, x2), dim=0)

            with autocast(enabled=args.fp16_precision):
                features = model(images)
                logits, labels = info_nce_loss(features, args)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

      
        if epoch >= 10:
            scheduler.step()

        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader)}")
                # 5. Checkpoint 
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_name = f'checkpoint_{epoch+1:04d}.pth.tar'
            save_path = os.path.join(save_dir, checkpoint_name)
            
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=save_path)
            print(f"Checkpoint saved: {save_path}")

    print(f"\nTraining finished! Final model saved to {save_dir}")


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

if __name__ == "__main__":
    main()