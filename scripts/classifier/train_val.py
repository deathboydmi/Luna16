import argparse
import os
import random
import shutil
import time
import warnings
import logging as log

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import model as m

parser = argparse.ArgumentParser(description='PyTorch LUNA16 Training')
parser.add_argument('data', metavar='DIR',
					help='path to dataset')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
					help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
					metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
					metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=5, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
					help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
					help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
					help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
					help='seed for initializing training. ')

best_prec1 = 0
#torch.set_printoptions(threshold=512*256)

log.basicConfig(filename='./log_train_val.log', format='%(asctime)s %(message)s', level=log.INFO)

def main():
	global args, best_prec1
	args = parser.parse_args()

	args.distributed = args.world_size > 1
	if args.distributed:
		dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
								world_size=args.world_size)
	model = m.NodulesClassifier()
	# for module in model.modules():
		# 	if type(module) == nn.BatchNorm2d:
		# 		for param in module.parameters():
		# 				param.requires_grad = False
		
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr)
	# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
	# 						args.lr,
	# 						momentum=args.momentum,
	# 						weight_decay=args.weight_decay)

	if args.resume:
		if os.path.isfile(args.resume):
			log.info("=> loading checkpoint '{}'".format(args.resume))
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_prec1 = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			log.info("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch']))
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch']))
		else:
			log.info("=> no checkpoint found at '{}'".format(args.resume))
			print("=> no checkpoint found at '{}'".format(args.resume))

	traindir = os.path.join(args.data, 'train')
	valdir = os.path.join(args.data, 'val')
	normalize = transforms.Normalize(mean=[0.485], std=[0.229])

	train_dataset = datasets.ImageFolder(
		traindir,
		transforms.Compose([
			transforms.Grayscale(),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		]))

	if args.distributed:
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
	else:
		train_sampler = None

	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
		num_workers=args.workers, pin_memory=True, sampler=train_sampler)

	val_loader = torch.utils.data.DataLoader(
		datasets.ImageFolder(valdir, transforms.Compose([
			transforms.Grayscale(),
			transforms.ToTensor(),
			normalize,
		])),
		batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)

	if args.evaluate:
		validate(val_loader, model, criterion)
		return

	for epoch in range(args.start_epoch, args.epochs):
		if args.distributed:
			train_sampler.set_epoch(epoch)

		log.info('------ Epoch #{0} ------'.format(epoch))		
		train(train_loader, model, criterion, optimizer, epoch)
		adjust_learning_rate(optimizer, epoch)

		prec1 = validate(val_loader, model, criterion)

		is_best_acc = prec1 > best_prec1

		best_prec1 = max(prec1, best_prec1)
		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'best_prec1': best_prec1,
			'optimizer' : optimizer.state_dict(),
		}, is_best_acc, filename="./checkpoint/model_chp.pth.tar")


def train(train_loader, model, criterion, optimizer, epoch):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()

	model.train()

	end = time.time()
	for i, (input, target) in enumerate(train_loader):
		data_time.update(time.time() - end)

		output = model(input)

		loss = criterion(output, target)

		losses.update(loss.item(), input.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			log.info('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(		  
				   epoch, i, len(train_loader), batch_time=batch_time,
				   data_time=data_time, loss=losses))
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
				   epoch, i, len(train_loader), batch_time=batch_time,
				   data_time=data_time, loss=losses))

def validate(val_loader, model, criterion):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	with torch.no_grad():
		end = time.time()
		for i, (input, target) in enumerate(val_loader):

			output = model(input)
			loss = criterion(output, target)

			prec1 = accuracy(output, target)
			losses.update(loss.item(), input.size(0))
			top1.update(float(prec1[0]), input.size(0))

			batch_time.update(time.time() - end)
			end = time.time()

			log.info('Test: [{0}/{1}]\t'
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
					i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))
			print('Test: [{0}/{1}]\t'
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
					i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

		log.info(' * Prec@1 {top1.avg:.3f}'
			  .format(top1=top1))
		print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
	return top1.avg

# need changes
def save_checkpoint(state, is_best_acc, filename):
	torch.save(state, filename)
	if is_best_acc:
		shutil.copyfile(filename, filename.replace("_chp", "_best_acc"))


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 100 every 15 epochs"""
	lr = args.lr * (0.01 ** (epoch // 15))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res

if __name__ == '__main__':
	main()

#endfile