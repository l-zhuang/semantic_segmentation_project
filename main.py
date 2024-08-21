import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

import transforms as ext_transforms
from data.utils import enet_weighing, median_freq_balancing
import utils
# from data import Kitti as dataset
from data.camvid import CamVid as dataset
# from models.enet import ENet
from train import Train
from test import Test
from metric.iou import IoU



# for improvement
import lovasz_losses as L
from models.enet import ENet
# Default Arguments
height=512
width=512
dataset_dir='./'
batch_size=10
workers=4
mode='train'
imshow_batch=True
weighing='Enet'
ignore_unlabeled=True
class_weights=0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

image_transform = transforms.Compose(
    [transforms.Resize((height, width)),
        transforms.ToTensor()])

label_transform = transforms.Compose([
    transforms.Resize((height, width), Image.NEAREST),
    ext_transforms.PILToLongTensor()
])

# Load the training set as tensors
train_set = dataset(
    dataset_dir,
    transform=image_transform,
    label_transform=label_transform)
train_loader = data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    )

# Load the validation set as tensors
val_set = dataset(
    dataset_dir,
    mode='val',
    transform=image_transform,
    label_transform=label_transform)
val_loader = data.DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
    )

# Load the test set as tensors

test_set = dataset(
    dataset_dir,
    mode='test',
    transform=image_transform,
    label_transform=label_transform)
test_loader = data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    )

# Get encoding between pixel valus in label images and RGB colors
class_encoding = train_set.color_encoding
del class_encoding['road_marking']
# Get number of classes to predict
num_classes = len(class_encoding)

# Print information for debugging
print("Number of classes to predict:", num_classes)
print("Train dataset size:", len(train_set))



# # Get a batch of samples to display
# if args.mode.lower() == 'test':
#     images, labels = iter(test_loader).next()
# else:
#     images, labels = iter(train_loader).next()
# print("Image size:", images.size())
# print("Label size:", labels.size())
# print("Class-color encoding:", class_encoding)
# # Show a batch of samples and labels
# if imshow_batch:
#     print("Close the figure window to continue...")
#     label_to_rgb = transforms.Compose([
#         ext_transforms.LongTensorToRGBPIL(class_encoding),
#         transforms.ToTensor()
#     ])
#     color_labels = utils.batch_transform(labels, label_to_rgb)
#     utils.imshow_batch(images, color_labels)


# Get class weights from the selected weighing technique
print("\nWeighing technique:", weighing)
print("Computing class weights...")
print("(this can take a while depending on the dataset size)")

class_weights = enet_weighing(train_loader, num_classes)
class_weights = torch.from_numpy(class_weights).float().to(device)

# Set the weight of the unlabeled class to 0
if ignore_unlabeled:
    ignore_index = list(class_encoding).index('unlabeled')
    class_weights[ignore_index] = 0

print("Class weights:", class_weights)



############ Train ##################
print("\nTraining...\n")

# Hyperparameters
learning_rate=5e-4
weight_decay=2e-4
lr_decay_epochs=100
lr_decay=0.5
ignore_unlabeled=True
start_epoch=0
epochs=200
print_step=True

num_classes = len(class_encoding)

# Intialize ENet
model = ENet(num_classes).to(device)



# Check if the network architecture is correct
# print(model)

# We are going to use the CrossEntropyLoss loss function as it's most
# frequentely used in classification problems with multiple classes which
# fits the problem. This criterion  combines LogSoftMax and NLLLoss.
criterion_entropy = nn.CrossEntropyLoss(weight=class_weights)
#criterion_entropy = nn.CrossEntropyLoss(weight=class_weights)
criterion_lovasz = L.lovasz_softmax
# ENet authors used Adam as the optimizer
optimizer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)

# Learning rate decay scheduler
lr_updater = lr_scheduler.StepLR(optimizer, lr_decay_epochs,lr_decay)

# Evaluation metric
if ignore_unlabeled:
    ignore_index = list(class_encoding).index('unlabeled')
else:
    ignore_index = None

metric = IoU(num_classes, ignore_index=ignore_index)

# # Optionally resume from a checkpoint
# if args.resume:
#     model, optimizer, start_epoch, best_miou = utils.load_checkpoint(
#         model, optimizer, args.save_dir, args.name)
#     print("Resuming from model: Start epoch = {0} "
#             "| Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
# else:
#     start_epoch = 0
#     best_miou = 0

# Show a batch of samples and labels
images, labels = next(iter(train_loader))
print("Image size:", images.size())
print("Label size:", labels.size())
print("Class-color encoding:", class_encoding)

imshow_batch=True
if imshow_batch:
    print("Close the figure window to continue...")
    label_to_rgb = transforms.Compose([
        ext_transforms.LongTensorToRGBPIL(class_encoding),
        transforms.ToTensor()
    ])
    color_labels = utils.batch_transform(labels, label_to_rgb)
    utils.imshow_batch(images, color_labels)


# Start Training
print("Training Start...")

train = Train(model, train_loader, optimizer, criterion_entropy, metric, device)
val = Test(model, val_loader, criterion_entropy, metric, device)

epoch_l=[]
iou_l=[]
miou_l=[]
loss_l=[]
miou_val_l=[]
loss_val_l=[]

for epoch in range(epochs):
    print(">>>> [Epoch: {0:d}] Training".format(epoch))

    epoch_loss, (iou, miou) = train.run_epoch(print_step)
    epoch_loss_val, (iou_val, miou_val) = val.run_epoch(print_step)
    lr_updater.step()

    print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}| IoU: {2:.4f}".format(epoch, epoch_loss, miou,iou))
    print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}| IoU: {2:.4f}".format(epoch, epoch_loss_val,  miou_val,iou_val))
    epoch_l.append(epoch)
    #iou_l.append(iou)
    miou_l.append(miou)
    loss_l.append(epoch_loss)
    miou_val_l.append(miou_val)
    loss_val_l.append(epoch_loss_val)

    # if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
    #     print(">>>> [Epoch: {0:d}] Validation".format(epoch))

    #     loss, (iou, miou) = val.run_epoch(args.print_step)

    #     print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
    #             format(epoch, loss, miou))

    #     # Print per class IoU on last epoch or if best iou
    #     if epoch + 1 == args.epochs or miou > best_miou:
    #         for key, class_iou in zip(class_encoding.keys(), iou):
    #             print("{0}: {1:.4f}".format(key, class_iou))

    #     # Save the model if it's the best thus far
    #     if miou > best_miou:
    #         print("\nBest model thus far. Saving...\n")
    #         best_miou = miou
    #         utils.save_checkpoint(model, optimizer, epoch + 1, best_miou,
    #                                 args)
    #utils.save_checkpoint(model, optimizer, epoch + 1, miou)



###### Final Results Plotting and Evaluation ######

import matplotlib.pyplot as plt
import numpy as np
# Plot each line with a label
plt.plot(epoch_l, miou_l, label='Training Mean Iou',color='red')
plt.plot(epoch_l, loss_l, label='Training Average Loss',color='green')

# Add labels and a legend
plt.xlabel('# of epochs')
plt.ylabel('Metrics')
plt.title('Evalutaion on the Traning Set')
plt.legend()

# Show the plot
plt.savefig('./pic/TrainingCurve.png')
plt.show()

plt.plot(epoch_l, miou_val_l, label='Validation Mean Iou', color='blue')
plt.plot(epoch_l, loss_val_l, label='Validation Average Loss', color='green')

# Add labels and a legend
plt.xlabel('# of epochs')
plt.ylabel('Metrics')
plt.title('Evalutaion on the Validation Set')
plt.legend()

# Show the plot
plt.savefig('./pic/ValidationCurv.png')
plt.show()

name = 'model.pth'
save_dir = './'

assert os.path.isdir(
    save_dir), "The directory \"{0}\" doesn't exist.".format(save_dir)



#### Save model Checkpoint #####
model_path = os.path.join(save_dir, name)
checkpoint = {
    'epoch': epoch,
    'miou': miou,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict()
}
torch.save(checkpoint, model_path)








