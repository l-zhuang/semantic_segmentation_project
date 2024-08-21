import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

import transforms as ext_transforms
#from args import get_arguments
from data.utils import enet_weighing, median_freq_balancing
import utils
# from data import Kitti as dataset
from data.camvid import CamVid as dataset
# Get the arguments
# args = get_arguments()
# from models.enet import ENet
from models.enet import ENet
from train import Train
from metric.iou import IoU
from test import Test


# Arguments
height=512
width=512
dataset_dir='./'
batch_size=10
workers=4
mode='train'
imshow_batch=True
weighing='Enet'

class_weights=0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Define transforms that would convert images and labels to Tensor
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



imshow_batch=True
class_encoding = test_set.color_encoding
# Get number of classes to predict
del class_encoding['road_marking']

num_classes = len(class_encoding)
class_weights = enet_weighing(test_loader, num_classes)
class_weights = torch.from_numpy(class_weights).float().to(device)

ignore_unlabeled=True
# Set the weight of the unlabeled class to 0
if ignore_unlabeled:
    ignore_index = list(class_encoding).index('unlabeled')
    class_weights[ignore_index] = 0

print("Class weights:", class_weights)

# Visualize a batch of test examples
images, labels = next(iter(test_loader))
print("Image size:", images.size())
print("Label size:", labels.size())

print("Close the figure window to continue...")

# Convert labels tensor to rgb PIL images
label_to_rgb = transforms.Compose([
    ext_transforms.LongTensorToRGBPIL(class_encoding),
    transforms.ToTensor()
])
color_labels = utils.batch_transform(labels, label_to_rgb)
utils.imshow_batch(images, color_labels)

############################
pth_name='./run/model_original_200.pth'

# Load Saved Model Checkpoint
model = ENet(num_classes)
checkpoint = torch.load(pth_name)
model.load_state_dict(checkpoint['state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer'])

# Make predictions!
model.eval()
with torch.no_grad():
    predictions = model(images)

# Predictions is one-hot encoded with "num_classes" channels.
# Convert it to a single int using the indices where the maximum (1) occurs

_, predictions = torch.max(predictions.data, 1)

label_to_rgb = transforms.Compose([
    ext_transforms.LongTensorToRGBPIL(class_encoding),
    transforms.ToTensor()
])
color_predictions = utils.batch_transform(predictions, label_to_rgb)
utils.imshow_batch(images.data.cpu(), color_predictions)


#### Evaluate Test Set  ####


ignore_unlabeled=True
# Evaluation metric
if ignore_unlabeled:
    ignore_index = list(class_encoding).index('unlabeled')
else:
    ignore_index = None

metric = IoU(num_classes, ignore_index=ignore_index)
criterion_entropy = nn.CrossEntropyLoss(weight=class_weights)

# Evaluate Test Set
model=model.to('cuda')

train= Test(model, train_loader, criterion_entropy, metric, device)
epoch_loss_train, (iou_train, miou_train) = train.run_epoch(False)

val = Test(model, val_loader, criterion_entropy, metric, device)
epoch_loss_val, (iou_val, miou_val) = val.run_epoch(False)

test = Test(model, test_loader, criterion_entropy, metric, device)
epoch_loss_test, (iou_test, miou_test) = test.run_epoch(False)

# print(type(miou_test))
# print(type(iou_test))

print("Final Training loss: {0:.4f} | Mean IoU: {1:.4f}".format(epoch_loss_train, miou_train))
print("Final Validation loss: {0:.4f} | Mean IoU: {1:.4f}".format(epoch_loss_val, miou_val))
print("Final Testing loss: {0:.4f} | Mean IoU: {1:.4f}".format(epoch_loss_test, miou_test))

classesname=list(class_encoding.keys())
color_list = list(class_encoding.values())
color_list_normalized = [(r / 255, g / 255, b / 255, 1) for r, g, b in color_list]

# print('Train IoU')
# for i,iou in enumerate(iou_train):
#     print(f'{classesname[i]}: {iou:.4f}')
# print('Valid IoU')
# for i,iou in enumerate(iou_val):
#     print(f'{classesname[i]}: {iou:.4f}')    
# print('Test IoU')
# for i,iou in enumerate(iou_test):
#     print(f'{classesname[i]}: {iou:.4f}')

import matplotlib.pyplot as plt 
# Plotting the bar chart
def plot_bar(iou,miou,path):
    plt.bar(classesname, iou, color=color_list_normalized)
    plt.xlabel('Classes')
    plt.ylabel('IoU')
    # Rotating x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.title(f'IoU for Each Class (MIoU = {miou:.4f})')
    # plt.savefig(path)
    plt.show()

plot_bar(iou_train,miou_train,'./pic/iou_train_mish_200.png')
plot_bar(iou_val,miou_val,'./pic/iou_valid_mish_200.png')
plot_bar(iou_test,miou_test,'./pic/iou_test_mish_200.png')

print(iou_test)



