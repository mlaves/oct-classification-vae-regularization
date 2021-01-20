#!/usr/bin/env python3

import torch
from torch.utils.data import DataLoader
import argparse
import tqdm
import numpy as np
from data_generator import KermanyDataset
from utils import accuracy
from models import ResNetClassifier, VariationalAutoEncoderModelShort, kld_loss, ae_loss


# enable cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Train solitary classifier on small OCT dataset for comparison.')
parser.add_argument('--model', metavar='M', type=str, help='Specify the model',
                    choices=['resnet', 'vae'])
parser.add_argument('--bs', metavar='N', type=int, help='Specify the batch size N', default=1,
                    choices=list(range(1, 65)))
parser.add_argument('--ls', metavar='S', type=int, help='Specify the latent size S', default=128,
                    choices=list(range(1, 1025)))
parser.add_argument('snapshot', metavar='W', type=str, help='Specify the snapshot')
args = parser.parse_args()

print("Eval model: " + str(args.model))
print("Eval with batch_size: " + str(args.bs))
print('')

# dimension properties
batch_size = args.bs
val_batch_size = batch_size
latent_size = args.ls
snapshot_filename = args.snapshot
num_classes = 4

color = True
resize_to = (224, 224)

# test on full train set
dataset_test = KermanyDataset("/home/laves/Downloads/OCT2017_3/test",
                              crop_to=(384, 384), resize_to=resize_to, color=color)
dataloader_test = DataLoader(dataset_test, batch_size=val_batch_size, shuffle=True)

assert len(dataset_test) > 0

print("Test dataset length:", len(dataset_test))
print('')

# create a model
if args.model == 'vae':
    model = VariationalAutoEncoderModelShort(num_classes=num_classes, latent_size=latent_size).to(device)
    try:
        checkpoint = torch.load(snapshot_filename)
        pretrained_dict = checkpoint['state_dict']
        model.load_state_dict(pretrained_dict)
        print("Loading pretrained weights from " + snapshot_filename)
    except FileNotFoundError as e:
        print(e)
        exit(-1)
else:
    try:
        model = ResNetClassifier(num_classes=num_classes).to(device)
        checkpoint = torch.load(snapshot_filename)
        pretrained_dict = checkpoint['state_dict']
        model.load_state_dict(pretrained_dict)
        print("Loading pretrained weights from " + snapshot_filename)
    except FileNotFoundError as e:
        print(e)
        exit(-1)

print('')  # print empty line before training output

# go through test set
print("Going through test set.")
# create loss function
criterion = torch.nn.CrossEntropyLoss()
model.eval()
results = np.zeros((len(dataloader_test), 2, 4))

with torch.no_grad():

    test_losses = []
    test_accuracies = []

    batches = tqdm.tqdm(dataloader_test)
    for i, (data, target) in enumerate(batches):
        data, target = data.to(device), target.to(device)

        if args.model == 'vae':
            output, means, log_vars, recon = model(data)
        else:
            output = model(data)

        test_loss = criterion(output, target)

        # sum epoch loss
        test_losses.append(test_loss.item())

        # calculate batch train accuracy
        batch_acc = accuracy(output, target)
        test_accuracies.append(batch_acc)

        # print current loss
        batches.set_description("l: {:4f}, %: {:4f}".format(np.mean(test_losses), np.mean(test_accuracies)))

        # save results
        results[i, 0, target.squeeze().data.cpu().numpy()] = 1.0
        results[i, 1] = output.squeeze().data.cpu().numpy()

print("test mean loss:", np.mean(test_losses))
print("test mean accuracies:", np.mean(test_accuracies))
np.save(f"results_test_{args.model}.npy", results)
