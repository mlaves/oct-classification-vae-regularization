#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')  # headless plotting
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import tqdm
import numpy as np
from data_generator import KermanyDataset
from utils import accuracy
from tensorboardX import SummaryWriter
from models import ResNetClassifier, VariationalAutoEncoderModelShort, kld_loss, ae_loss


# enable cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Train solitary classifier on small OCT dataset for comparison.')
parser.add_argument('--model', metavar='M', type=str, help='Specify the autoencoder model',
                    choices=['resnet', 'vae'])
parser.add_argument('--bs', metavar='N', type=int, help='Specify the batch size N', default=1,
                    choices=list(range(1, 65)))
parser.add_argument('--epochs', metavar='E', type=int, help='Specify the number of epochs E', default=100,
                    choices=list(range(1, 201)))
parser.add_argument('--ls', metavar='S', type=int, help='Specify the latent size S', default=128,
                    choices=list(range(1, 1025)))
args = parser.parse_args()

print("Train model: " + str(args.model))
print("Train with batch_size: " + str(args.bs))
print("Training epochs: " + str(args.epochs))
print("Train with latent_size: " + str(args.ls))
print('')

# setup tensorboardx
writer = SummaryWriter()

# dimension properties
batch_size = args.bs
val_batch_size = batch_size
latent_size = args.ls
num_classes = 4

color = True
resize_to = (224, 224)

dataset_train = KermanyDataset("/home/laves/Downloads/OCT2017_3/train",
                               crop_to=(384, 384), resize_to=resize_to, color=color)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

dataset_valid = KermanyDataset("/home/laves/Downloads/OCT2017_3/val",
                               crop_to=(384, 384), resize_to=resize_to, color=color)
dataloader_valid = DataLoader(dataset_valid, batch_size=val_batch_size)

# test on full train set
dataset_test = KermanyDataset("/home/laves/Downloads/OCT2017_3/test",
                              crop_to=(384, 384), resize_to=resize_to, color=color)
dataloader_test = DataLoader(dataset_test, batch_size=val_batch_size)

assert len(dataset_train) > 0
assert len(dataset_valid) > 0
assert len(dataset_test) > 0

print("Train dataset length:", len(dataset_train))
print("Valid dataset length:", len(dataset_valid))
print("Test dataset length:", len(dataset_test))
print('')

# create a model
if args.model == 'vae':
    model = VariationalAutoEncoderModelShort(num_classes=num_classes, latent_size=latent_size)
    model.freeze_resnet()
    model.to(device)
else:
    model = ResNetClassifier(num_classes=num_classes).to(device)

# calculate number of trainable parameters
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Total trainable parameters: {:,}".format(params))

# create your optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,
                             betas=(0.9, 0.999),
                             weight_decay=1e-8)
lr_scheduler = ReduceLROnPlateau(optimizer, patience=5)

# create loss function
criterion = torch.nn.CrossEntropyLoss()

start_epoch = 0

print('')  # print empty line before training output

# save accuracies and losses during training
train_losses = []
train_accuracies = []
valid_losses = []
valid_accuracies = []
epochs = args.epochs
e = 0
batch_counter = 0
batch_counter_valid = 0

try:
    for e in range(start_epoch, epochs):  # loop over the dataset multiple times

        # go through training set
        model.train()
        print("lr =", optimizer.param_groups[0]['lr'])

        epoch_train_loss = []
        epoch_train_acc = []
        is_best = False

        data, target, output = None, None, None
        batches = tqdm.tqdm(dataloader_train)
        for data, target in batches:
            data, target = data.to(device), target.to(device)

            # in your training loop:
            optimizer.zero_grad()  # zero the gradient buffers
            if args.model == 'vae':
                output, means, log_vars, recon = model(data)
                ls_reg_loss = kld_loss(means, log_vars)

                loss_classifier = criterion(output, target)
                loss_recon = ae_loss(recon, data)

                train_loss = 1.0 * loss_classifier + 0.1 * loss_recon + 0.1 * ls_reg_loss

                train_loss.backward()
                optimizer.step()  # Does the update

                writer.add_scalar('data/train_loss_recon', loss_recon.item(), batch_counter)
                writer.add_scalar('data/train_ls_reg_loss', ls_reg_loss.item(), batch_counter)
            else:
                output = model(data)
                train_loss = criterion(output, target)
                train_loss.backward()
                optimizer.step()  # Does the update

            # print current loss
            batches.set_description("loss: {:4f}".format(train_loss.item()))

            # sum epoch loss
            epoch_train_loss.append(train_loss.item())

            # calculate batch train accuracy
            batch_acc = accuracy(output, target)
            epoch_train_acc.append(batch_acc)

            writer.add_scalar('data/train_classifier_loss', train_loss.item(), batch_counter)
            writer.add_scalar('data/train_acc', batch_acc, batch_counter)
            batch_counter += 1

        epoch_train_loss = np.mean(epoch_train_loss)
        epoch_train_acc = np.mean(epoch_train_acc)

        # go through validation set
        model.eval()
        with torch.no_grad():

            epoch_valid_loss = []
            epoch_valid_acc = []

            batches = tqdm.tqdm(dataloader_valid)
            for data, target in batches:
                data, target = data.to(device), target.to(device)

                if args.model == 'vae':
                    output, means, log_vars, recon = model(data)
                    ls_reg_loss = kld_loss(means, log_vars)

                    loss_classifier = criterion(output, target)
                    loss_recon = ae_loss(recon, data)

                    valid_loss = 1.0 * loss_classifier + 0.1 * loss_recon + 0.1 * ls_reg_loss

                    writer.add_scalar('data/valid_loss_recon', loss_recon.item(), batch_counter_valid)
                    writer.add_scalar('data/valid_ls_reg_loss', ls_reg_loss.item(), batch_counter_valid)
                else:
                    output = model(data)
                    valid_loss = criterion(output, target)

                # print current loss
                batches.set_description("loss: {:4f}".format(valid_loss.item()))

                # sum epoch loss
                epoch_valid_loss.append(valid_loss.item())

                # calculate batch train accuracy
                batch_acc = accuracy(output, target)
                epoch_valid_acc.append(batch_acc)

                writer.add_scalar('data/valid_classifier_loss', valid_loss.item(), batch_counter_valid)
                writer.add_scalar('data/valid_acc', batch_acc, batch_counter_valid)
                batch_counter_valid += 1

        epoch_valid_loss = np.mean(epoch_valid_loss)
        epoch_valid_acc = np.mean(epoch_valid_acc)

        print("Epoch {:d}: loss: {:4f}, acc: {:4f}, val_loss: {:4f}, val_acc: {:4f}"
              .format(e,
                      epoch_train_loss,
                      epoch_train_acc,
                      epoch_valid_loss,
                      epoch_valid_acc,
                      ))

        # save epoch losses
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        valid_losses.append(epoch_valid_loss)
        valid_accuracies.append(epoch_valid_acc)

        if valid_losses[-1] <= np.min(valid_losses):
            is_best = True

        if is_best:
            filename = "./snapshots/" + args.model + "_best.pth.tar"
            print("Saving best weights so far with val_loss: {:4f}".format(valid_losses[-1]))
            torch.save({
                'epoch': e,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_losses': train_losses,
                'valid_losses': valid_losses,
            }, filename)

        save_at_epoch = True
        if save_at_epoch and e == epochs-1:
            filename = "./snapshots/" + args.model + "_" + str(e) + ".pth.tar"
            print("Saving weights at epoch {:d}".format(e))
            torch.save({
                'epoch': e,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_losses': train_losses,
                'train_accs': train_accuracies,
                'val_losses': valid_losses,
                'val_accs': valid_accuracies,
                }, filename)

        # schedule lr after epoch 9
        if e >= 19:
            lr_scheduler.step(epoch_train_loss)

        # finetune resnet part after some training of vae part
        if e == 19 and args.model == 'vae':
            model.unfreeze_resnet()
            print('')
            print('unfreezing resnet')
            # calculate number of trainable parameters
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("New count of trainable parameters: {:,}".format(params))

        print('')

        # plot losses
        plt.figure()
        plt.plot(range(len(train_losses)), train_losses, marker='x')
        plt.plot(range(len(valid_losses)), valid_losses, marker='x')
        plt.title("loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig("loss_" + args.model + ".pdf", dpi=300)
        plt.figure()
        plt.plot(range(len(train_accuracies)), train_accuracies, marker='x')
        plt.plot(range(len(valid_accuracies)), valid_accuracies, marker='x')
        plt.title("accuracy")
        plt.xlabel("epoch")
        plt.ylabel("acc")
        plt.savefig("acc_" + args.model + ".pdf", dpi=300)
        plt.close('all')

except KeyboardInterrupt:
    print("Caught keyboard interrupt, quitting...")
    filename = "./snapshots/" + args.model + "_" + str(e) + ".pth.tar"
    print("Saving weights at epoch {:d}".format(e))
    torch.save({
        'epoch': e,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_accs': train_accuracies,
        'val_losses': valid_losses,
        'val_accs': valid_accuracies,
    }, filename)

# go through test set
print("Going through test set.")
model.eval()
with torch.no_grad():

    test_losses = []
    test_accuracies = []

    batches = tqdm.tqdm(dataloader_test)
    for data, target in batches:
        data, target = data.to(device), target.to(device)

        if args.model == 'vae':
            output, means, log_vars, recon = model(data)
        else:
            output = model(data)

        test_loss = criterion(output, target)

        # print current loss
        batches.set_description("loss: {:4f}".format(test_loss.item()))

        # sum epoch loss
        test_losses.append(test_loss.item())

        # calculate batch train accuracy
        batch_acc = accuracy(output, target)
        test_accuracies.append(batch_acc)

print("test mean loss:", np.mean(test_losses))
print("test mean accuracies:", np.mean(test_accuracies))
