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
from models import VariationalAutoEncoderModelShort
from models import ae_loss, kld_loss
import os
from tensorboardX import SummaryWriter
from utils import accuracy
import datetime


# enable cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
start_time = datetime.datetime.now().replace(microsecond=0).isoformat().replace(':', '-')


parser = argparse.ArgumentParser(description='Learn both from labeled and unlabeled data.')
parser.add_argument('--model', metavar='M', type=str, help='Specify the autoencoder model',
                    choices=['vae'])
parser.add_argument('--alpha', metavar='A', type=float, help='Specify weight of the reconstruction weight A',
                    default=1.0)
parser.add_argument('--beta', metavar='B', type=float, help='Specify weight of the latent space regularizer loss B',
                    default=1e-1)
parser.add_argument('--gamma', metavar='G', type=float, help='Specify weight of the classifier G',
                    default=1.0)
parser.add_argument('--bs', metavar='N', type=int, help='Specify the batch size N', default=1,
                    choices=list(range(1, 129)))
parser.add_argument('--epochs', metavar='E', type=int, help='Specify the number of epochs E', default=50,
                    choices=list(range(1, 201)))
args = parser.parse_args()

print("Train autoencoder variant: " + str(args.model))
print("Train with alpha: " + str(args.alpha))
print("Train with beta: " + str(args.beta))
print("Train with gamma: " + str(args.gamma))
print("Train with batch_size: " + str(args.bs))
print("Training epochs: " + str(args.epochs))
print('')

out_path = "./out_" + args.model + "/"

# create out path
if not os.path.exists(out_path):
    os.mkdir(out_path)

# setup tensorboardx
writer = SummaryWriter()

# dimension properties
batch_size = args.bs
val_batch_size = args.bs
num_classes = 4
latent_size = 256

# train on train (large unlabeled) and test (small labeled), eval on val, final eval on whole train set
dataset_train = KermanyDataset("/home/laves/Downloads/OCT2017/train2",
                               crop_to=(384, 384), resize_to=(299, 299), color=False)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataset_valid = KermanyDataset("/home/laves/Downloads/OCT2017/test",
                               crop_to=(384, 384), resize_to=(299, 299), color=False)
dataloader_valid = DataLoader(dataset_valid, batch_size=val_batch_size, shuffle=True)
dataset_test = KermanyDataset("/home/laves/Downloads/OCT2017/train",
                              crop_to=(384, 384), resize_to=(299, 299), color=False)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

assert len(dataset_train) > 0
assert len(dataset_valid) > 0
assert len(dataset_test) > 0

print("Train dataset length:", len(dataset_train))
print("Valid dataset length:", len(dataset_valid))
print("Test dataset length:", len(dataset_test))

# create a model
if args.model == 'vae':
    model = VariationalAutoEncoderModelShort(
        num_classes=num_classes,
        out_channels=1,
        latent_size=latent_size).to(device)
else:
    raise ValueError("specified model not available")

print("latent space size:", latent_size)

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
criterion_recon = ae_loss
criterion_classifier = torch.nn.CrossEntropyLoss()
criterion_latent_reg = kld_loss

alpha = args.alpha  # reconstruction weight
beta = args.beta  # latent space regularizer weight
gamma = args.gamma  # classification weight

print("Training from scratch.")
# model.init_weights()  # init with xavier

print('')  # print empty line before training output

# save accuracies and losses during training
train_losses = []
valid_losses = []
valid_classifier_losses = []
train_accuracies = []
valid_accuracies = []
start_epoch = 0
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
        epoch_discriminator_loss = []
        is_best = False

        batches_train = tqdm.tqdm(dataloader_train)
        for x_train, y_train in batches_train:

            x_train, y_train = x_train.to(device), y_train.to(device)

            # first, train on labeled data
            optimizer.zero_grad()

            y_predict, means, log_vars, x_recon = model(x_train)
            ls_reg_loss = criterion_latent_reg(means, log_vars)

            loss_classifier = criterion_classifier(y_predict, y_train)
            loss_recon = criterion_recon(x_recon, x_train)

            train_loss = gamma * loss_classifier + alpha * loss_recon + beta * ls_reg_loss

            train_loss.backward()
            optimizer.step()  # Does the update

            # sum epoch loss
            epoch_train_loss.append(train_loss.item())

            # calculate train accuracy
            train_acc = accuracy(y_predict, y_train)
            epoch_train_acc.append(train_acc)

            writer.add_scalar('data/train_classifier_loss', loss_classifier.item(), batch_counter)
            writer.add_scalar('data/train_recon_loss', loss_recon.item(), batch_counter)
            writer.add_scalar('data/train_ls_reg_loss', ls_reg_loss.item(), batch_counter)
            writer.add_scalar('data/train_loss', train_loss.item(), batch_counter)

            batch_counter += 1

            # print current loss and acc
            batches_train.set_description(
                "loss: {:4f}, acc: {:4f}".format(np.mean([test_loss.item(), train_loss.item()]),
                                                 np.mean([test_acc, train_acc])))
            del x_train, y_train, y_predict, x_recon, loss_classifier, loss_recon, ls_reg_loss

        epoch_train_loss = np.mean(epoch_train_loss)
        epoch_train_acc = np.mean(epoch_train_acc)
        train_accuracies.append(epoch_train_acc)

        # go through validation set
        model.eval()

        epoch_valid_loss = []
        epoch_valid_classifier_loss = []
        epoch_valid_acc = []
        epoch_discriminator_loss_valid = []

        with torch.no_grad():
            batches = tqdm.tqdm(dataloader_valid)
            for x, y in batches:
                x, y = x.to(device), y.to(device)

                if args.model in ['ae', 'dae']:
                    y_predict, z, x_recon = model(x)
                    ls_reg_loss = criterion_latent_reg(z)
                elif args.model in ['vae', 'dvae']:
                    y_predict, means, log_vars, x_recon = model(x)
                    ls_reg_loss = criterion_latent_reg(means, log_vars)

                loss_classifier = criterion_classifier(y_predict, y)
                loss_recon = criterion_recon(x_recon, x)

                valid_loss = gamma * loss_classifier + alpha * loss_recon + beta * ls_reg_loss

                # print current loss
                batches.set_description("loss: {:4f}".format(valid_loss.item()))

                # sum epoch loss
                epoch_valid_loss.append(valid_loss.item())
                epoch_valid_classifier_loss.append(loss_classifier.item())
                valid_acc = accuracy(y_predict, y)
                epoch_valid_acc.append(valid_acc)

                writer.add_scalar('data/valid_classifier_loss', loss_classifier.item(), batch_counter_valid)
                writer.add_scalar('data/valid_recon_loss', loss_recon.item(), batch_counter_valid)
                writer.add_scalar('data/valid_ls_reg_loss', ls_reg_loss.item(), batch_counter_valid)
                writer.add_scalar('data/valid_loss', valid_loss.item(), batch_counter_valid)
                writer.add_scalar('data/valid_acc', valid_acc, batch_counter_valid)
                batch_counter_valid += 1

        epoch_valid_loss = np.mean(epoch_valid_loss)
        epoch_valid_classifier_loss = np.mean(epoch_valid_classifier_loss)
        epoch_valid_acc = np.mean(epoch_valid_acc)

        lr_scheduler.step(epoch_train_loss)

        print("Epoch {:d}: loss: {:4f}, acc: {:4f}, val_loss: {:4f}, val_acc: {:4f}"
              .format(e,
                      epoch_train_loss,
                      epoch_train_acc,
                      epoch_valid_loss,
                      epoch_valid_acc,
                      ))

        plt.figure(figsize=(5, 10))
        num_subplots = x_recon.size(0) if x_recon.size(0) <= 10 else 10
        for i in range(num_subplots):
            plt.subplot(5, 2, i + 1)
            plt.imshow(x_recon.data.cpu().numpy()[i, 0])
            plt.title(str(y.data.cpu().numpy()[i]))
            plt.axis('off')
        plt.savefig("./" + out_path + "/{}_train.png".format(e), dpi=300)
        plt.clf()
        plt.close()

        plt.figure(figsize=(5, 10))
        for i in range(num_subplots):
            plt.subplot(5, 2, i + 1)
            plt.imshow(x.data.cpu().numpy()[i, 0])
            plt.title(str(y.data.cpu().numpy()[i]))
            plt.axis('off')
        plt.savefig("./" + out_path + "/{}_input.png".format(e), dpi=300)
        plt.clf()
        plt.close()

        # save epoch losses
        train_losses.append(epoch_train_loss)
        valid_losses.append(epoch_valid_loss)
        valid_classifier_losses.append(epoch_valid_classifier_loss)

        valid_accuracies.append(epoch_valid_acc)

        # keep weights if mean loss is best
        if np.mean([train_losses[-1], valid_losses[-1]]) <= np.min(np.mean([train_losses, valid_losses], axis=0)):
            is_best = True

        if is_best:
            filename = f"../../snapshots/erfnet_ae2_{args.model}_best_{start_time}.pth.tar"
            print("Saving best weights so far with mean loss: {:4f}"
                  .format(np.mean([train_losses[-1], valid_losses[-1]])))
            torch.save({
                'epoch': e,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_losses': train_losses,
                'valid_losses': valid_losses,
                'train_acc': train_accuracies,
                'valid_acc': valid_accuracies
            }, filename)

        save_at_epoch = True
        if save_at_epoch and e == epochs-1:
            filename = f"../../snapshots/erfnet_ae2_{args.model}_{e}_{start_time}.pth.tar"
            print("Saving weights at epoch {:d}".format(e))
            torch.save({
                'epoch': e,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_losses': train_losses,
                'valid_losses': valid_losses,
                'train_acc': train_accuracies,
                'valid_acc': valid_accuracies
                }, filename)

        print('')

        # plot losses
        plt.figure()
        plt.plot(range(len(train_losses)), train_losses, marker='x', label='L_train')
        plt.plot(range(len(valid_losses)), valid_losses, marker='x', label='L_valid')
        plt.plot(range(len(valid_classifier_losses)), valid_classifier_losses, marker='x', label='L_c,valid')
        plt.title("loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig("loss_" + args.model + ".pdf", dpi=300)
        plt.close('all')

        # plot accuracy
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
    filename = f"../../snapshots/erfnet_ae2_{args.model}_{e}_{start_time}.pth.tar"
    print("Saving weights at epoch {:d}".format(e))
    torch.save({
        'epoch': e,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'train_acc': train_accuracies,
        'valid_acc': valid_accuracies
    }, filename)

# go through test set
print(f"Going through test set for {args.model}.")
with torch.no_grad():
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    test_losses = []
    test_accuracies = []

    try:
        batches = tqdm.tqdm(dataloader_train)
        for data, target in batches:
            data, target = data.to(device), target.to(device)

            output = model(data)
            y_pred = output[0]
            test_loss = criterion(y_pred, target)

            # sum epoch loss
            test_losses.append(test_loss.item())

            # calculate batch train accuracy
            batch_acc = accuracy(y_pred, target)
            test_accuracies.append(batch_acc)

            # print current loss and acc
            batches.set_description(
                "loss: {:4f}, acc: {:4f}".format(test_loss.item(), np.mean(test_accuracies)))
    except KeyboardInterrupt:
        pass

    print(f"{args.model} test mean loss:", np.mean(test_losses))
    print(f"{args.model} test mean accuracies:", np.mean(test_accuracies))
