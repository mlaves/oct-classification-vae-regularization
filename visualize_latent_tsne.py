#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import argparse
import tqdm
import numpy as np
from data_generator import KermanyDataset
from sklearn.manifold import TSNE
from models import VariationalAutoEncoderModelShort


# enable cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# some preparations for using LaTeX
plt.rcParams['text.latex.preamble'] = [r"\usepackage{times}"]
params = {'text.usetex': True,
          'font.size': 8,
          'font.family': 'serif',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)


def main(batch_size=1):
    # dimension properties
    num_classes = 4

    # datasets and dataloaders for CamVid train and valid
    dataset_train = KermanyDataset("/home/laves/Downloads/OCT2017_3/train",
                                   crop_to=(384, 384), resize_to=(224, 224), color=True)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

    assert len(dataset_train) > 0

    print("Dataset length:", len(dataset_train))

    # create a model
    net = VariationalAutoEncoderModelShort(num_classes=num_classes, latent_size=128).to(device)

    try:
        filename = "./snapshots/vae_best.pth.tar"
        checkpoint = torch.load(filename)
        pretrained_dict = checkpoint['state_dict']
        net.load_state_dict(pretrained_dict)
        print("Loading pretrained weights from " + filename)
    except FileNotFoundError as e:
        print(e)
        exit(-1)

    net.eval()

    # loop over the dataset and collect latent space representations
    latent_space = []
    latent_space_labels = []
    batches = tqdm.tqdm(dataloader_train)
    for x, target in batches:
        x = x.to(device)

        _, z, log_vars, _ = net(x)

        z = z.data.cpu().numpy()
        target = target.data.cpu().numpy()

        for i in range(target.shape[0]):
            latent_space.append(z[i])
            latent_space_labels.append(target[i])

    latent_space = np.array(latent_space)[::1]  # change slice to make it more sparse
    latent_space_labels = np.array(latent_space_labels)[::1]
    ls_embedded = TSNE().fit_transform(latent_space)

    plt.figure(figsize=(4, 3))
    plt.scatter(
        ls_embedded[:, 0], ls_embedded[:, 1],
        c=latent_space_labels,
        s=2.0, cmap=plt.get_cmap('viridis', num_classes)
        )
    cbar = plt.colorbar(ticks=np.arange(0, num_classes-1, (num_classes-1)/num_classes)+(num_classes-1)/(2*num_classes))
    cbar.ax.set_yticklabels(['CNV', 'DME', 'DRUSEN', 'NORMAL'])
    # plt.title("latent space t-SNE projection")
    plt.xlabel(r'dimension 1')
    plt.ylabel(r'dimension 2')
    plt.tight_layout()
    plt.savefig("ls_tsne_proj_vae.pdf", dpi=300)

    plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train different networks for semantic segmentation.')
    parser.add_argument('--bs', metavar='N', type=int, help='Specify the batch size N', default=1,
                        choices=list(range(1, 65)))
    args = parser.parse_args()
    main(batch_size=args.bs)
