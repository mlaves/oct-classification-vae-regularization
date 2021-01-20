import torch
import erfnet
import torchvision.models


# enable cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ae_loss(recon_x, x):
    # loss = torch.nn.functional.binary_cross_entropy(recon_x, x)

    if recon_x.size(-1) != x.size(-1):
        recon_x = torch.nn.functional.interpolate(recon_x, size=(x.size(-2), x.size(-1)))
    loss = torch.nn.functional.mse_loss(recon_x, x)

    return loss


def kld_loss(mean, log_var):
    """
    see Appendix B from VAE paper:
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114
    0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    :param mean: vector of latent space mean values
    :param log_var: vector of latent space log variances
    :return: loss value
    """

    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return kld / mean.size(0)  # norm by batch size


class ResNetClassifier(torch.nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self._encoder = torchvision.models.resnet34(pretrained=True)

        # freeze pretrained layers
        # self._encoder.eval()
        # for param in self._encoder.parameters():
        #     param.requires_grad = False
        # replace last fc layer
        # self._encoder.fc = torch.nn.Linear(512, 1000)

        self._final_fc = torch.nn.Linear(1000, num_classes)

    def forward(self, x):
        y = self._encoder(x)
        y = self._final_fc(y)

        return y


class VariationalAutoEncoderModelShort(torch.nn.Module):

    def __init__(self, num_classes, in_channels=1, out_channels=1, latent_size=256):
        super().__init__()

        self._latent_size = latent_size
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._num_classes = num_classes

        self._encoder = torchvision.models.resnet34(pretrained=True)

        self._classifier = torch.nn.Linear(self._latent_size, self._num_classes)
        self._linear_means = torch.nn.Linear(1000, self._latent_size)
        self._linear_log_vars = torch.nn.Linear(1000, self._latent_size)

        self._decoder1 = torch.nn.Linear(self._latent_size + self._num_classes, 4096)
        self._decoder2 = erfnet.Decoder2(out_channels=3)

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def freeze_resnet(self):
        # freeze pretrained layers
        self._encoder.eval()
        for param in self._encoder.parameters():
            param.requires_grad = False

        # replace last fc layer
        self._encoder.fc = torch.nn.Linear(512, 1000)

    def unfreeze_resnet(self):
        # freeze pretrained layers
        self._encoder.train()
        for param in self._encoder.parameters():
            param.requires_grad = True

    def encoder(self, x):
        """
        Combination of encoder and classifier.

        :param x: Input tensor of images.
        :return: (y, means, log_vars) tuple of class label predictions and latent encoding of input images.
        """
        h = self._encoder(x)

        means = self._linear_means(h)
        log_vars = self._linear_log_vars(h)

        z = self.reparameterize(means, log_vars)

        if self.training:
            y = self._classifier(z)
        else:
            y = self._classifier(means)

        return y, z, means, log_vars

    def decoder(self, y, z):
        decoder1_input = torch.cat([y, z], dim=1)
        x_dec1 = self._decoder1(decoder1_input)
        x_dec1 = torch.nn.functional.relu(x_dec1)
        x_dec1 = x_dec1.view(-1, 64, 8, 8)
        x_recon = self._decoder2(x_dec1)
        x_recon = torch.sigmoid(x_recon)

        return x_recon

    def forward(self, x):
        y, z, means, log_vars = self.encoder(x)

        print(y, z)
        exit()

        x_recon = self.decoder(y, z)

        return y, means, log_vars, x_recon
