import torch
from torch import nn


class ADFSoftmax(nn.Module):

    def __init__(self, dim=1, min_variance=0.0001):
        super(ADFSoftmax, self).__init__()
        self.dim = dim
        self.min_variance = min_variance

    def keep_variance(self, x):
        return x + self.min_variance

    def forward(self, features_mean, features_variance, eps=1e-5):
        """Softmax function applied to a multivariate Gaussian distribution.
        It works under the assumption that features_mean and features_variance
        are the parameters of a the indepent gaussians that contribute to the
        multivariate gaussian.
        Mean and variance of the log-normal distribution are computed following
        https://en.wikipedia.org/wiki/Log-normal_distribution."""

        features_variance = self.keep_variance(features_variance)
        log_gaussian_mean = features_mean + 0.5 * features_variance
        log_gaussian_variance = 2 * log_gaussian_mean

        log_gaussian_mean = torch.exp(log_gaussian_mean)
        log_gaussian_variance = torch.exp(log_gaussian_variance)
        log_gaussian_variance = log_gaussian_variance * (
            torch.exp(features_variance) - 1)

        # now find the mean and variance of the denominator of the softmax
        denominator_mean = torch.sum(log_gaussian_mean,
                                     dim=self.dim,
                                     keepdim=True)
        denominator_variance = torch.sum(log_gaussian_variance,
                                         dim=self.dim,
                                         keepdim=True)
        # now approximate the mean and variance of this using taylor expansion approx.
        outputs_mean = log_gaussian_mean / (denominator_mean + eps)
        outputs_variance = outputs_mean ** 2.0 * (log_gaussian_variance +
                                                  denominator_variance +
                                                  eps)
        # again make sure output variance is kept
        outputs_variance = self.keep_variance(outputs_variance)
        return outputs_mean, outputs_variance
