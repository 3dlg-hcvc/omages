import os
import sys
import numpy as np
import scipy # should be version 1.11.1
import torch

from point_e.evals.feature_extractor import PointNetClassifier, get_torch_devices
from point_e.evals.fid_is import compute_statistics
from point_e.evals.npz_stream import NpzStreamer    

class PFID_evaluator():
    def __init__(self, devices=['cuda:0'], batch_size=256, cache_dir='~/.temp/PFID_evaluator'):
        self.__dict__.update(locals())
        cache_dir = os.path.expanduser(cache_dir)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.devices = [torch.device(d) for d in devices]
        self.clf = PointNetClassifier(devices=self.devices, cache_dir=cache_dir, device_batch_size=self.batch_size)

    def compute_pfid(self, pc_1, pc_2, return_feature=False):

        # print("computing first batch activations")
        # save clouds to npz files
        npz_path1 = os.path.join(self.cache_dir, "temp1.npz")
        npz_path2 = os.path.join(self.cache_dir, "temp2.npz")
        np.savez(npz_path1, arr_0=pc_1)
        np.savez(npz_path2, arr_0=pc_2)

        features_1, _ = self.clf.features_and_preds(NpzStreamer(npz_path1))
        stats_1 = compute_statistics(features_1)
        # print(features_1.max(), features_1.min(), features_1.mean(), features_1.std() )
        # print(stats_1.mu.shape, stats_1.sigma.shape)

        features_2, _ = self.clf.features_and_preds(NpzStreamer(npz_path2))
        stats_2 = compute_statistics(features_2)
        # print(features_2.max(), features_2.min(), features_2.mean(), features_2.std() )
        # print(stats_2.mu.shape, stats_2.sigma.shape)

        if return_feature:
            return features_1, features_2
        
        #PFID = stats_1.frechet_distance(stats_2) # same result as the next line
        PFID= frechet_distance(stats_1.mu, stats_1.sigma, stats_2.mu, stats_2.sigma)
        PKID = kernel_distance(features_1, features_2)

        print(f"P-FID: {PFID}", f"P-KID: {PKID}")
        return dict(PFID=PFID, PKID=PKID)


# from https://github.com/GaParmar/clean-fid/blob/main/cleanfid/fid.py
"""
Numpy implementation of the Frechet Distance.
The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
and X_2 ~ N(mu_2, C_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
Stable version by Danica J. Sutherland.
Params:
    mu1   : Numpy array containing the activations of a layer of the
            inception net (like returned by the function 'get_predictions')
            for generated samples.
    mu2   : The sample mean over activations, precalculated on an
            representative data set.
    sigma1: The covariance matrix over activations for generated samples.
    sigma2: The covariance matrix over activations, precalculated on an
            representative data set.
"""
def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


"""
Compute the KID score given the sets of features
"""
def kernel_distance(feats1, feats2, num_subsets=100, max_subset_size=1000):
    n = feats1.shape[1]
    m = min(min(feats1.shape[0], feats2.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = feats2[np.random.choice(feats2.shape[0], m, replace=False)]
        y = feats1[np.random.choice(feats1.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = t / num_subsets / m
    return float(kid)