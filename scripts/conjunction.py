"""
Use the approach in Heller et al. nimg 2007 to perform conjunction inference
"""
import numpy as np
from scipy.stats import norm, t


def _conjunction_inference_from_z_values(Z, u=.5, verbose=False):
    """ Returns a conjunction z-values against the null hypothesis:
    A proportion of less than u values per row of Z display an effect
    Stouffer approach is used.

    Parameters
    ----------
    Z: arraz of shape (n_tests, n)
       Presumably z values
    u: int in [0, 1[
       desired proportion level
    """
    if (u >= 1) or (u < 0):
        raise ValueError("u should be between 0 and 1")
    p = int((1 - u) * Z.shape[1])
    Z_ = np.sort(Z, 1)
    if verbose:
        print(p)
    return np.sum(Z_[:, :p], 1) / np.sqrt(p)


def conjunction_inference_from_z_images(z_images, masker, u=.5):
    """ Mass univariate test for the null hypothesis:
    A proportion of less than u values per row of Z display an effect
    Stouffer approach is used.

    Parameters
    -----------
    z_images: nims,
              Z images provided as input
    u: int in [0, 1[
       desired proportion level
    """
    Z = masker.transform(z_images)
    conj = _conjunction_inference_from_z_values(Z, u)
    conj_img = masker.inverse_transform(conj)
    return conj_img


def example(shape=(10, 10), n_samples=20):
    x = np.random.randn(shape[0], shape[1], n_samples)
    x[1:4, 2:5] += 1.
    for i in range(n_samples):
        x[:, :, i] *= 10 * np.random.rand(1)
    X = np.reshape(x, (shape[0] * shape[1], n_samples))
    one_sample = X.mean(1) / X.std(1) * np.sqrt(n_samples - 1)
    rfx = norm.isf(t.sf(one_sample, n_samples - 1)).reshape(shape)
    conj = _conjunction_inference_from_z_values(X).reshape(shape)
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    plt.figure()
    plt.subplot(121)
    plt.imshow(rfx, interpolation='nearest', cmap=cm.bwr, vmin=0, vmax=4)
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(conj, interpolation='nearest', cmap=cm.bwr, vmin=0, vmax=4)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    z = np.random.randn(10, 20)
    print(_conjunction_inference_from_z_values(z))
    z[:, :10] += 10
    print(_conjunction_inference_from_z_values(z))
    print(_conjunction_inference_from_z_values(z, 0.45))
    print(_conjunction_inference_from_z_values(z, 0.25))
    example()
