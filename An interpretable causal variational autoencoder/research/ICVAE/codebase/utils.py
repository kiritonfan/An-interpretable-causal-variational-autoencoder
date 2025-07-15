import numpy as np
import os
import shutil
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils import data
import torch.utils.data as Data
from torch.distributions.multivariate_normal import MultivariateNormal
from PIL import Image


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
bce = torch.nn.BCEWithLogitsLoss(reduction='none')
bce3 = torch.nn.BCELoss(reduction='none')


def mask_threshold(x):
    x = (x + 0.5).int().float()
    return x


def label_cov(labels):
    cov = torch.from_numpy(np.cov(labels, rowvar=False)).to(device)
    return cov


def get_labelcov_prior(batchsize, cov):
    v = torch.zeros(batchsize, cov.size()[0], cov.size()[1])
    for i in range(batchsize):
        v[i] = cov
    mean = torch.zeros(batchsize, cov.size()[1])
    return mean, v


def vector_expand(v):
    V = torch.zeros(v.size()[0], v.size()[1], v.size()[1]).to(device)
    for i in range(v.size()[0]):
        for j in range(v.size()[1]):
            V[i, j, j] = v[i, j]
    return V


def block_matmul(a, b):
    return None


def multivariate_sample(m, cov):
    m = m.reshape(m.size()[0], 4)
    z = torch.zeros(m.size())
    for i in range(z.size()[0]):
        z[i] = MultivariateNormal(m[i].cpu(), cov[i].cpu()).sample()
    return z.to(device)


def kl_multinormal_cov(qm, qv, pm, pv):
    KL = torch.zeros(qm.size()[0]).to(device)
    for i in range(qm.size()[0]):
        KL[i] = 0.5 * (torch.log(torch.det(pv[i])) - torch.log(torch.det(qv[i])) +
                       torch.trace(torch.inverse(pv[i])) * torch.trace(torch.inverse(qv[i])) +
                       torch.norm(qm[i]) * torch.norm(pv[i], p=1))
    return KL


def conditional_sample_gaussian(m, v):
    sample = torch.randn(m.size()).to(device)
    z = m + (v ** 0.5) * sample
    return z


def condition_gaussian_parameters(h, dim=1):
    m, h = torch.split(h, h.size(1) // 2, dim=1)
    m = torch.reshape(m, [-1, 3, 4])
    h = torch.reshape(h, [-1, 3, 4])
    v = F.softplus(h) + 1e-8
    return m, v


def condition_prior(scale, label, dim):
    label = label.to(device)
    
    batch_size = label.size()[0]
    z_dim = scale.size()[0]
    
    mean = torch.zeros((batch_size, z_dim, dim)).to(device)
    var = torch.ones((batch_size, z_dim, dim)).to(device)
    
    for i in range(z_dim):
        for j in range(batch_size):
            idx = torch.where(label[j][i] > 0)[0]
            if idx.size()[0] > 0:
                mul = scale[i][1]
            else:
                mul = scale[i][0]
            mean[j][i] = torch.ones(dim, device=device) * mul
            
    return mean, var


def bce2(r, x):
    return x * torch.log(r + 1e-7) + (1 - x) * torch.log(1 - r + 1e-7)


def sample_multivariate(cov, loc=None):
    latent_code = torch.distributions.multivariate_normal.MultivariateNormal(loc, covariance_matrix=cov,
                                                                             precision_matrix=None, scale_tril=None,
                                                                             validate_args=None)
    return latent_code


def get_covariance_matrix(A):
    assert A.size()[1] == A.size()[2]
    I = torch.zeros(A.size()).to(device)
    i = torch.eye(n=A.size()[1]).to(device)
    for j in range(A.size()[0]):
        I[j] = torch.inverse(torch.mm(torch.t((A[j] - i)), (A[j] - i)))

    return I


def sample_gaussian(m, v):
    """
    Samples from a Gaussian distribution: m + sqrt(v) * eps, where eps ~ N(0, 1).
    """
    v = torch.clamp(v, min=1e-8)
    epsilon = torch.randn_like(m)
    z = m + torch.sqrt(v) * epsilon
    return z


def log_normal(x, m, v):
    """
    Computes the elem-wise log probability of a Gaussian and then sums over the
    last dim. Basically we're assuming all dims are batch dims except for the
    last dim.
    """
    const = -0.5 * x.size(-1) * torch.log(2 * torch.tensor(np.pi))
    log_det = -0.5 * torch.sum(torch.log(v), dim=-1)
    log_exp = -0.5 * torch.sum((x - m) ** 2 / v, dim=-1)
    log_prob = const + log_det + log_exp
    return log_prob


def log_normal_mixture(z, m, v):
    """
    Computes log probability of a uniformly-weighted Gaussian mixture.
    """
    z = z.unsqueeze(1)
    log_probs = log_normal(z, m, v)
    log_prob = log_mean_exp(log_probs, 1)
    return log_prob


def gaussian_parameters(h, dim=-1):
    """
    Converts generic real-valued representations into mean and variance
    parameters of a Gaussian distribution
    """
    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v


def log_bernoulli_with_logits(x, logits):
    """
    Computes the log probability of a Bernoulli given its logits
    """
    print("x size in bernoulli logits:", x.size())
    log_prob = -bce(input=logits, target=x).sum(-1)
    return log_prob


def log_bernoulli_with_logits_nosigmoid(x, logits):
    """
    Computes the log probability of a Bernoulli given its logits
    """
    log_prob = bce3(logits, x).sum(-1)
    return log_prob


def kl_cat(q, log_q, log_p):
    element_wise = (q * (log_q - log_p))
    kl = element_wise.sum(-1)
    return kl


def kl_normal(qm, qv, pm, pv):
    """
    Computes the KL divergence between two normal distributions: KL[q || p].
    """
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv/pv + (qm-pm).pow(2)/pv - 1)
    element_wise = torch.clamp(element_wise, min=1e-8, max=1e8)
    
    if element_wise.dim() > 1:
        kl = element_wise.sum(dim=-1).mean()
    else:
        kl = element_wise.mean()
    
    if kl.dim() > 0:
        kl = kl.mean()
        
    return kl


def duplicate(x, rep):
    return x.expand(rep, *x.shape).reshape(-1, *x.shape[1:])


def log_mean_exp(x, dim):
    return log_sum_exp(x, dim) - np.log(x.size(dim))


def log_sum_exp(x, dim=0):
    max_x = torch.max(x, dim)[0]
    new_x = x - max_x.unsqueeze(dim).expand_as(x)
    return max_x + (new_x.exp().sum(dim)).log()


def load_model_by_name(model, global_step):
    file_path = os.path.join('checkpoints',
                             model.name,
                             'model-{:05d}.pt'.format(global_step))
    state = torch.load(file_path)
    model.load_state_dict(state)
    print("Loaded from {}".format(file_path))


def evaluate_lower_bound(model, labeled_test_subset, run_iwae=True):
    check_model = isinstance(model, CVAE) or isinstance(model, GMVAE) or isinstance(model, LVAE)
    assert check_model, "This function is only intended for VAE and GMVAE"

    print('*' * 80)
    print("LOG-LIKELIHOOD LOWER BOUNDS ON TEST SUBSET")
    print('*' * 80)

    xl, _ = labeled_test_subset
    torch.manual_seed(0)
    xl = torch.bernoulli(xl)

    def detach_torch_tuple(args):
        return (v.detach() for v in args)

    def compute_metrics(fn, repeat):
        metrics = [0, 0, 0]
        for _ in range(repeat):
            niwae, kl, rec = detach_torch_tuple(fn(xl))
            metrics[0] += niwae / repeat
            metrics[1] += kl / repeat
            metrics[2] += rec / repeat
        return metrics

    # Run multiple times to get low-var estimate
    nelbo, kl, rec = compute_metrics(model.negative_elbo_bound, 100)
    print("NELBO: {}. KL: {}. Rec: {}".format(nelbo, kl, rec))

    if run_iwae:
        for iw in [1, 10, 100, 1000]:
            repeat = max(100 // iw, 1)  # Do at least 100 iterations
            fn = lambda x: model.negative_iwae_bound(x, iw)
            niwae, kl, rec = compute_metrics(fn, repeat)
            print("Negative IWAE-{}: {}".format(iw, niwae))


def evaluate_classifier(model, test_set):
    check_model = isinstance(model, SSVAE)
    assert check_model, "This function is only intended for SSVAE"

    print('*' * 80)
    print("CLASSIFICATION EVALUATION ON ENTIRE TEST SET")
    print('*' * 80)

    X, y = test_set
    pred = model.cls.classify(X)
    accuracy = (pred.argmax(1) == y).float().mean()
    print("Test set classification accuracy: {}".format(accuracy))


def save_model_by_name(model, global_step):
    save_dir = os.path.join('checkpoints', model.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
    state = model.state_dict()
    torch.save(state, file_path)
    print('Saved to {}'.format(file_path))


def prepare_writer(model_name, overwrite_existing=False):
    log_dir = os.path.join('logs', model_name)
    save_dir = os.path.join('checkpoints', model_name)
    if overwrite_existing:
        delete_existing(log_dir)
        delete_existing(save_dir)
    writer = None
    return writer


def log_summaries(writer, summaries, global_step):
    pass


def delete_existing(path):
    if os.path.exists(path):
        print("Deleting existing path: {}".format(path))
        shutil.rmtree(path)


def reset_weights(m):
    try:
        m.reset_parameters()
    except AttributeError:
        pass


def get_mnist_data(device, use_test_subset=True):
    preprocess = transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=preprocess),
        batch_size=100,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=preprocess),
        batch_size=100,
        shuffle=True)

    X_train = train_loader.dataset.train_data.to(device).reshape(-1, 784).float() / 255
    y_train = train_loader.dataset.train_labels.to(device)
    X_test = test_loader.dataset.test_data.to(device).reshape(-1, 784).float() / 255
    y_test = test_loader.dataset.test_labels.to(device)

    X = X_test if use_test_subset else X_train
    y = y_test if use_test_subset else y_train

    xl, yl = [], []
    for i in range(10):
        idx = y == i
        idx_choice = get_mnist_index(i, test=use_test_subset)
        xl += [X[idx][idx_choice]]
        yl += [y[idx][idx_choice]]
    xl = torch.cat(xl).to(device)
    yl = torch.cat(yl).to(device)
    yl = yl.new(np.eye(10)[yl])
    labeled_subset = (xl, yl)

    return train_loader, labeled_subset, (X_test, y_test)


def get_mnist_index(i, test=True):
    train_idx = np.array([[2732, 2607, 1653, 3264, 4931, 4859, 5827, 1033, 4373, 5874],
                          [5924, 3468, 6458, 705, 2599, 2135, 2222, 2897, 1701, 537],
                          [2893, 2163, 5072, 4851, 2046, 1871, 2496, 99, 2008, 755],
                          [797, 659, 3219, 423, 3337, 2745, 4735, 544, 714, 2292],
                          [151, 2723, 3531, 2930, 1207, 802, 2176, 2176, 1956, 3622],
                          [3560, 756, 4369, 4484, 1641, 3114, 4984, 4353, 4071, 4009],
                          [2105, 3942, 3191, 430, 4187, 2446, 2659, 1589, 2956, 2681],
                          [4180, 2251, 4420, 4870, 1071, 4735, 6132, 5251, 5068, 1204],
                          [3918, 1167, 1684, 3299, 2767, 2957, 4469, 560, 5425, 1605],
                          [5795, 1472, 3678, 256, 3762, 5412, 1954, 816, 2435, 1634]])

    test_idx = np.array([[684, 559, 629, 192, 835, 763, 707, 359, 9, 723],
                         [277, 599, 1094, 600, 314, 705, 551, 87, 174, 849],
                         [537, 845, 72, 777, 115, 976, 755, 448, 850, 99],
                         [984, 177, 755, 797, 659, 147, 910, 423, 288, 961],
                         [265, 697, 639, 544, 543, 714, 244, 151, 675, 510],
                         [459, 882, 183, 28, 802, 128, 128, 53, 550, 488],
                         [756, 273, 335, 388, 617, 42, 442, 543, 888, 257],
                         [57, 291, 779, 430, 91, 398, 611, 908, 633, 84],
                         [203, 324, 774, 964, 47, 639, 131, 972, 868, 180],
                         [1000, 846, 143, 660, 227, 954, 791, 719, 909, 373]])

    if test:
        return test_idx[i]

    else:
        return train_idx[i]


def get_svhn_data(device):
    preprocess = transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN('data', split='extra', download=True, transform=preprocess),
        batch_size=100,
        shuffle=True)

    return train_loader, (None, None), (None, None)


def gumbel_softmax(logits, tau, eps=1e-8):
    U = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(U + eps) + eps)
    y = logits + gumbel
    y = F.softmax(y / tau, dim=1)
    return y


class DeterministicWarmup(object):
    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1 / n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t
        return self.t


class FixedSeed:
    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        np.random.set_state(self.state)