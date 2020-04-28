"""
Template for Question 2 of hwk3.
@author: Samuel Lavoie
"""
import torch
import q2_sampler
import q2_model


def lp_reg(x, y, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. Also consider that the norm used is the L2 norm. This is important to consider,
    because we make the assumption that your implementation follows this notation when testing your function. ***

    :param x: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution P.
    :param y: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution Q.
    :param critic: (Module) - torch module that you want to regularize.
    :return: (FloatTensor) - shape: (1,) - Lipschitz penalty
    """
    unif = torch.distributions.uniform.Uniform(0., 1.)
    t = unif.rsample((x.shape[0], 1))
    #x.data = x.data + gaussian.sample()
    xhat = x*t + (1.-t)*y
    xhat.requires_grad = True

    #z = x + torch.rand_like(x)
    #x.requires_grad = True
    critic_x = critic(xhat)

    g = torch.autograd.grad(torch.sum(critic_x), xhat, create_graph=True)
    norm = torch.norm(g[0], p=2, dim=1)
    one = torch.ones_like(norm)
    maxi = torch.max(torch.zeros_like(one), norm-one)
    return torch.mean(maxi*maxi)



def vf_wasserstein_distance(x, y, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. This is important to consider, because we make the assuption that your implementation
    follows this notation when testing your function. ***

    :param p: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Wasserstein distance
    :return: (FloatTensor) - shape: (1,) - Estimate of the Wasserstein distance
    """
    return torch.mean(critic(x)) - torch.mean(critic(y))


def vf_squared_hellinger(x, y, critic):
    """
    Complete me. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Nowazin et al: https://arxiv.org/pdf/1606.00709.pdf
    In other word, x are samples from the distribution P and y are samples from the distribution Q. Please note that the Critic is unbounded. ***

    :param p: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Squared Hellinger.
    :return: (FloatTensor) - shape: (1,) - Estimate of the Squared Hellinger
    """
    critic_x = critic(x)
    critic_y = critic(y)
    one = torch.ones_like(critic_x)
    t = one - torch.exp(-critic_y)
    return torch.mean(one - torch.exp(-critic_x)) + torch.mean(- t / (one - t))


if __name__ == '__main__':
    # Example of usage of the code provided for answering Q2.5 as well as recommended hyper parameters.
    lambda_reg_lp = 50  # Recommended hyper parameters for the lipschitz regularizer.

    sh = []
    w = []
    wlp = []

    thetas = [theta * 0.1 for theta in range(0, 21)]

    for theta1 in thetas:
        modelsh = q2_model.Critic(2)
        #modelw = q2_model.Critic(2)
        modelwlp = q2_model.Critic(2)

        optimsh = torch.optim.SGD(modelsh.parameters(), lr=1e-3)
        #optimw = torch.optim.SGD(modelw.parameters(), lr=1e-3)
        optimwlp = torch.optim.SGD(modelwlp.parameters(), lr=1e-3)

        sampler1 = iter(q2_sampler.distribution1(0, 512))
        sampler2 = iter(q2_sampler.distribution1(theta1, 512))

        for i in range(500):
            x = torch.Tensor(next(sampler1))
            y = torch.Tensor(next(sampler2))

            modelsh.zero_grad()
            #modelw.zero_grad()
            modelwlp.zero_grad()

            sq_he_loss = - vf_squared_hellinger(x, y, modelsh)
            # w_loss = - vf_wasserstein_distance(x, y, modelw)
            wass = vf_wasserstein_distance(x, y, modelwlp)
            wlp_loss = - wass + lambda_reg_lp * lp_reg(x, y, modelwlp)

            sq_he_loss.backward()
            #w_loss.backward()
            wlp_loss.backward()

            optimsh.step()
            #optimw.step()
            optimwlp.step()

        sh += [-sq_he_loss.item()]
        print("sh ", sh)
        #w += [-w_loss.item()]
        #print("w ", w)
        wlp += [wass.item()]
        print("wlp ", wlp)

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    plt.figure(1)
    plt.xlabel(r'$\theta$')
    plt.ylabel('estimated Squared Hellinger distance')
    plt.title('Estimated Squared Hellinger distance for different ' + r'$\theta$')

    #plt.figure(2)
    #plt.xlabel(r'$\theta$')
    #plt.ylabel('estimated Earth-Mover distance')
    #plt.title('Estimated Earth-Mover distance for different ' + r'$\theta$')

    plt.figure(2)
    plt.xlabel(r'$\theta$')
    plt.ylabel('estimated Earth-Mover distance')
    plt.title('Estimated Earth-Mover distance for different ' + r'$\theta$')


    plt.figure(1)
    plt.plot(thetas, sh)

    #plt.figure(2)
    #plt.plot(thetas, w)

    plt.figure(2)
    plt.plot(thetas, wlp)

    plt.figure(1)
    plt.savefig('sh.jpg')

    #plt.figure(2)
    #plt.savefig('w.jpg')

    plt.figure(2)
    plt.savefig('wlp.jpg')