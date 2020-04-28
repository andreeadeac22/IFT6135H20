"""
Template for Question 3.
@author: Samuel Lavoie
"""
import torch
from q3_sampler import svhn_sampler
from q3_model import Critic, Generator
from torch import optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.animation as animation
from IPython.display import HTML


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
    t = unif.rsample(x.shape)
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


if __name__ == '__main__':
    # Example of usage of the code provided and recommended hyper parameters for training GANs.
    data_root = './'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_iter = 50000  # N training iterations
    n_critic_updates = 5  # N critic updates per generator update
    lp_coeff = 10  # Lipschitz penalty coefficient
    train_batch_size = 64
    test_batch_size = 64
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.9
    z_dim = 100
    n_epochs = 200

    train_loader, valid_loader, test_loader = svhn_sampler(data_root, train_batch_size, test_batch_size)

    generator = Generator(z_dim=z_dim).to(device)
    critic = Critic().to(device)

    optim_critic = optim.Adam(critic.parameters(), lr=lr, betas=(beta1, beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))

    # WGAN
    noise = torch.FloatTensor(train_batch_size, z_dim)
    fixed_noise = torch.FloatTensor(9, z_dim).normal_(0, 1)

    G_losses = []
    D_losses = []
    img_list = []

    # COMPLETE TRAINING PROCEDURE
    iters = 0
    batch_num = 0

    for epoch in range(n_epochs):
        for data in train_loader:
            batch_num += 1
            real_img, _ = data

            ############################
            # (1) Update D network
            ###########################
            for p in critic.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            critic.zero_grad()

            # Generate batch of latent vectors
            noise = torch.randn(real_img.shape[0], z_dim, device=device)
            # Generate fake image batch with G
            fake = generator(noise)

            wgan_loss = vf_wasserstein_distance(real_img, fake.detach(), critic)
            wlp_loss = - wgan_loss + lp_coeff * lp_reg(real_img, fake.detach(), critic)
            wlp_loss.backward()
            # Update D
            optim_critic.step()

            if batch_num % 5 == 0:
                iters +=1
                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                for p in critic.parameters():
                    p.requires_grad = False  # to avoid computation
                generator.zero_grad()
                fake = generator(noise)
                errG = torch.mean(- critic(fake)).view(1)
                errG.backward()
                optim_generator.step()

                # Output training stats
                if iters % 50 == 0:
                    print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f' % (iters, n_iter, wlp_loss.item(), errG.item()))

                # Save Losses for plotting later
                G_losses.append(errG.detach().item())
                D_losses.append(wlp_loss.detach().item())

                if iters % 200 == 0:
                    # do checkpointing
                    torch.save(critic.state_dict(), 'critic.pt')
                    torch.save(generator.state_dict(), 'generator.pt')


                    with torch.no_grad():
                        fake = generator(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                    vutils.save_image(fake, "temp" + iters + ".png", normalize=True)  # normalize=True!!!!





    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # %%capture
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(train_loader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()

    # COMPLETE QUALITATIVE EVALUATION
