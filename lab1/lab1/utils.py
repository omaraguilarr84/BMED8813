import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.distributions as dist

from tqdm.auto import trange
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import seaborn as sns
import numpy as np

from scipy.integrate import odeint
from scipy.signal import find_peaks
from scipy.sparse import diags

import neo
from elephant.spike_train_generation import inhomogeneous_poisson_process
import quantities as pq

from matplotlib import animation
from IPython.display import HTML
from tempfile import NamedTemporaryFile
import base64
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.patches import Circle

import cvxpy as cvx

def train_model(model, 
                train_dataset, 
                val_dataset,
                objective,
                regularizer=None,
                num_epochs=100, 
                lr=0.1,
                momentum=0.9,
                lr_step_size=25,
                lr_gamma=0.9):
    # progress bars
    pbar = trange(num_epochs)
    pbar.set_description("---")
    inner_pbar = trange(len(train_dataset))
    inner_pbar.set_description("Batch")

    # data loaders for train and validation
    train_dataloader = DataLoader(train_dataset, batch_size=1)
    val_dataloader = DataLoader(val_dataset, batch_size=1)
    dataloaders = dict(train=train_dataloader, val=val_dataloader)

    # use standard SGD with a decaying learning rate
    optimizer = optim.SGD(model.parameters(), 
                          lr=lr, 
                          momentum=momentum)
    scheduler = lr_scheduler.StepLR(optimizer, 
                                    step_size=lr_step_size, 
                                    gamma=lr_gamma)
    
    # Keep track of the best model
    best_model_wts = deepcopy(model.state_dict())
    best_loss = 1e8

    # Track the train and validation loss
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            # set model to train/validation as appropriate
            if phase == 'train':
                model.train()
                inner_pbar.reset()
            else:
                model.eval()
            
            # track the running loss over batches
            running_loss = 0
            running_size = 0
            for datapoint in dataloaders[phase]:
                stim_t = datapoint['stimulus'].squeeze(0)
                spikes_t = datapoint['spikes'].squeeze(0)
                if phase == "train":
                    with torch.set_grad_enabled(True):
                        optimizer.zero_grad()
                        # compute the model output and loss
                        output_t = model(stim_t)
                        loss_t = objective(output_t, spikes_t)
                        # only add the regularizer in the training phase
                        if regularizer is not None:
                            loss_t += regularizer(model)

                        # take the gradient and perform an sgd step
                        loss_t.backward()
                        optimizer.step()
                    inner_pbar.update(1)
                else:
                    # just compute the loss in validation
                    output_t = model(stim_t)
                    loss_t = objective(output_t, spikes_t)

                assert torch.isfinite(loss_t)
                running_loss += loss_t.item()
                running_size += 1
            
            # compute the train/validation loss and update the best
            # model parameters if this is the lowest validation loss yet
            running_loss /= running_size
            if phase == "train":
                train_losses.append(running_loss)
            else:
                val_losses.append(running_loss)
                if running_loss < best_loss:
                    best_loss = running_loss
                    best_model_wts = deepcopy(model.state_dict())

        # Update the learning rate
        scheduler.step()

        # Update the progress bar
        pbar.set_description("Epoch {:03} Train {:.4f} Val {:.4f}"\
                             .format(epoch, train_losses[-1], val_losses[-1]))
        pbar.update(1)

    # load best model weights
    model.load_state_dict(best_model_wts)

    return torch.tensor(train_losses), torch.tensor(val_losses)

#@title
sns.set_context("notebook")

# initialize a color palette for plotting
palette = sns.xkcd_palette(["windows blue",
                            "red",
                            "medium green",
                            "dusty purple",
                            "orange",
                            "amber",
                            "clay",
                            "pink",
                            "greyish"])

def plot_stimulus_weights(glm):
    num_neurons = glm.num_neurons
    max_delay = glm.max_delay

    fig, axs = plt.subplots(num_neurons, 3, figsize=(8, 4 * num_neurons), 
                            gridspec_kw=dict(width_ratios=[1, 1.9, .1]))

    temporal_weights = glm.temporal_conv.weight[:, 0].to("cpu").detach()
    bias = glm.temporal_conv.bias.to("cpu").detach()
    spatial_weights = glm.spatial_conv.weight.to("cpu").detach()
    spatial_weights = spatial_weights.reshape(num_neurons, 50, 50)

    # normalize and flip the spatial weights
    for n in range(num_neurons):
        # Flip if spatial weight peak is negative
        if torch.allclose(spatial_weights[n].min(), 
                       -abs(spatial_weights[n]).max()):
            spatial_weights[n] = -spatial_weights[n]
            temporal_weights[n] = -temporal_weights[n]

        # Normalize
        scale = torch.linalg.norm(spatial_weights[n])
        spatial_weights[n] /= scale
        temporal_weights[n] *= scale

    # Set the same limits for each neuron
    vlim = abs(spatial_weights).max()
    ylim = abs(temporal_weights).max()
    
    for n in range(num_neurons):
        axs[n, 0].plot(torch.arange(-max_delay+1, 1) * 10, temporal_weights[n])
        axs[n, 0].set_ylim(-ylim, ylim)
        axs[n, 0].plot(torch.arange(-max_delay+1, 1) * 10, torch.zeros(max_delay), ':k')
        if n < num_neurons - 1:
            axs[n, 0].set_xticklabels([])
        else:
            axs[n, 0].set_xlabel("$\Delta t$ [ms]")

        im = axs[n, 1].imshow(spatial_weights[n], 
                              vmin=-vlim, vmax=vlim, cmap="RdBu")
        axs[n, 1].set_xticklabels([])
        axs[n, 1].set_yticklabels([])
        axs[n, 1].set_xlabel("neuron {}".format(n + 1))
        plt.colorbar(im, cax=axs[n, 2])

        axs[0, 0].set_title("temporal weights")
        axs[0, 1].set_title("spatial weights")
    

def train_ae(model, train_loader, validation_loader, device, epochs=300):
    # Training Parameters
    training_outputs = []
    training_losses = []
    validation_outputs = []
    validation_losses = []

    # Step 1: using Mean-Squared-Error MSE Loss function
    MSE_function = torch.nn.MSELoss().to(device)

    # Using an Adam Optimizer with lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2, weight_decay = 1e-8)
    # ----------------------------------------------
    t = trange(epochs, desc='epoch desc', leave=True)
    for epoch in t:
        # training on train set
        model.train()

        # Loop through your training data
        training_loss_this_epoch = 0.0
        for batch_idx, batch in enumerate(train_loader):
            # STEP 2: pull out the data from your batch
            batch_data = batch['data']

            # STEP 3: get the reconstructed data from the Autoencoder Output
            reconstructed = model(batch_data)

            # STEP 4: calculate the loss function between the reconstrucion and original data
            loss = MSE_function(reconstructed, batch_data)

            # set gradients to zero
            optimizer.zero_grad()
            # the gradient is computed and stored
            loss.backward()
            # perform the parameter update
            optimizer.step()
            training_loss_this_epoch += float(loss.detach())

        training_loss_this_epoch /= len(train_loader)
        training_losses.append(training_loss_this_epoch)

        t.set_description("Training loss: "+str(training_loss_this_epoch), refresh=True)

        # put model into evaluation mode
        model.eval()
        validation_loss_this_epoch = 0.0
        # loop through your validation/validation data
        for validation_batch_idx, validation_batch in enumerate(validation_loader):
            # STEP_5: pull out the data from your validation batch
            validation_batch_data = validation_batch['data']

            # STEP 6: get the reconstructed data from the Autoencoder Output
            reconstructed_validation = model(validation_batch_data)

            # STEP 7: calculate the loss function between the reconstrucion and original data
            validation_loss = MSE_function(reconstructed_validation, validation_batch_data)

            validation_loss_this_epoch += float(validation_loss.detach())

        # STEP 8: append the averaged validation losses over batches to the validation loss list
        validation_loss_this_epoch /= len(validation_loader)
        validation_losses.append(validation_loss_this_epoch)
    
    return model, training_losses, validation_losses

def integrated_oscillator(dt, num_steps, x0=0, y0=1, angular_frequency=2*np.pi*1e-3):

    assert isinstance(num_steps, int), "num_steps has to be integer"
    t = dt*np.arange(num_steps)
    x = x0*np.cos(angular_frequency*t) + y0*np.sin(angular_frequency*t)
    y = -x0*np.sin(angular_frequency*t) + y0*np.cos(angular_frequency*t)
    return t, np.array((x, y))


def integrated_lorenz(dt, num_steps, x0=0, y0=1, z0=1.05,
                      sigma=10, rho=28, beta=2.667, tau=1e3):

    def _lorenz_ode(point_of_interest, timepoint, sigma, rho, beta, tau):

        x, y, z = point_of_interest

        x_dot = (sigma*(y - x)) / tau
        y_dot = (rho*x - y - x*z) / tau
        z_dot = (x*y - beta*z) / tau
        return x_dot, y_dot, z_dot

    assert isinstance(num_steps, int), "num_steps has to be integer"

    t = dt*np.arange(num_steps)
    poi = (x0, y0, z0)
    return t, odeint(_lorenz_ode, poi, t, args=(sigma, rho, beta, tau)).T


def random_projection(data, embedding_dimension, loc=0, scale=None):

    if scale is None:
        scale = 1 / np.sqrt(data.shape[0])
    projection_matrix = np.random.normal(loc, scale, (embedding_dimension, data.shape[0]))
    return np.dot(projection_matrix, data)


def generate_spiketrains(instantaneous_rates, num_trials, timestep):

    spiketrains = []
    for _ in range(num_trials):
        spiketrains_per_trial = []
        for inst_rate in instantaneous_rates:
            anasig_inst_rate = neo.AnalogSignal(inst_rate, sampling_rate=1/timestep, units=pq.Hz)
            spiketrains_per_trial.append(inhomogeneous_poisson_process(anasig_inst_rate))
        spiketrains.append(spiketrains_per_trial)

    return spiketrains


def generate_syn_data(num_trials, num_neurons):
    # set parameters for the integration of the harmonic oscillator
    timestep = 1 * pq.ms
    trial_duration = 2 * pq.s
    num_steps = int((trial_duration.rescale('ms')/timestep).magnitude)

    # set parameters for spike train generation
    max_rate = 70 * pq.Hz

    # generate a low-dimensional latent variables
    times_oscillator, oscillator_latent_2dim = integrated_oscillator(
        timestep.magnitude, num_steps=num_steps)
    times_oscillator = (times_oscillator*timestep.units).rescale('s')

    # random projection to high-dimensional space
    oscillator_latent_Ndim = random_projection(
        oscillator_latent_2dim, embedding_dimension=num_neurons)

    # convert to instantaneous rate for Poisson process
    normed_latent = oscillator_latent_Ndim / oscillator_latent_Ndim.max()
    instantaneous_rates_oscillator = np.power(max_rate.magnitude, normed_latent)

    # generate spike trains
    spiketrains_oscillator = generate_spiketrains(
        instantaneous_rates_oscillator, num_trials, timestep)
    
    return times_oscillator, oscillator_latent_2dim, spiketrains_oscillator

def generate_templates(num_channels, len_waveform, num_neurons):
    # Make (semi) random templates
    templates = []
    for k in range(num_neurons):
        center = dist.Uniform(0.0, num_channels).sample()
        width = dist.Uniform(1.0, 1.0 + num_channels / 10.0).sample()
        spatial_factor = torch.exp(-0.5 * (torch.arange(num_channels) - center)**2 / width**2)
        
        dt = torch.arange(len_waveform)
        period = len_waveform / (dist.Uniform(1.0, 2.0).sample())
        z = (dt - 0.75 * period) / (.25 * period)
        warp = lambda x: -torch.exp(-x) + 1
        window = torch.exp(-0.5 * z**2)
        shape = torch.sin(2 * torch.pi * dt / period)
        temporal_factor = warp(window * shape)

        template = torch.outer(spatial_factor, temporal_factor)
        template /= torch.linalg.norm(template)
        templates.append(template)
    
    return torch.stack(templates)
    

def generate_syn_sorting(num_timesteps, 
             num_channels, 
             len_waveform, 
             num_neurons, 
             mean_amplitude=15,
             shape_amplitude=3.0,
             noise_std=1, 
             sample_freq=1000):
    """Create a random set of model parameters and sample data.

    Parameters:
    num_timesteps: integer number of time samples in the data
    num_channels: integer number of channels
    len_waveform: integer duration (number of samples) of each template
    num_neurons: integer number of neurons
    """    
    # Make semi-random templates
    templates = generate_templates(num_channels, len_waveform, num_neurons)

    # Make random amplitudes
    amplitudes = torch.zeros((num_neurons, num_timesteps))
    for k in range(num_neurons):
        num_spikes = dist.Poisson(num_timesteps / sample_freq * 10.0).sample()
        sample_shape = (1 + int(num_spikes),)
        times = dist.Categorical(torch.ones(num_timesteps) / num_timesteps).sample(sample_shape)
        amps = dist.Gamma(shape_amplitude, shape_amplitude / mean_amplitude).sample(sample_shape)
        amplitudes[k, times] = amps

        # Only keep spikes separated by at least D
        times, props = find_peaks(amplitudes[k], distance=len_waveform, height=1e-3)
        amplitudes[k] = 0
        amplitudes[k, times] = torch.tensor(props['peak_heights'], dtype=torch.float32)

    # Convolve the signal with each row of the multi-channel template
    data = F.conv1d(amplitudes.unsqueeze(0),
                    templates.permute(1, 0, 2).flip(dims=(2,)),
                    padding=len_waveform-1)[0, :, :-(len_waveform-1)]
    
    data += dist.Normal(0.0, noise_std).sample(data.shape)

    return templates, amplitudes, data

def plot_model(templates, amplitude, data, scores=None, lw=2, figsize=(12, 6)):
    """Plot the raw data as well as the underlying signal amplitudes and templates.
    
    amplitude: (K,T) array of underlying signal amplitude
    template: (K,N,D) array of template that is convolved with signal
    data: (N, T) array (channels x time)
    scores: optional (K,T) array of correlations between data and template
    """    
    # prepend dimension if data and template are 1d
    data = torch.atleast_2d(data)
    N, T = data.shape
    amplitude = torch.atleast_2d(amplitude)
    K, _ = amplitude.shape
    templates = templates.reshape(K, N, -1)
    D = templates.shape[-1]
    dt = torch.arange(D)
    if scores is not None:
        scores = torch.atleast_2d(scores)

    # Set up figure with 2x2 grid of panels
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, K + 1, height_ratios=[1, 2], width_ratios=[1] * K + [2 * K])

    # plot the templates
    t_spc = 1.05 * abs(templates).max()
    for n in range(K):
        ax = fig.add_subplot(gs[1, n])
        ax.plot(dt.numpy(), (templates[n].T - t_spc * torch.arange(N)).numpy(), 
                '-', color=palette[n % len(palette)], lw=lw)
        ax.set_xlabel("delay $d$")
        ax.set_xlim([0, D])
        ax.set_yticks(-t_spc * torch.arange(N))
        ax.set_yticklabels([])
        ax.set_ylim(-N * t_spc, t_spc)
        if n == 0:
            ax.set_ylabel("channels $n$")
        ax.set_title("$W_{{ {} }}$".format(n+1))

    # plot the amplitudes for each neuron
    ax = fig.add_subplot(gs[0, -1])
    a_spc = 1.05 * abs(amplitude).max()
    if scores is not None:
        a_spc = max(a_spc, 1.05 * abs(scores).max())

    for n in range(K):
        ax.plot(amplitude[n] - a_spc * n, '-', color=palette[n % len(palette)], lw=lw)
        
        if scores is not None:
            ax.plot(scores[n] - a_spc * n, ':', color=palette[n % len(palette)], lw=lw,
                label="$X \star W$")
        
    ax.set_xlim([0, T])
    ax.set_xticklabels([])
    ax.set_yticks(-a_spc * torch.arange(K).numpy())
    ax.set_yticklabels([])
    ax.set_ylabel("neurons $k$")
    ax.set_title("amplitude $a$")
    if scores is not None:
        ax.legend()

    # plot the data
    ax = fig.add_subplot(gs[1, -1])
    d_spc = 1.05 * abs(data).max()
    ax.plot((data.T - d_spc * torch.arange(N)).numpy(), '-', color='gray', lw=lw)
    ax.set_xlabel("time $t$")
    ax.set_xlim([0, T])
    ax.set_yticks(-d_spc * torch.arange(N).numpy())
    ax.set_yticklabels([])
    ax.set_ylim(-N * d_spc, d_spc)
    # ax.set_ylabel("channels $c$")
    ax.set_title("data $\mathbb{E}[X]$")

def plot_templates(templates, 
                   indices,
                   scale=0.1,
                   n_cols=8,
                   panel_height=6,
                   panel_width=1.25,
                   colors=('k',),
                   label="neuron",
                   sample_freq=30000,
                   fig=None,
                   axs=None):
    n_subplots = len(indices)
    n_cols = min(n_cols, n_subplots)
    n_rows = int(torch.ceil(torch.tensor(n_subplots / n_cols)))

    if fig is None and axs is None:
        fig, axs = plt.subplots(n_rows, n_cols, 
                                figsize=(panel_width * n_cols, panel_height * n_rows),
                                sharex=True, sharey=True)
    
    n_units, n_channels, spike_width = templates.shape
    timestamps = torch.arange(-spike_width // 2, spike_width//2) / sample_freq
    for i, ind in enumerate(indices):
        row, col = i // n_cols, i % n_cols
        ax = axs[row, col] if n_rows > 1 else axs[col]
        color = colors[i % len(colors)]
        ax.plot((timestamps * 1000).numpy(), 
                (templates[ind].T - scale * torch.arange(n_channels)).numpy(), 
                '-', color=color, lw=1)
        
        ax.set_title("{} {:d}".format(label, ind + 1))
        ax.set_xlim(timestamps[0] * 1000, timestamps[-1] * 1000)
        ax.set_yticks(-scale * torch.arange(0, n_channels+1, step=4))
        ax.set_yticklabels(torch.arange(0, n_channels+1, step=4).numpy() + 1)
        ax.set_ylim(-scale * n_channels, scale)

        if i // n_cols == n_rows - 1:
            ax.set_xlabel("time [ms]")
        if i % n_cols == 0:
            ax.set_ylabel("channel")

        # plt.tight_layout(pad=0.1)

    # hide the remaining axes
    for i in range(n_subplots, len(axs)):
        row, col = i // n_cols, i % n_cols
        ax = axs[row, col] if n_rows > 1 else axs[col]
        ax.set_visible(False)

    return fig, axs

_VIDEO_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

def _anim_to_html(anim, fps=20):
    # todo: todocument
    fname = "./calcium.mp4"
    if not hasattr(fname, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(fname, fps=fps, codec ="libx264")
            video = open(fname, "rb").read()
        anim._encoded_video = base64.b64encode(video)

    return _VIDEO_TAG.format(anim._encoded_video.decode('ascii'))

def play_calcium(movie, fps=30, speedup=1, fig_height=6):
    # First set up the figure, the axis, and the plot element we want to animate
    Py, Px, T = movie.shape
    fig, ax = plt.subplots(1, 1, figsize=(fig_height * Px/Py, fig_height))
    im = plt.imshow(movie[..., 0], interpolation='None', cmap=plt.cm.gray)
    tx = plt.text(0.75, 0.05, 't={:.3f}s'.format(0), 
                  color='white',
                  fontdict=dict(size=12),
                  horizontalalignment='left',
                  verticalalignment='center', 
                  transform=ax.transAxes)
    plt.axis('off')

    def animate(i):
        im.set_data(movie[..., i * speedup])
        tx.set_text("t={:.3f}s".format(i * speedup / fps))
        return im, 

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, 
                                   frames=T // speedup, 
                                   interval=1, 
                                   blit=True)
    plt.close(anim._fig)

    # return an HTML video snippet
    print("Preparing animation. This may take a minute...")
    return HTML(_anim_to_html(anim, fps=30))

def plot_peaks(local_correlations, filtered_correlations, peaks, height, width, NEURON_WIDTH):
    def _plot_panel(ax, im, title):
        h = ax.imshow(im, cmap="Greys_r")
        ax.set_title(title)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_axis_off()

        # add a colorbar of the same height
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad="2%")
        plt.colorbar(h, cax=cax)

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    _plot_panel(axs[0], local_correlations, "local correlations")
    _plot_panel(axs[1], filtered_correlations, "filtered correlations")
    _plot_panel(axs[2], local_correlations, "candidate neurons")

    # Draw circles around the peaks
    for n, yx in enumerate(peaks):
        y, x = yx
        axs[2].add_patch(Circle((x, y), 
                                radius=NEURON_WIDTH/2, 
                                facecolor='none', 
                                edgecolor='red', 
                                linewidth=1))
        
        axs[2].text(x, y, "{}".format(n),
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontdict=dict(size=10, weight="bold"),
                    color='r')

def plot_deconvolved_traces(traces, denoised_traces, amplitudes):
    num_neurons, num_frames = traces.shape

    # Plot the traces and our denoised estimates
    scale = torch.quantile(traces, .995, dim=1, keepdims=True)
    offset = -torch.arange(num_neurons)

    # Plot points at the time frames where the (normalized) amplitudes are > 0.05
    sparse_amplitudes = amplitudes / scale
    # sparse_amplitudes = torch.isclose(sparse_amplitudes, 0, atol=0.05)
    sparse_amplitudes[sparse_amplitudes < 0.05] = torch.nan
    sparse_amplitudes[sparse_amplitudes > 0.05] = 0.0

    plt.figure(figsize=(12, 8))
    plt.plot(((traces / scale).T + offset ).numpy(), color=palette[0], lw=1, alpha=0.5)
    plt.plot(((denoised_traces / scale).T + offset).numpy(), color=palette[0], lw=2)
    plt.plot(((sparse_amplitudes).T + offset).numpy(), color=palette[1], marker='o', markersize=2)
    plt.xlabel("time (frames)")
    plt.xlim(0, num_frames)
    plt.ylabel("neuron")
    plt.yticks(-torch.arange(0, num_neurons, step=5), 
               labels=torch.arange(0, num_neurons, step=5).numpy())
    plt.ylim(-num_neurons, 2)
    plt.title("raw and denoised fluorescence traces")

def plot_footprints_traces(flat_data, params, hypers, height, width, FPS, plot_bkgd=True, indices=None):
    U = params["footprints"].reshape(-1, height, width)
    u0 = params["bkgd_footprint"].reshape(height, width)
    C = params["traces"]
    c0 = params["bkgd_trace"]
    N, T = C.shape

    if indices is None: 
        indices = torch.arange(N)

    def _plot_factor(footprint, trace, title):
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        vlim = abs(footprint).max()
        h = ax1.imshow(footprint.numpy(), vmin=-vlim, vmax=vlim, cmap="RdBu")
        ax1.set_title(title)
        ax1.set_axis_off()

        # add a colorbar of the same height
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad="2%")
        plt.colorbar(h, cax=cax)

        ax2 = divider.append_axes("right", size="150%", pad="75%")
        ts = torch.arange(T) / FPS
        ax2.plot(ts.numpy(), trace.numpy(), color=palette[0], lw=2)
        ax2.set_xlabel("time (sec)")
        ax2.set_ylabel("fluorescence trace")
        
    if plot_bkgd:
        _plot_factor(u0, c0, "background")

    for k in indices:            
        _plot_factor(U[k], C[k], "neuron {}".format(k))

def deconvolve(trace, 
               noise_std=1.0, 
               epsilon=1.0,
               tau=0.300 * 30,
               full_output=False,
               verbose=False):
    """Deconvolve a noisy calcium trace (aka "target") by solving a 
    the convex optimization problem described above.

    Parameters
    ----------
    trace: a shape (T,) tensor containing the noisy trace.
    noise_std: scalar noise standard deviation $\sigma$
    epsilon: extra slack for the norm constraint. 
        (Typically > 0 and certainly > -1)
    tau: the time constant of the calcium indicator decay.
    full_output: if True, return a dictionary with the deconvolved 
        trace and a bunch of extra info, otherwise just return the trace.
    verbose: flag to pass to the CVX solver to print more info.
    """
    assert trace.ndim == 1
    T = len(trace)

    ###
    # YOUR CODE BELOW

    # Initialize the variable we're optimizing over
    c = cvx.Variable(T)
    b = cvx.Variable(1)

    # Create the sparse matrix G with 1 on the diagonal and 
    # -e^{-1/\tau} on the first lower diagonal
    G = diags([[1.0]*T,[-np.exp(-1.0/tau)]], [0, -1])

    # set the threshold to (1+\epsilon) \sigma \sqrt{T}
    theta = (1 + epsilon) * noise_std * np.sqrt(T)

    # Define the objective function
    objective = cvx.Minimize(cvx.norm1(G @ c))
    
    # Set the constraints. 
    # PUT THE NORM CONSTRAINT FIRST, THEN THE NON-NEGATIVITY CONSTRAINT!
    constraints = [cvx.sum_squares(trace - c - b) <= theta*theta, G @ c >= 0]

    # Construct the problem
    prob = cvx.Problem(objective, constraints)
    ###

    # Solve the optimization problem. 
    try:
        # First try the default solver then revert to SCS if it fails.
        result = prob.solve(verbose=verbose)
    except Exception as e:
        print("Default solver failed with exception:")
        print(e)
        print("Trying 'solver=SCS' instead.")
        # if this still fails we give up!
        result = prob.solve(verbose=verbose, solver="SCS")

    # Make sure the result is finite (i.e. it found a feasible solution)
    if torch.isinf(torch.tensor(result)): 
        raise Exception("solver failed to find a feasible solution!")

    all_results = dict(
        trace=c.value,
        baseline=b.value,
        result=result,
        amplitudes=G @ c.value,
        lagrange_multiplier=constraints[0].dual_value[0]
    )
    assert torch.numel(torch.tensor(constraints[0].dual_value)) == 1, \
        "Make sure your first constraint is on the norm of the residual."

    return all_results if full_output else c.value

def plot_reduction(y1, y2, y3, vector):
    y1_female = []
    y2_female = []
    y3_female = []

    y1_male = []
    y2_male = []
    y3_male = []

    for idx, gender_label in enumerate(vector):
        if gender_label == 1:
            y1_female.append(y1[idx])
            y2_female.append(y2[idx])
            y3_female.append(y3[idx])

        elif gender_label == 0:
            y1_male.append(y1[idx])
            y2_male.append(y2[idx])
            y3_male.append(y3[idx])

    print(len(y1_female), len(y2_female), len(y3_female))
    print(len(y1_male), len(y2_male), len(y3_male))

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection='3d', computed_zorder=False)

    ax.scatter3D(y1_female, y2_female, y3_female, color='purple', alpha=0.25)

    ax.scatter3D(y1_male, y2_male, y3_male, color='goldenrod', alpha=0.25)

    ax.set_xlabel('d1')
    ax.set_ylabel('d2')
    ax.set_zlabel('d3')

    ax.legend(['female', 'male'])
    