import torch
import pytorch_lightning as pl

class LitDiffusionModel(pl.LightningModule):
    def __init__(self, n_dim=3, n_steps=200, lbeta=1e-5, ubeta=1e-2, scheduler_type='linear'):
        super().__init__()
        """
        If you include more hyperparams (e.g. `n_layers`), be sure to add that to `argparse` from `train.py`.
        Also, manually make sure that this new hyperparameter is being saved in `hparams.yaml`.
        """
        self.lr = 0.001
        self.save_hyperparameters()

        """
        Your model implementation starts here. We have separate learnable modules for `time_embed` and `model`.
        You may choose a different architecture altogether. Feel free to explore what works best for you.
        If your architecture is just a sequence of `torch.nn.XXX` layers, using `torch.nn.Sequential` will be easier.
        
        `time_embed` can be learned or a fixed function based on the insights you get from visualizing the data.
        If your `model` is different for different datasets, you can use a hyperparameter to switch between them.
        Make sure that your hyperparameter behaves as expecte and is being saved correctly in `hparams.yaml`.
        """
        # self.time_embed = None
        w = 256
        t_w = 16
        self.time_embed = torch.nn.Sequential(torch.nn.Linear(1, 16),
                                  torch.nn.Tanh(),
                                  torch.nn.Linear(16, t_w),
                                  torch.nn.Tanh())
        self.model = torch.nn.Sequential(
          torch.nn.Linear(t_w+3, w),
          torch.nn.Tanh(),
          torch.nn.Linear(w, 2*w),
          torch.nn.Tanh(),
          torch.nn.Linear(2*w, w),
          torch.nn.Tanh(),
          torch.nn.Linear(w, 3)
        )


        """
        Be sure to save at least these 2 parameters in the model instance.
        """
        self.n_steps = n_steps
        self.n_dim = n_dim

        """
        Sets up variables for noise schedule
        """
        self.betas = self.init_alpha_beta_schedule(lbeta, ubeta, scheduler_type)
        self.alphas = 1. - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hats)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hats)

        
        self.inv_sqrt_alphas = torch.pow(self.alphas, -0.5)
        self.pre_noise_terms = self.betas / self.sqrt_one_minus_alpha_hat
        self.sigmas = torch.pow(self.betas, 0.5)
        

    def forward(self, x, t):
        """
        Similar to `forward` function in `nn.Module`. 
        Notice here that `x` and `t` are passed separately. If you are using an architecture that combines
        `x` and `t` in a different way, modify this function appropriately.
        """
        if not isinstance(t, torch.Tensor):
            t = torch.LongTensor([t]).expand(x.size(0))
        # t_embed = self.time_embed(t)
        # t = torch.LongTensor([t]).expand(x.size(0))
        t = (t.float() / self.n_steps) - 0.5
        # t_embed = torch.sin(t).unsqueeze(1)
        t_embed = self.time_embed(t)
        return self.model(torch.cat((x, t_embed), dim=1).float())

    def init_alpha_beta_schedule(self, lbeta, ubeta, scheduler_type):
        """
        Set up your noise schedule. You can perhaps have an additional hyperparameter that allows you to
        switch between various schedules for answering q4 in depth. Make sure that this hyperparameter 
        is included correctly while saving and loading your checkpoints.
        """
        betas = None
        def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
            betas = beta_end * torch.ones(num_diffusion_timesteps, dtype=torch.float64)
            warmup_time = int(num_diffusion_timesteps * warmup_frac)
            betas[:warmup_time] = torch.linspace(beta_start, beta_end, warmup_time, dtype=torch.float64)
            return betas
        if scheduler_type == 'linear':
            betas = torch.linspace(lbeta, ubeta, self.n_steps, dtype=torch.float64)
        elif scheduler_type == 'const':
            betas = ubeta * torch.ones(self.n_steps, dtype=torch.float64)
        elif scheduler_type == 'quad':
            betas = torch.linspace(lbeta ** 0.5, ubeta ** 0.5, self.n_steps, dtype=torch.float64) ** 2
        elif scheduler_type == 'warmup10':
            betas = _warmup_beta(lbeta, ubeta, self.n_steps, 0.1)
        elif scheduler_type == 'warmup50':
            betas = _warmup_beta(lbeta, ubeta, self.n_steps, 0.5)
        elif scheduler_type == 'cosine':  # https://openreview.net/forum?id=-NEXDKk8gZ
            steps = self.n_steps+1
            s = 0.008
            x = torch.linspace(0, self.n_steps, steps, dtype=torch.float64)
            alphas_cumprod = torch.cos(((x / self.n_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            #print(betas)
        else:
            print("No such scheduler exist!!")

        assert betas.shape == (self.n_steps,)

        return betas


    def q_sample(self, x, t):
        """
        Sample from q given x_t.
        """
        Ɛ = torch.randn_like(x)
        return self.sqrt_alpha_hat[t] * x + self.sqrt_one_minus_alpha_hat[t] * Ɛ, Ɛ

    def p_sample(self, x, t):
        """
        Sample from p given x_t.
        """
        z = torch.randn(x.shape).to(self.device)
        if t == 1:
            z = z * 0
        steps = torch.full((x.shape,), t, dtype=torch.int, device=self.device).unsqueeze(1)
        model_output = self.forward(x, steps)
        x = self.inv_sqrt_alphas[t - 1] * (x - self.pre_noise_terms[t - 1] * model_output) + self.sigmas[t - 1] * z
        return x

    def training_step(self, batch, batch_idx):
        """
        Implements one training step.
        Given a batch of samples (n_samples, n_dim) from the distribution you must calculate the loss
        for this batch. Simply return this loss from this function so that PyTorch Lightning will 
        automatically do the backprop for you. 
        Refer to the DDPM paper [1] for more details about equations that you need to implement for
        calculating loss. Make sure that all the operations preserve gradients for proper backprop.
        Refer to PyTorch Lightning documentation [2,3] for more details about how the automatic backprop 
        will update the parameters based on the loss you return from this function.

        References:
        [1]: https://arxiv.org/abs/2006.11239
        [2]: https://pytorch-lightning.readthedocs.io/en/stable/
        [3]: https://www.pytorchlightning.ai/tutorials
        """
        # print("forward-->>")
        t = torch.randint(low=1, high=self.n_steps, size=(batch.shape[0],)).unsqueeze(1)
        # print("-"*10, f"t shape - {t.shape}")
        x_t, noise = self.q_sample(batch, t)
        # print("-"*10, f"x_t shape - {x_t.shape}")
        predicted_noise = self.forward(x_t, t)
        # print("-"*10, f"predicted_noise shape - {predicted_noise.shape}")
        
        loss = (noise - predicted_noise) ** 2
        loss = loss.mean()
        return loss

    def sample(self, n_samples, progress=False, return_intermediate=False):
        """
        Implements inference step for the DDPM.
        `progress` is an optional flag to implement -- it should just show the current step in diffusion
        reverse process.
        If `return_intermediate` is `False`,
            the function returns a `n_samples` sampled from the learned DDPM
            i.e. a Tensor of size (n_samples, n_dim).
            Return: (n_samples, n_dim)(final result from diffusion)
        Else
            the function returns all the intermediate steps in the diffusion process as well 
            i.e. a Tensor of size (n_samples, n_dim) and a list of `self.n_steps` Tensors of size (n_samples, n_dim) each.
            Return: (n_samples, n_dim)(final result), [(n_samples, n_dim)(intermediate) x n_steps]
        """
        x = torch.randn(n_samples, 3)
        all_x = [x]
        for i in range(self.n_steps, 0, -1):
            z = torch.randn(x.shape).to(self.device)
            if i == 1:
                z = z * 0
            steps = torch.full((n_samples,), i, dtype=torch.int, device=self.device).unsqueeze(1)
            model_output = self.forward(x, steps)
            x = self.inv_sqrt_alphas[i - 1] * (x - self.pre_noise_terms[i - 1] * model_output) + self.sigmas[i - 1] * z
            if return_intermediate:
                all_x.append(x)
        if return_intermediate:
            return x, all_x
        return x
    
    def configure_optimizers(self):
        """
        Sets up the optimizer to be used for backprop.
        Must return a `torch.optim.XXX` instance.
        You may choose to add certain hyperparameters of the optimizers to the `train.py` as well.
        In our experiments, we chose one good value of optimizer hyperparameters for all experiments.
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.001)
