"""Generates VSP training batches using bounded GPU workers."""
import os

import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info
from vspsampling.download import DDPM as VSP_DDPM, REPO as VSP_ROOT
from vspsampling.models.conditional_ddpm import ConditionalDDPM


class VSPDDPMSampler:
    """Uses pretrained VSP DDPM. observations must already be normalized."""
    def __init__(self, device, n_steps=50):
        self.device = torch.device(device)
        self.n_steps = int(n_steps)

        ckpt = torch.load(os.path.join(VSP_ROOT, VSP_DDPM),
                          map_location="cpu", weights_only=False)
        cfg = ckpt["args"]
        self.model = ConditionalDDPM(
            d_x=41, d_s=20, width=cfg.width, depth=cfg.depth, T=cfg.T,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.float().eval()
        self.normalizer = ckpt["y_normalizer"]

    def normalize(self, y):
        return self.normalizer.normalize(y.to(device="cpu",
                                              dtype=torch.float32))

    @torch.no_grad()
    def draw(self, y_normalized, n, generator):
        """``(n_obs, n, d)`` float64 samples, n per observation row."""
        y = y_normalized.to(self.device, dtype=torch.float32)
        n = int(n)
        seed = int(torch.randint(0, 2**31 - 1, (1,),
                                 generator=generator).item())
        g = torch.Generator(device=self.device).manual_seed(seed)

        samples = []
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            for obs in y:
                cond = obs.unsqueeze(0).expand(n, -1)
                samples.append(self.model.sample(
                    cond, n_steps=self.n_steps, generator=g))
        finally:
            torch.set_default_dtype(previous_dtype)
        return torch.stack(samples).to(torch.float64)


class VSPTrainingBatches(Dataset):
    def __init__(self, y, args):
        self.y = y.float().cpu()
        self.args = args
        self.sampler = None

    def __len__(self):
        return int(self.args.n_steps)

    def __getitem__(self, step):
        a = self.args
        if self.sampler is None:
            worker = get_worker_info()
            device = "cpu" if worker is None else f"cuda:{worker.id}"
            if worker is not None:
                torch.cuda.set_device(worker.id)
            self.sampler = VSPDDPMSampler(device, int(a.ddim_steps))

        g = torch.Generator().manual_seed(int(a.base_seed) + 7 + step)
        m_idx = torch.randint(len(a.m_list), (1,), generator=g).item()
        m = int(a.m_list[m_idx])
        idx = torch.randperm(len(self.y), generator=g)[:int(a.batch_obs)]
        y = self.y[idx]
        samples = self.sampler.draw(y, m + int(a.n_mu_train), g).cpu()
        return m, y, samples


def training_batches(y, args):
    workers = int(args.sampler_workers)
    if workers < 0 or workers > torch.cuda.device_count():
        raise ValueError("sampler_workers must be between 0 and the visible GPU count")
    if len(y) == 0 or int(args.n_steps) < 1:
        raise ValueError("Training needs observations and at least one step")

    kwargs = {}
    if workers:
        kwargs = dict(multiprocessing_context="spawn", prefetch_factor=2, timeout=300)
    loader = DataLoader(
        VSPTrainingBatches(y, args), batch_size=None, num_workers=workers,
        pin_memory=False,
        generator=torch.Generator().manual_seed(int(args.base_seed)),
        **kwargs,
    )
    print(f"[samples] workers={workers}; "
          f"prefetch={2 * workers if workers else 0} batches", flush=True)
    yield from loader #generator