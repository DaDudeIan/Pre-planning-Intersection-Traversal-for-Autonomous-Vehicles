import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import gudhi              # installed transitively by `pip install topologyx`
import numpy as np


class TopologyLoss(nn.Module):
    def __init__(self, img_size=(400, 400), background_id=0,
                 p=2, w_b0=1.0, w_b1=1.0, min_pers=0.0,
                 smooth_sigma=None):
        super().__init__()
        self.H, self.W = img_size
        self.bg        = background_id
        self.p, self.w0, self.w1 = p, w_b0, w_b1
        self.eps       = min_pers
        self.sigma     = smooth_sigma
        if smooth_sigma:
            self.register_buffer("gk", self._gauss(smooth_sigma))

    # ---------------------------------------------------------------- utils
    @staticmethod
    def _gauss(s, k=None):
        if k is None:
            k = 2*math.ceil(3*s)+1
        c = torch.arange(k)-k//2
        w = torch.exp(-(c**2)/(2*s**2)); w /= w.sum()
        return torch.outer(w,w)[None,None]

    @staticmethod
    def _F2C(idxF, H, W):
        # Gudhi uses Fortran order
        r, c = np.divmod(idxF, H)
        return torch.as_tensor(r + c*H, dtype=torch.long)

    # -------------------------------------------------------------- Gudhi call
    @torch.no_grad()
    def _bars(self, img):
        cc = gudhi.CubicalComplex(top_dimensional_cells=img.cpu().numpy())
        cc.compute_persistence()
        pairs, _ = cc.cofaces_of_persistence_pairs()
        h0 = pairs[0] if len(pairs) else np.empty((0,2), int)
        h1 = pairs[1] if len(pairs)>1 else np.empty((0,2), int)
        return h0, h1           # births/deaths as flat indices (F-order)

    # ---------------------------------------------------------------- forward
    def forward(self, logits):
        assert logits.shape[-2:] == (self.H, self.W), "wrong spatial size"
        probs  = F.softmax(logits, dim=1)
        fg     = 1.0 - probs[:, self.bg]          # foreground prob  ∈[0,1]

        # optional blur
        if self.sigma:
            fg = F.conv2d(fg.unsqueeze(1), self.gk, padding="same").squeeze(1)

        # ─── critical sign inversion so Gudhi gets a sub-level function ───
        f = 1.0 - fg                             # low value = sure foreground
        # -------------------------------------------------------------------

        flat = f.view(f.size(0), -1)
        loss = logits.sum()*0                    # keeps autograd graph alive

        for b in range(f.size(0)):               # loop batch
            h0, h1   = self._bars(f[b])
            # drop infinite bar(s) in H0:
            h0       = h0[~np.isinf(h0[:,1])]

            for P, w in ((h0,self.w0),(h1,self.w1)):
                if P.size==0: continue
                ib = self._F2C(P[:,0], self.H, self.W).to(logits.device)
                id = self._F2C(P[:,1], self.H, self.W).to(logits.device)
                birth, death = flat[b][ib], flat[b][id]
                pers = (death - birth)           # >0 in sub-level filtration
                mask = pers > self.eps
                if mask.any():
                    loss = loss + w*(pers[mask]**self.p).sum()

        return loss / logits.size(0)
    
def debug_topology(loss_layer, logits, max_print=10):
    """
    Print the persistence pairs Gudhi returns so you can see what's going on.
    """
    with torch.no_grad():
        probs  = F.softmax(logits, dim=1)
        fg_map = 1.0 - probs[:, loss_layer.bg]     # same convention as loss

        if loss_layer.sigma is not None:
            fg_map = F.conv2d(fg_map.unsqueeze(1),
                              loss_layer.gauss_k, padding="same"
                             ).squeeze(1)

    for b in range(fg_map.size(0)):
        h0, h1 = loss_layer._ph_pairs(fg_map[b])
        print(f"\nImage {b}:  H0 bars = {len(h0)},  H1 bars = {len(h1)}")
        print("  H0 (birth, death) first few:", h0[:max_print])
        print("  H1 (birth, death) first few:", h1[:max_print])
        
def alpha(epoch, alpha_hi = 0.95, alpha_lo = 0.55, T_warm = 50, N_epochs = 500):
    if epoch < T_warm:
        return alpha_hi
    r = (epoch - T_warm) / max(1, N_epochs - T_warm)
    return alpha_hi - (alpha_hi - alpha_lo) * r