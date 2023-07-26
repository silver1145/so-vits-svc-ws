from __future__ import annotations

from typing import Any, Literal
from cm_time import timer

import torch
import numpy as np
from torch import nn
from numpy import dtype, float32, ndarray
from so_vits_svc_fork.f0 import f0_to_coarse, normalize_f0
from so_vits_svc_fork.modules import commons, synthesizers
from so_vits_svc_fork.inference import core

has_patch = False


class Volume_Extractor:
    def __init__(self, hop_size = 512):
        self.hop_size = hop_size
        
    def extract(self, audio): # audio: 2d tensor array
        if not isinstance(audio,torch.Tensor):
           audio = torch.Tensor(audio)
        n_frames = int(audio.size(-1) // self.hop_size)
        audio2 = audio ** 2
        audio2 = torch.nn.functional.pad(audio2, (int(self.hop_size // 2), int((self.hop_size + 1) // 2)), mode = 'reflect')
        volume = torch.nn.functional.unfold(audio2[:,None,None,:],(1,self.hop_size),stride=self.hop_size)[:,:,:n_frames].mean(dim=1)[0]
        volume = torch.sqrt(volume)
        return volume


def patch_synthesizers():
    old_synthesizertrn_init = synthesizers.SynthesizerTrn.__init__
    def new_synthesizertrn_init(self, *k, **kw):
        old_synthesizertrn_init(self, *k, **kw)
        self.vol_embedding = kw.get("vol_embedding") or False
        if self.vol_embedding:
            self.emb_vol = nn.Linear(1, self.hidden_channels)
    synthesizers.SynthesizerTrn.__init__ = new_synthesizertrn_init

    def new_synthesizertrn_infer(
        self, c, f0, uv, g=None, noice_scale=0.35, predict_f0=False, vol=None
    ):
        c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
        g = self.emb_g(g).transpose(1, 2)
        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        # vol
        vol = (
            self.emb_vol(vol[:, :, None]).transpose(1, 2)
            if vol is not None and self.vol_embedding
            else 0
        )

        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2) + vol

        if predict_f0:
            lf0 = 2595.0 * torch.log10(1.0 + f0.unsqueeze(1) / 700.0) / 500
            norm_lf0 = normalize_f0(lf0, x_mask, uv, random_scale=False)
            pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)
            f0 = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)

        z_p, m_p, logs_p, c_mask = self.enc_p(
            x, x_mask, f0=f0_to_coarse(f0), noice_scale=noice_scale
        )
        z = self.flow(z_p, c_mask, g=g, reverse=True)

        # MB-iSTFT-VITS
        if self.mb:
            o, o_mb = self.dec(z * c_mask, g=g)
        else:
            o = self.dec(z * c_mask, g=g, f0=f0)
        return o

    synthesizers.SynthesizerTrn.infer = new_synthesizertrn_infer

def patch_svc():
    old_svc_init = core.Svc.__init__
    def new_svc_init(self, *k, **kw):
        old_svc_init(self, *k, **kw)
        self.vol_embedding = self.hps.model.get('vol_embedding') or False
        if self.vol_embedding:
            self.volume_extractor = Volume_Extractor(self.hop_size)
    core.Svc.__init__ = new_svc_init

    LOG = core.LOG
    def new_svc_infer(
        self,
        speaker: int | str,
        transpose: int,
        audio: ndarray[Any, dtype[float32]],
        cluster_infer_ratio: float = 0,
        auto_predict_f0: bool = False,
        noise_scale: float = 0.4,
        f0_method: Literal["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"] = "dio",
    ) -> tuple[torch.Tensor, int]:
        audio = audio.astype(np.float32)
        # get speaker id
        if isinstance(speaker, int):
            if len(self.spk2id.__dict__) >= speaker:
                speaker_id = speaker
            else:
                raise ValueError(
                    f"Speaker id {speaker} >= number of speakers {len(self.spk2id.__dict__)}"
                )
        else:
            if speaker in self.spk2id.__dict__:
                speaker_id = self.spk2id.__dict__[speaker]
            else:
                LOG.warning(f"Speaker {speaker} is not found. Use speaker 0 instead.")
                speaker_id = 0
        speaker_candidates = list(
            filter(lambda x: x[1] == speaker_id, self.spk2id.__dict__.items())
        )
        if len(speaker_candidates) > 1:
            raise ValueError(
                f"Speaker_id {speaker_id} is not unique. Candidates: {speaker_candidates}"
            )
        elif len(speaker_candidates) == 0:
            raise ValueError(f"Speaker_id {speaker_id} is not found.")
        speaker = speaker_candidates[0][0]
        sid = torch.LongTensor([int(speaker_id)]).to(self.device).unsqueeze(0)

        # get unit f0
        c, f0, uv = self.get_unit_f0(audio, transpose, cluster_infer_ratio, speaker, f0_method)

        # inference
        with torch.no_grad():
            with timer() as t:
                vol = self.volume_extractor.extract(torch.FloatTensor(audio).to(self.device)[None,:])[None,:].to(self.device) if self.vol_embedding else None
                audio = self.net_g.infer(
                    c,
                    f0=f0,
                    g=sid,
                    uv=uv,
                    predict_f0=auto_predict_f0,
                    noice_scale=noise_scale,
                    vol=vol,
                )[0, 0].data.float()
            audio_duration = audio.shape[-1] / self.target_sample
            LOG.info(f"Inference time: {t.elapsed:.2f}s, RTF: {t.elapsed / audio_duration:.2f}")
        torch.cuda.empty_cache()
        return audio, audio.shape[-1]
    core.Svc.infer = new_svc_infer

def patch():
    global has_patch
    if has_patch:
        return

    patch_synthesizers()
    patch_svc()

    has_patch = True
