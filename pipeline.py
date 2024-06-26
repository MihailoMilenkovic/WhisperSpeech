# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/7. Pipeline.ipynb.

# %% auto 0
__all__ = ["Pipeline"]

# %% ../nbs/7. Pipeline.ipynb 1
from os.path import expanduser
import traceback
from pathlib import Path

from t2s_up_wds_mlang_enclm import TSARTransformer
from s2a_delar_mup_wds_mlang import SADelARTransformer
from a2wav import Vocoder
import inference
from default_speaker import default_speaker


# %% ../nbs/7. Pipeline.ipynb 2
class Pipeline:

    def __init__(
        self,
        t2s_location=None,
        s2a_location=None,
        optimize=True,
        torch_compile=False,
        device=None,
    ):
        self.default_speaker = default_speaker
        if device is None:
            device = inference.get_compute_device()
        self.device = device
        args = dict()
        try:
            self.t2s = TSARTransformer.load_model(
                local_filename=t2s_location, device=device
            )  # use obtained compute device
            if optimize:
                self.t2s.optimize(torch_compile=torch_compile)
        except:
            print("Failed to load the T2S model:")
            print(traceback.format_exc())
        try:
            self.s2a = SADelARTransformer.load_model(
                local_filename=s2a_location, device=device
            )  # use obtained compute device
            if optimize:
                self.s2a.optimize(torch_compile=torch_compile)
        except:
            print("Failed to load the S2A model:")
            print(traceback.format_exc())

        self.vocoder = Vocoder(device=device)
        self.encoder = None

    def extract_spk_emb(self, fname):
        # """Extracts a speaker embedding from the first 30 seconds of the give audio file."""
        raise NotImplementedError(
            "Speaker embeddings model not supported for inference examples"
        )
        # import torchaudio

        # if self.encoder is None:
        #     device = self.device
        #     if device == "mps":
        #         device = "cpu"  # operator 'aten::_fft_r2c' is not currently implemented for the MPS device
        #     from speechbrain.pretrained import EncoderClassifier

        #     self.encoder = EncoderClassifier.from_hparams(
        #         "speechbrain/spkrec-ecapa-voxceleb",
        #         savedir=expanduser("~/.cache/speechbrain/"),
        #         run_opts={"device": device},
        #     )
        # audio_info = torchaudio.info(fname)
        # actual_sample_rate = audio_info.sample_rate
        # num_frames = actual_sample_rate * 30  # specify 30 seconds worth of frames
        # samples, sr = torchaudio.load(fname, num_frames=num_frames)
        # samples = samples[:, :num_frames]
        # samples = self.encoder.audio_normalizer(samples[0], sr)
        # spk_emb = self.encoder.encode_batch(samples.unsqueeze(0))

        # return spk_emb[0, 0].to(self.device)

    def generate_atoks(self, text, speaker=None, lang="en", cps=15, step_callback=None):
        if speaker is None:
            speaker = self.default_speaker
        elif isinstance(speaker, (str, Path)):
            speaker = self.extract_spk_emb(speaker)
        text = text.replace("\n", " ")
        stoks = self.t2s.generate(text, cps=cps, lang=lang, step=step_callback)[0]
        # drop all padding tokens (they should only appear at the end)
        stoks = stoks[stoks != 512]
        atoks = self.s2a.generate(stoks, speaker.unsqueeze(0), step=step_callback)
        return atoks

    def generate(self, text, speaker=None, lang="en", cps=15, step_callback=None):
        return self.vocoder.decode(
            self.generate_atoks(
                text, speaker, lang=lang, cps=cps, step_callback=step_callback
            )
        )

    def generate_to_file(
        self, fname, text, speaker=None, lang="en", cps=15, step_callback=None
    ):
        self.vocoder.decode_to_file(
            fname,
            self.generate_atoks(text, speaker, lang=lang, cps=cps, step_callback=None),
        )
