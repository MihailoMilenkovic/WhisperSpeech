import argparse
import torch

from pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description="Text Generation")
    parser.add_argument(
        "--s2a-model-ckpt-dir",
        type=str,
        default="collabora/whisperspeech:s2a-q4-tiny-en+pl.model",
        help="Path to s2a model checkpoint",
    )
    parser.add_argument(
        "--t2s-model-ckpt-dir",
        type=str,
        default=None,
        help="Path to t2s model checkpoint",
    )
    parser.add_argument("--speaker-embedding-file", type=str, default=None)
    parser.add_argument("--language", type=str, default="en")
    args = parser.parse_args()
    language, s2a_ckpt, t2s_ckpt, speaker_file = (
        args.language,
        args.s2a_model_ckpt_dir,
        args.t2s_model_ckpt_dir,
        args.speaker_embedding_file,
    )
    torch.manual_seed(42)
    tts_pipe = Pipeline(s2a_ref=s2a_ckpt, t2s_ref=t2s_ckpt)

    save_path = "output.wav"
    tts_pipe.generate_to_file(
        save_path,
        "This is a test",
        lang=language,
        # cps=10,
        speaker=speaker_file,
    )


if __name__ == "__main__":
    with torch.no_grad():
        main()
