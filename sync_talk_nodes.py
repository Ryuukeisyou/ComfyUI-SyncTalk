import subprocess
import os, sys

cur_dir = os.path.dirname(__file__)
sync_talk_dir = os.path.join(cur_dir, "repos", "SyncTalk")
sys.path.append(sync_talk_dir)
import numpy as np
import torch

from .utils import Logger

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
asr_models = ["ave", "hubert", "deepspeech"]

class LoadAve:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {}
        }
    
    RETURN_TYPES = ("SYNC_TALK_AVE",)
    RETURN_NAMES = ("ave",)
    FUNCTION = "run"
    CATEGORY = "SyncTalk/Process"

    def run(self):
        from .repos.SyncTalk.nerf_triplane.network import AudioEncoder
        model = AudioEncoder().to(device_type).eval()
        ckpt = torch.load(os.path.join(os.path.dirname(__file__), "repos","SyncTalk", "nerf_triplane", "checkpoints", "audio_visual_encoder.pth"))
        model.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()})
        return (model, )

class AveProcess:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ave": ("SYNC_TALK_AVE", {}),
                "wav_file": ("STRING", {
                    "default": "path/to/wav/file"
                }),
                "save_output": ("BOOLEAN", {
                    "default": False
                })
            }
        }

    RETURN_TYPES = ("SYNC_TALK_AUDIO_FEATURES", "STRING",)
    RETURN_NAMES = ("st_auds", "ave_npy",)

    FUNCTION = "run"

    OUTPUT_NODE = True

    CATEGORY = "SyncTalk/Process"

    def run(self, ave, wav_file:str, save_output:bool):
        from .sync_talk_utils import aud_process_ave
        auds = aud_process_ave(ave, wav_file, device_type=device_type)
        npy_path = None
        if save_output:
            npy_path = wav_file.replace(".wav", "_ave.npy")
            np.save(npy_path, auds.numpy())
        return (auds, npy_path,)

class LoadHubert:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "facebook/hubert-large-ls960-ft",
                })
            }
        }
    
    RETURN_TYPES = ("SYNC_TALK_HUBERT",)
    RETURN_NAMES = ("hubert",)
    FUNCTION = "run"
    CATEGORY = "SyncTalk/Process"

    def run(self, model_path: str):
        from transformers import Wav2Vec2Processor, HubertModel
        print("Loading the Wav2Vec2 Processor...")
        wav2vec2_processor = Wav2Vec2Processor.from_pretrained(model_path)
        print("Loading the HuBERT Model...")
        hubert_model = HubertModel.from_pretrained(model_path).to(device_type)
        return ([wav2vec2_processor, hubert_model],)

class HubertProcess:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hubert": ("SYNC_TALK_HUBERT", {}),
                "wav_file": ("STRING", {}),
                "save_output": ("BOOLEAN", {
                    "default": False
                })
            }
        }
    
    RETURN_TYPES = ("SYNC_TALK_AUDIO_FEATURES", "STRING",)
    RETURN_NAMES = ("st_auds", "hubert_npy",)

    FUNCTION = "run"

    OUTPUT_NODE = True

    CATEGORY = "SyncTalk/Process"

    def run(self, hubert, wav_file: str, save_output: bool):
        from .sync_talk_utils import aud_process_hubert
        wav2vec2_processor, hubert_model = hubert
        hubert_hidden = aud_process_hubert(wav2vec2_processor, hubert_model, wav_file, device_type=device_type)
        npy_path = None
        if save_output:
            npy_path = wav_file.replace(".wav", "_hu.npy")
            np.save(npy_path, hubert_hidden.detach().numpy())
        return (hubert_hidden, npy_path,)

class LoadDeepSpeech:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": None,
                })
            }
        }
    
    RETURN_TYPES = ("SYNC_TALK_DEEPSPEECH",)
    FUNCTION = "run"
    CATEGORY = "SyncTalk/Process"

    def run(self, model_path: str | None):
        if model_path is None:
            from .repos.SyncTalk.data_utils.deepspeech_features.deepspeech_store import get_deepspeech_model_file
            model_path = get_deepspeech_model_file()
        from .repos.SyncTalk.data_utils.deepspeech_features.deepspeech_features import prepare_deepspeech_net
        deepspeech = prepare_deepspeech_net(model_path)
        return (deepspeech,)

class DeepSpeechProcess:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "deepspeech": ("SYNC_TALK_DEEPSPEECH", {}),
                "wav_file": ("STRING", {}),
                "save_output": ("BOOLEAN", {
                    "default": False
                })
            }
        }
    
    RETURN_TYPES = ("SYNC_TALK_AUDIO_FEATURES", "STRING",)
    RETURN_NAMES = ("st_auds", "deepspeech_npy",)

    FUNCTION = "run"

    OUTPUT_NODE = True

    CATEGORY = "SyncTalk/Process"

    def run(self, deepspeech, wav_file: str, save_output: bool):
        assert wav_file.endswith(".wav")

        import pkg_resources
        installed_packages = [pkg.key for pkg in pkg_resources.working_set]
        if "tensorflow" not in installed_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])

        from .sync_talk_utils import aud_process_deepspeech
        windows = aud_process_deepspeech(deepspeech, wav_file)
        out_file_path = None
        if save_output:
            out_file_path = wav_file.replace(".wav", "_ds.npy")
            np.save(out_file_path, np.array(windows))

        return (windows, out_file_path, )

class LoadInferenceData:
    def __init__(self) -> None:
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {
                    "default": "path/to/data/"
                }),
                "preload": (["none", "cpu", "gpu"], {})
            }
        }
    
    RETURN_TYPES = ("SYNC_TALK_DATA_LOADER",)
    RETURN_NAMES = ("st_data",)

    FUNCTION = "run"

    CATEGORY = "SyncTalk"

    def run(self, path: str, preload: str):
        from .sync_talk_utils import Opt
        opt = Opt(True)

        opt.asr = True

        match preload:
            case "none":
                preload = 0
            case "cpu":
                preload = 1
            case "gpu":
                preload = 2
        
        from .repos.SyncTalk.nerf_triplane.provider import NeRFDataset
        opt.path = path
        opt.preload = preload

        test_set = NeRFDataset(opt, device=device_type, type='train')
        # a manual fix to test on the training dataset
        test_set.training = False 
        test_set.num_rays = -1

        return (test_set,)

class LoadNeRFNetwork:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_path": ("STRING", {
                    "default": "path/to/checkpoint"
                }),
                "asr_model": (asr_models, {})
            }
        }

    RETURN_TYPES = ("SYNC_TALK_NERF_NETWORK",)
    RETURN_NAMES = ("st_nerf",)

    FUNCTION = "run"
    
    CATEGORY = "SyncTalk"
    
    def run(self, checkpoint_path: str, asr_model: str):
        from .sync_talk_utils import Opt
        opt = Opt(True)
        opt.asr_model = asr_model

        from .repos.SyncTalk.nerf_triplane.network import NeRFNetwork
        model = NeRFNetwork(opt)
        model.to(device_type)

        checkpoint_dict = torch.load(checkpoint_path, map_location=device_type)

        if 'model' not in checkpoint_dict:
            model.load_state_dict(checkpoint_dict)
            Logger.log("[INFO] loaded bare model.")
            return (model,)
        
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_dict['model'], strict=False)
        Logger.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            Logger.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            Logger.log(f"[WARN] unexpected keys: {unexpected_keys}") 
        
        if 'mean_count' in checkpoint_dict:
            model.mean_count = checkpoint_dict['mean_count']
        if 'mean_density' in checkpoint_dict:
            model.mean_density = checkpoint_dict['mean_density']
        if 'mean_density_torso' in checkpoint_dict:
            model.mean_density_torso = checkpoint_dict['mean_density_torso']
        
        epoch = checkpoint_dict['epoch']
        global_step = checkpoint_dict['global_step']
        Logger.log(f"[INFO] load at epoch {epoch}, global step {global_step}")
        return (model,)

class LoadAudioFile:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aud": ("STRING", {
                    "default": "path/to/audio/file"
                }),
                "asr_model": (asr_models, {})
            }
        }
    
    RETURN_TYPES = ("SYNC_TALK_AUDIO_FEATURES",)

    FUNCTION = "run"

    # OUTPUT_NODE = True

    CATEGORY = "SyncTalk"

    def run(self, aud: str, asr_model: str):
        from .sync_talk_utils import get_aud_file_features
        auds = get_aud_file_features(aud, asr_model)
        return auds

class Inference:
    def __init__(self) -> None:
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "st_data": ("SYNC_TALK_DATA_LOADER", {}),
                "st_nerf": ("SYNC_TALK_NERF_NETWORK", {}),
                "st_auds": ("SYNC_TALK_AUDIO_FEATURES", {}),
                "start": ("INT", {
                    "default": 0,
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "run"

    # OUTPUT_NODE = True

    CATEGORY = "SyncTalk"

    def run(self, st_data, st_nerf, st_auds, start:int):
        from .sync_talk_utils import Opt, inference_auds
        opt = Opt(True)
        all_preds = inference_auds(st_nerf, st_data, st_auds, opt, start=start, device_type=device_type)

        return (all_preds,)