import os
import numpy as np
import torch
import tqdm
from transformers import Wav2Vec2Processor, HubertModel
from .repos.SyncTalk.nerf_triplane.network import AudioEncoder
from .repos.SyncTalk.nerf_triplane.utils import AudDataset, get_audio_features, linear_to_srgb, blend_with_mask_cuda
from .repos.SyncTalk.nerf_triplane.network import NeRFNetwork
from .repos.SyncTalk.nerf_triplane.provider import NeRFDataset
from torch.utils.data import DataLoader
from torch import Tensor
from .utils import Logger

class Opt:
    def __init__(self, O: bool) -> None:

        self.path = ''
        self.O = False
        self.test = False
        self.test_train = False
        self.data_range = [0, -1]
        self.workspace = 'workspace'
        self.seed = 0
        
        # training options
        self.iters = 200000
        self.lr = 1e-2
        self.lr_net = 1e-3
        self.ckpt = 'latest'
        self.num_rays = 4096 * 16
        self.cuda_ray = False
        self.max_steps = 16
        self.num_steps = 16
        self.upsample_steps = 0
        self.update_extra_interval = 16
        self.max_ray_batch = 4096

        # loss set
        self.warmup_step = 10000
        self.amb_aud_loss = 1
        self.amb_eye_loss = 1
        self.unc_loss = 1
        self.lambda_amb = 1e-4
        self.pyramid_loss = 0

        # network backbone options
        self.fp16 = False

        self.bg_img = ''
        self.fbg = False
        self.exp_eye = False
        self.fix_eye = -1
        self.smooth_eye = False
        self.bs_area = 'upper'
        self.au45 = False
        self.torso_shrink = 0.8

        # dataset options
        self.color_space = 'srgb'
        self.preload = 0

        # (the default value is for the fox dataset)
        self.bound = 1
        self.scale = 4
        self.offset = [0, 0, 0]
        self.dt_gamma = 1/256
        self.min_near = 0.05
        self.density_thresh = 10
        self.density_thresh_torso = 0.01
        self.patch_size = 1

        self.init_lips = False
        self.finetune_lips = False
        self.smooth_lips = False

        self.torso = False
        self.head_ckpt = ''

        # else
        self.att = 2
        self.aud = ''
        self.emb = False
        self.portrait = False
        self.ind_dim = 4
        self.ind_num = 20000

        self.ind_dim_torso = 8

        self.amb_dim = 2
        self.part = False
        self.part2 = False

        self.train_camera = False
        self.smooth_path = False
        self.smooth_path_window = 7

        # asr
        self.asr = False
        self.asr_wav = ''
        self.asr_play = False

        self.asr_model = 'deepspeech'
        self.asr_save_feats = False

        # audio FPS
        self.fps = 50

        # sliding window left-middle-right length (unit: 20ms)
        self.l = 10
        self.m = 50
        self.r = 10
        if O:
            self.cuda_ray = True
            self.fp16 = True
            self.exp_eye = True

@torch.no_grad()
def get_hubert_from_16k_speech(wav2vec2_processor, hubert_model, speech, device_type) -> Tensor:
    if speech.ndim ==2:
        speech = speech[:, 0] # [T, 2] ==> [T,]
    input_values_all = wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000).input_values # [1, T]
    input_values_all = input_values_all.to(device_type)
    # For long audio sequence, due to the memory limitation, we cannot process them in one run
    # HuBERT process the wav with a CNN of stride [5,2,2,2,2,2], making a stride of 320
    # Besides, the kernel is [10,3,3,3,3,2,2], making 400 a fundamental unit to get 1 time step.
    # So the CNN is euqal to a big Conv1D with kernel k=400 and stride s=320
    # We have the equation to calculate out time step: T = floor((t-k)/s)
    # To prevent overlap, we set each clip length of (K+S*(N-1)), where N is the expected length T of this clip
    # The start point of next clip should roll back with a length of (kernel-stride) so it is stride * N
    kernel = 400
    stride = 320
    clip_length = stride * 1000
    num_iter = input_values_all.shape[1] // clip_length
    expected_T = (input_values_all.shape[1] - (kernel-stride)) // stride
    res_lst = []
    for i in range(num_iter):
        if i == 0:
            start_idx = 0
            end_idx = clip_length - stride + kernel
        else:
            start_idx = clip_length * i
            end_idx = start_idx + (clip_length - stride + kernel)
        input_values = input_values_all[:, start_idx: end_idx]
        hidden_states = hubert_model.forward(input_values).last_hidden_state # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    if num_iter > 0:
        input_values = input_values_all[:, clip_length * num_iter:]
    else:
        input_values = input_values_all
    # if input_values.shape[1] != 0:
    if input_values.shape[1] >= kernel: # if the last batch is shorter than kernel_size, skip it            
        hidden_states = hubert_model(input_values).last_hidden_state # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    ret = torch.cat(res_lst, dim=0).cpu() # [T, 1024]
    # assert ret.shape[0] == expected_T
    assert abs(ret.shape[0] - expected_T) <= 1
    if ret.shape[0] < expected_T:
        ret = torch.nn.functional.pad(ret, (0,0,0,expected_T-ret.shape[0]))
    else:
        ret = ret[:expected_T]
    return ret

def make_even_first_dim(tensor: Tensor) -> Tensor:
    size = list(tensor.size())
    if size[0] % 2 == 1:
        size[0] -= 1
        return tensor[:size[0]]
    return tensor

def aud_process_ave(model: AudioEncoder, wav_file: str, emb=False, device_type='cuda') -> Tensor:
    dataset = AudDataset(wav_file)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    outputs = []
    for mel in data_loader:
        mel = mel.to(device_type)
        with torch.no_grad():
            out = model(mel)
            outputs.append(out)
    outputs = torch.cat(outputs, dim=0).cpu()
    first_frame, last_frame = outputs[:1], outputs[-1:]
    aud_features = torch.cat([first_frame.repeat(2, 1), outputs, last_frame.repeat(2, 1)], dim=0).unsqueeze(0)
    
    # support both [N, 16] labels and [N, 16, K] logits
    if len(aud_features.shape) == 3:
        aud_features = aud_features.float().permute(1, 0, 2)  # [N, 16, 29] --> [N, 29, 16]

        if emb:
            print(f'[INFO] argmax to aud features {aud_features.shape} for --emb mode')
            aud_features = aud_features.argmax(1)  # [N, 16]

    else:
        assert emb, "aud only provide labels, must use --emb"
        aud_features = aud_features.long()

    Logger.log(f'[INFO] load {wav_file} aud_features: {aud_features.shape}')

    return aud_features

def aud_process_hubert(wav2vec2_processor: Wav2Vec2Processor, hubert_model: HubertModel, wav_file: str, emb=False, device_type='cuda') -> Tensor:
    assert wav_file.endswith(".wav")
    
    import soundfile as sf
    import librosa
    from .sync_talk_utils import get_hubert_from_16k_speech, make_even_first_dim

    speech, sr = sf.read(wav_file)
    speech_16k = librosa.resample(speech, orig_sr=sr, target_sr=16000)
    Logger.log("SR: {} to {}".format(sr, 16000))

    hubert_hidden = get_hubert_from_16k_speech(wav2vec2_processor, hubert_model, speech_16k, device_type)
    hubert_hidden = make_even_first_dim(hubert_hidden).reshape(-1, 2, 1024)

    aud_features = hubert_hidden
    # support both [N, 16] labels and [N, 16, K] logits
    if len(aud_features.shape) == 3:
        aud_features = aud_features.float().permute(0, 2, 1)  # [N, 16, 29] --> [N, 29, 16]

        if emb:
            print(f'[INFO] argmax to aud features {aud_features.shape} for --emb mode')
            aud_features = aud_features.argmax(1)  # [N, 16]

    else:
        assert emb, "aud only provide labels, must use --emb"
        aud_features = aud_features.long()

    Logger.log(f'[INFO] load {wav_file} aud_features: {aud_features.shape}')

    return aud_features

def aud_process_deepspeech(deepspeech, wav_file: str, emb=False) -> Tensor:
    import tensorflow as tf
    from scipy.io import wavfile
    from .repos.SyncTalk.data_utils.deepspeech_features.deepspeech_features import pure_conv_audio_to_deepspeech

    graph, logits_ph, input_node_ph, input_lengths_ph = deepspeech

    audio_file_path = wav_file
    
    audio_window_size=1,
    audio_window_stride=1
    
    with tf.compat.v1.Session(graph=graph) as session:
        audio_sample_rate, audio = wavfile.read(audio_file_path)
        if audio.ndim != 1:
            Logger.log(
                "[WARN] Audio has multiple channels, the first channel is used")
            audio = audio[:, 0]
        ds_features = pure_conv_audio_to_deepspeech(
            audio=audio,
            audio_sample_rate=audio_sample_rate,
            audio_window_size=audio_window_size,
            audio_window_stride=audio_window_stride,
            num_frames=None,
            net_fn=lambda x: session.run(
                logits_ph,
                feed_dict={
                    input_node_ph: x[np.newaxis, ...],
                    input_lengths_ph: [x.shape[0]]}))

        net_output = ds_features.reshape(-1, 29)
        win_size = 16
        zero_pad = np.zeros((int(win_size / 2), net_output.shape[1]))
        net_output = np.concatenate(
            (zero_pad, net_output, zero_pad), axis=0)
        windows = []
        for window_index in range(0, net_output.shape[0] - win_size, 2):
            windows.append(
                net_output[window_index:window_index + win_size])
    
    aud_features = windows
    # support both [N, 16] labels and [N, 16, K] logits
    if len(aud_features.shape) == 3:
        aud_features = aud_features.float().permute(0, 2, 1)  # [N, 16, 29] --> [N, 29, 16]

        if emb:
            print(f'[INFO] argmax to aud features {aud_features.shape} for --emb mode')
            aud_features = aud_features.argmax(1)  # [N, 16]

    else:
        assert emb, "aud only provide labels, must use --emb"
        aud_features = aud_features.long()

    Logger.log(f'[INFO] load {wav_file} aud_features: {aud_features.shape}')

    return aud_features

def get_aud_file_features(aud: str, asr_model: str, emb = False) -> Tensor:
    """
    Get audio feature from aud file.
    """
    if asr_model == 'ave':
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = AudioEncoder().to(device).eval()
            ckpt = torch.load(os.path.join(os.path.dirname(__file__), 'repos/SyncTalk/nerf_triplane/checkpoints/audio_visual_encoder.pth'))
            model.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()})
            dataset = AudDataset(aud)
            data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
            outputs = []
            for mel in data_loader:
                mel = mel.to(device)
                with torch.no_grad():
                    out = model(mel)
                outputs.append(out)
            outputs = torch.cat(outputs, dim=0).cpu()
            first_frame, last_frame = outputs[:1], outputs[-1:]
            aud_features = torch.cat([first_frame.repeat(2, 1), outputs, last_frame.repeat(2, 1)], dim=0).numpy()
        except:
            print(f'[ERROR] If do not use Audio Visual Encoder, replace it with the npy file path.')
    else:
        try:
            aud_features = np.load(aud)
        except:
            print(f'[ERROR] If do not use Audio Visual Encoder, replace it with the npy file path.')

    
    if asr_model == 'ave':
        aud_features = torch.from_numpy(aud_features).unsqueeze(0)

        # support both [N, 16] labels and [N, 16, K] logits
        if len(aud_features.shape) == 3:
            aud_features = aud_features.float().permute(1, 0, 2)  # [N, 16, 29] --> [N, 29, 16]

            if emb:
                print(f'[INFO] argmax to aud features {aud_features.shape} for --emb mode')
                aud_features = aud_features.argmax(1)  # [N, 16]

        else:
            assert emb, "aud only provide labels, must use --emb"
            aud_features = aud_features.long()

        print(f'[INFO] load {aud} aud_features: {aud_features.shape}')
    else:
        aud_features = torch.from_numpy(aud_features)

        # support both [N, 16] labels and [N, 16, K] logits
        if len(aud_features.shape) == 3:
            aud_features = aud_features.float().permute(0, 2, 1)  # [N, 16, 29] --> [N, 29, 16]

            if emb:
                print(f'[INFO] argmax to aud features {aud_features.shape} for --emb mode')
                aud_features = aud_features.argmax(1)  # [N, 16]

        else:
            assert emb, "aud only provide labels, must use --emb"
            aud_features = aud_features.long()

        print(f'[INFO] load {aud} aud_features: {aud_features.shape}')
    
    return aud_features

def inference_step(model: NeRFNetwork, data, opt, bg_color=None, perturb=False, device_type='cuda'):
    rays_o = data['rays_o'] # [B, N, 3]
    rays_d = data['rays_d'] # [B, N, 3]
    bg_coords = data['bg_coords'] # [1, N, 2]
    poses = data['poses'] # [B, 7]

    auds = data['auds'] # [B, 29, 16]
    index = data['index']
    H, W = data['H'], data['W']

    # allow using a fixed eye area (avoid eye blink) at test
    if opt.exp_eye and opt.fix_eye >= 0:
        eye = torch.FloatTensor([opt.fix_eye]).view(1, 1).to(device_type)
    else:
        eye = data['eye'] # [B, 1]

    if bg_color is not None:    
        bg_color = bg_color.to(device_type)
    else:
        bg_color = data['bg_color']

    model.testing = True
    outputs = model.render(rays_o, rays_d, auds, bg_coords, poses, eye=eye, index=index, staged=True, bg_color=bg_color, perturb=perturb, **vars(opt))
    model.testing = False

    pred_rgb = outputs['image'].reshape(-1, H, W, 3)
    # pred_depth = outputs['depth'].reshape(-1, H, W)
    
    pred_rgb = outputs['image'].reshape(-1, H, W, 3)
    # pred_depth = outputs['depth'].reshape(-1, H, W)

    return pred_rgb

@torch.no_grad()
def inference_auds(model: NeRFNetwork, dataset: NeRFDataset, auds: Tensor, opt: Opt, start=0, device_type='cuda'):
    model.eval()

    auds_size = auds.shape[0]
    
    pbar = tqdm.tqdm(total=auds_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    Logger.log(f"==> Start Inference")
    
    all_preds = []
    for i in range(auds_size):
        data = dataset.collate([start + i])
        data['auds'] = get_audio_features(auds, opt.att, i).to(device_type)
        with torch.amp.autocast(enabled=opt.fp16, device_type=device_type):
            preds = inference_step(model, data, opt, device_type=device_type)
        
        if opt.color_space == 'linear':
            preds = linear_to_srgb(preds)
        if opt.portrait:
                pred = blend_with_mask_cuda(preds[0], data["bg_gt_images"].squeeze(0), data["bg_face_mask"].squeeze(0))
                pred = (pred * 255).astype(np.uint8)
        else:
            pred = preds[0]
        
        all_preds.append(pred)
        pbar.update()
        
    return all_preds