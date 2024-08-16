import os
from .utils import *
from .sync_talk_nodes import *

class InstallStatus:
    def __init__(self, 
                 git_clone_sync_talk: bool, 
                 install_dependencies: bool):
        self.git_clone_sync_talk = git_clone_sync_talk
        self.install_dependencies = install_dependencies

    def to_dict(self):
        return {
            "git_clone_sync_talk": self.git_clone_sync_talk,
            "install_dependencies": self.install_dependencies
        }


# Prepare directories.
cur_dir = os.path.dirname(__file__)
install_status_path = os.path.join(cur_dir, "install_status.json")
sync_talk_dir = os.path.join(cur_dir, "repos/SyncTalk")
pytorch3d_dir = os.path.join(cur_dir, "repos/pytorch3d")
if not os.path.isdir(sync_talk_dir):
    os.makedirs(sync_talk_dir)
if not os.path.isdir(pytorch3d_dir):
    os.makedirs(pytorch3d_dir)

install_status = utils.read_json_as_class(install_status_path, InstallStatus)

# Clone forked SyncTalk repo.
if not install_status.git_clone_sync_talk:
    utils.clone_repository('https://github.com/Ryuukeisyou/SyncTalk.git', sync_talk_dir)
    install_status.git_clone_sync_talk = True
    utils.export_class_to_json(install_status, install_status_path)
else:
    pass
    # utils.pull_repository(sync_talk_dir)


# Install dependencies.
if install_status.git_clone_sync_talk and not install_status.install_dependencies:
    
    # Install portaudio19-dev.
    if not utils.is_package_installed("portaudio19-dev"):
        subprocess.check_call(["sudo", "apt-get", "install", "portaudio19-dev"])
    
    # Install requirements.txt
    sync_talk_requirements_path = os.path.join(sync_talk_dir, "requirements.txt")
    other_requirements_path = os.path.join(cur_dir, "requirements.txt")
    utils.install_requirements(sync_talk_requirements_path)
    utils.install_requirements(other_requirements_path)
    installed_packages = [pkg.key for pkg in pkg_resources.working_set]
    
    # Install pytorch3d
    if "pytorch3d" not in installed_packages:
        # Check if wheel is built.
        pytorch3d_wheel = utils.get_valid_wheel("pytorch3d", os.path.join(pytorch3d_dir, "dist"))
        if pytorch3d_wheel is None:
            # Build wheel.
            utils.clone_repository('https://github.com/facebookresearch/pytorch3d.git', pytorch3d_dir)
            pytorch3d_setup_path = os.path.join(pytorch3d_dir, "setup.py")
            subprocess.check_call([sys.executable, pytorch3d_setup_path, "bdist_wheel"], cwd=pytorch3d_dir)
            pytorch3d_wheel = utils.get_valid_wheel("pytorch3d", os.path.join(pytorch3d_dir, "dist"))   
        if pytorch3d_wheel is not None: 
            subprocess.check_call([sys.executable, "-m", "pip", "install", pytorch3d_wheel])
           
    # Install extensions.
    ext_list = ["shencoder", "freqencoder", "gridencoder", "raymarching-face"]
    missing_exts = set(ext_list) - set(installed_packages)
    if missing_exts:
        for ext in missing_exts:
            dirname = "raymarching" if ext == "raymarching-face" else ext
            ext_dir = os.path.join(sync_talk_dir, dirname)
            ext_wheel = utils.get_valid_wheel(ext, os.path.join(ext_dir, "dist"))
            if ext_wheel is None:
                # Build wheel.
                ext_setup_path = os.path.join(ext_dir, "setup.py")
                subprocess.check_call([sys.executable, ext_setup_path, "bdist_wheel"], cwd=ext_dir)
                ext_wheel = utils.get_valid_wheel(ext, os.path.join(ext_dir, "dist"))
            if ext_wheel is not None:
                subprocess.check_call([sys.executable, "-m", "pip", "install", ext_wheel])
    
    # Check installations.
    all_required_packages = ["pytorch3d"] + ext_list
    with open(sync_talk_requirements_path) as f:
        all_required_packages += f.read().splitlines()
    with open(other_requirements_path) as f:
        all_required_packages += f.read().splitlines()
    all_required_packages = [s.lower().replace("_", "-") for s in all_required_packages]
    installed_packages = [pkg.key for pkg in pkg_resources.working_set]
    missing_packages = set(all_required_packages) - set(installed_packages)
    if len(missing_packages) == 0:
        install_status.install_dependencies = True
        utils.export_class_to_json(install_status, install_status_path)
    

NODE_CLASS_MAPPINGS = {
    f'{LoadAve.__name__}(SyncTalk)': LoadAve,
    f'{AveProcess.__name__}(SyncTalk)': AveProcess,

    f'{LoadHubert.__name__}(SyncTalk)': LoadHubert,
    f'{HubertProcess.__name__}(SyncTalk)': HubertProcess,
    
    f'{LoadDeepSpeech.__name__}(SyncTalk)': LoadDeepSpeech,
    f'{DeepSpeechProcess.__name__}(SyncTalk)': DeepSpeechProcess,

    f'{LoadInferenceData.__name__}(SyncTalk)': LoadInferenceData,
    f'{LoadNeRFNetwork.__name__}(SyncTalk)': LoadNeRFNetwork,
    f'{Inference.__name__}(SyncTalk)': Inference
}

__all__ = ['NODE_CLASS_MAPPINGS']
