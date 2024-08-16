import os
import sys
import json
import subprocess
import pkg_resources

class Logger:
    def __init__(self) -> None:
        pass

    @classmethod
    def log(cls, msg):
        print(f'[SyncTalk]{msg}')
        return

def read_json_as_class(filepath: str, cls):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return cls(**data)


def export_class_to_json(obj, filepath):
    with open(filepath, 'w') as file:
        json.dump(obj.to_dict(), file, indent=4)
    print(f"Object exported to {filepath}")


def clone_repository(repo_url, clone_dir):
    try:
        subprocess.check_call(["git", "clone", repo_url, clone_dir])
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e.stderr}")


def pull_repository(repo_dir):
    try:
        subprocess.check_call(["git", "-C", repo_dir, "pull"])
        print(f"Repository in {repo_dir} has been updated.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e.stderr}")

def install_requirements(requirements_path: str):
    with open(os.path.join(requirements_path)) as f:
        required_packages = [s.replace("_", "-").lower() for s in f.read().splitlines()] 
        installed_packages = [pkg.key for pkg in pkg_resources.working_set]
        missing_packages = set(required_packages) - set(installed_packages)
        if 'opencv-contrib-python' in installed_packages or 'opencv-python-headless' in installed_packages:
            missing_packages = set(missing_packages) - set(['opencv-python'])
        if missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])

def is_package_installed(package_name) -> bool:
    try:
        # Run the dpkg command to check if the package is installed
        result = subprocess.run(['dpkg', '-l', package_name], capture_output=True, text=True)
        # Check if the output contains the package name
        return package_name in result.stdout
    except FileNotFoundError:
        return False


def get_valid_wheel(package_name: str, directory: str) -> str | None:
    # Get version and platform info
    import platform, glob
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_info = platform.system().lower()
    architecture = platform.architecture()[0]
    # Map architecture to wheel filename convention
    arch_map = {
        '64bit': 'x86_64',
        '32bit': 'x86'
    }
    arch_tag = arch_map.get(architecture, '')
    whl_files = glob.glob(os.path.join(directory, "*.whl"))
    for whl_file in whl_files:
        filename = os.path.basename(whl_file)
        parts = filename.split('-')
        if package_name in parts and python_version in parts and platform_info in parts[-1] and arch_tag in parts[-1]:
            return whl_file
    return None

