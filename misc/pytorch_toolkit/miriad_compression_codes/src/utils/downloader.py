from .get_config import get_config
import os
import zipfile
import gdown


def download_and_extract(path, url, expath):
    gdown.download(url, path, fuzzy=True)
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(expath)


def download_checkpoint(phase):
    config = get_config(action='download', phase=phase, config_path='configs/')
    if not os.path.exists('model_weights'):
        os.makedirs('model_weights')
    stage1_url = config['stage1']['drive_url']
    stage1_path = config['stage1']['dest_path']
    stage2_url = config['stage2']['drive_url']
    stage2_path = config['stage2']['dest_path']
    download_and_extract(path=stage1_path, url=stage1_url,
                         expath='model_weights/')
    download_and_extract(path=stage2_path, url=stage2_url,
                         expath='model_weights/')


def download_data(phase):
    config = get_config(action='download', phase = phase, config_path='configs/')
    if not os.path.exists('test_data'):
        os.makedirs('test_data')
        os.makedirs(os.path.join('test_data', 'prepared'))
        os.makedirs(os.path.join('test_data', 'prepared', 'bags'))
    data_url = config['test_data']['drive_url']
    data_path = config['test_data']['dest_path']
    download_and_extract(path=data_path, url=data_url,
                         expath='test_data/rbis_ddsm_sample/')