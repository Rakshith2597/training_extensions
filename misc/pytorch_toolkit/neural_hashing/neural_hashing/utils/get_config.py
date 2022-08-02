import os
import json


def get_config(action, model):
    """ action: train, test, export or gdrive
    """
    root_path = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.realpath(_file_))))
    config_path = os.path.join(root_path, 'configs')

    if action == 'download':
        with open(os.path.join(config_path, 'download_configs.json')) as f1:
            config = json.load(f1)
    else:
        if model == 'encoder':
            with open(os.path.join(config_path, 'encoder_config.json')) as f1:
                config_file = json.load(f1)
            config = config_file[action]
        elif model == 'classifier1':
            with open(os.path.join(config_path, 'classifier1_config.json')) as f1:
                config_file = json.load(f1)
            config = config_file[action]
        elif model == 'decoder':
            with open(os.path.join(config_path, 'decoder_config.json')) as f1:
                config_file = json.load(f1)
            config = config_file[action]

    return config
