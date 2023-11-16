

import yaml
from yaml import SafeLoader as yaml_Loader, SafeDumper as yaml_Dumper


def str_presenter(dumper, data):
  if len(data.splitlines()) > 1:  # check for multiline string
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
  return dumper.represent_scalar('tag:yaml.org,2002:str', data)
yaml.representer.SafeRepresenter.add_representer(str, str_presenter)

def safe_load(file_name, **kwargs):
    with open(file_name, 'r') as fp:
        return yaml.load(fp, Loader=yaml_Loader, **kwargs)
      
def safe_load_str(string, **kwargs):
    return yaml.load(string, Loader=yaml_Loader, **kwargs)

def safe_dump(data, file_name, sort_keys=False, **kwargs):
    with open(file_name, 'w') as fp:
        return yaml.dump(data, fp, Dumper=yaml_Dumper, sort_keys=sort_keys, **kwargs)

def read_config_from_file(config_file):
    return safe_load(config_file)

def save_config_to_file(config, config_file):
    return safe_dump(config, config_file)


