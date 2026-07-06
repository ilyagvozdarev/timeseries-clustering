import json, os, yaml


def read(file, reader):
    with open(file, encoding="utf-8") as f:
        try:
            return reader(f)
        except Exception as e:
            print(f'file = {file}')
            raise e    

def read_json(file):
    return read(file, json.load)

def read_yaml(file):
    return read(file, yaml.safe_load)

def read_config(file):
    ext = os.path.splitext(file)[-1].lower()
    exts = ['.json', '.yaml', '.yml']
    assert ext in exts, f'config ext not in {exts}'
    return (read_json if ext == '.json' else read_yaml)(file)