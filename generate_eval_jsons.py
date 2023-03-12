import os
import subprocess
import re
import sys

ROOT_DIR = "/home/aldo/mcs_ball_generation/"
python_path = "/home/aldo/miniconda3/envs/mcs/bin/python"
DATA_PATH = 'eval_tasks.txt'

def spawn_process(example_config, n, directory):
    cmd = f"{python_path} {ROOT_DIR}mcs-scene-generator/ile.py -c {example_config} -n {n} -p {ROOT_DIR}{directory}/scene_"
    print(cmd)
    with open(os.path.join(directory, 'output.log'), 'w') as f:
        subprocess.Popen(["nohup", "bash", "-c", cmd, "&"], stdout=f, stderr=subprocess.STDOUT)

def read_data(filename):
    with open(filename) as f:
        headers = f.readline().strip().split()
        data = []
        for line in f:
            row = line.strip().split('\t')
            row_dict = {headers[i]: value for i, value in enumerate(row)}
            data.append(row_dict)
    return data

def create_directories_and_spawn_processes(data):
    for d in data:
        directory = f"eval_scenes/{d['eval_num']}/{d['task_name']}"
        directory = directory.replace(" ", "_").lower()
        directory = re.sub('[^0-9a-zA-Z_/]+', '', directory)
        os.makedirs(ROOT_DIR + directory, exist_ok=True)
        os.makedirs(directory, exist_ok=True)
        spawn_process(d['example_config'], 100, directory)

if __name__ == "__main__":
    data = read_data(DATA_PATH)
    create_directories_and_spawn_processes(data)
