import os, sys, shutil, subprocess
import json, jsonlines
import contextlib, joblib
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pandas as pd


# REF: https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def load_jsonl(file_name):
    data = []
    with jsonlines.open(file_name) as reader:
        for obj in reader:
            data.append(obj)
    return data

def read_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def write_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def get_module_by_index(data_list, index):
    res = [data_list[i] for i in range(len(data_list)) if data_list[i]['index'] == index]
    assert len(res) == 1, print(f"Child with 'index' key value: {index}, has {res} matches in dataset.")
    res = res[0]
    return res


def find_all_dependency_modules(data_list, starting_index_list, ignore_module_not_found_error=True):
    dep_modules_set = set()
    visited_index_set = set()
    bfs_index_queue = [i for i in starting_index_list]
    while len(bfs_index_queue) > 0:
        _cur_index = bfs_index_queue.pop(0)
        visited_index_set.add(_cur_index)
        try:
            _cur_module = get_module_by_index(data_list=data_list, index=_cur_index)
            dep_modules_set.add(_cur_module['full_code'])
            for _c in _cur_module['children']:
                if _c not in visited_index_set:
                    bfs_index_queue.append(_c)
        except Exception as e:
            if not ignore_module_not_found_error:
                raise RuntimeError(e)
    return list(dep_modules_set)


def init_verilator_env():
    bash_cmd = "enroot create -n ubuntu2404 /mnt/shared/gpfs/escad_verilog_dataset/verilator_coverage_jinghua/enroot_utils/enroot_images/ubuntu_2404.sqsh"
    return subprocess.run([bash_cmd], shell=True, capture_output=True)


def run_verilator(modules_full_code_list, tmp_dir, force_rerun=False, timeout_seconds=120):
    if os.path.isdir(tmp_dir) and not force_rerun:
        return (tmp_dir, "dir already exists, assuming done already, skipping...")
    VERILATOR_PATH = "/mnt/shared/gpfs/tools/verilator/bin/verilator"
    VERILATOR_MOUNT_PATH = "/mnt/shared/gpfs/tools/verilator/"
    VERILOG_FILE_NAME = "src.v"
    TB_BINARY_RELATIVE_PATH = 'obj_dir'
    TB_EXECUTABLE = "Vsrc"
    WORKDIR_PATH  = tmp_dir
    ENROOT_CMD = f"enroot start --root -w --mount {WORKDIR_PATH}:{WORKDIR_PATH} --mount {VERILATOR_MOUNT_PATH}:{VERILATOR_MOUNT_PATH} --mount /usr/:/usr/ --mount /etc/:/etc/ ubuntu2404"
    VERILATOR_CMD = VERILATOR_PATH + f" --Mdir {os.path.join(tmp_dir, 'obj_dir')} --binary --coverage --no-timing -Wno-style -Wno-fatal  --bbox-sys  --bbox-unsup"
    RUN_CMD = f'{ENROOT_CMD}  "{VERILATOR_CMD} {os.path.join(tmp_dir, VERILOG_FILE_NAME)} && cd {os.path.join(tmp_dir, TB_BINARY_RELATIVE_PATH)} && ./{TB_EXECUTABLE}"'
    # create verilog file in tmp_dir and (over)write all modules inmodules_full_code_list
    os.makedirs(tmp_dir, exist_ok=True)
    with open(os.path.join(tmp_dir, VERILOG_FILE_NAME) , 'w') as f:
        f.write("\n\n".join(modules_full_code_list))
    # execute RUN_CMD
    try:
        cmd_result = subprocess.run([RUN_CMD], shell=True, cwd=tmp_dir, capture_output=True, timeout=timeout_seconds)
    except Exception as e:
        cmd_result = "ERROR: "+str(e)
    return (tmp_dir, cmd_result)


def parse_coverage_log(filepath):
    # extract the ending ints in coverage report file, one int per line.
    with open(filepath, 'r') as f:
        lines = f.readlines()
    coverage_results = []
    for l in lines:
        try:
            coverage_results.append(int(l.split(" ")[-1]))
        except:
            pass
    return coverage_results



# enroot start --root -w  --mount /mnt/shared/gpfs/tools/verilator/:/mnt/shared/gpfs/tools/verilator/ --mount /usr/:/usr/ --mount /etc/:/etc/ --mount /mnt/shared/gpfs/escad_verilog_dataset/verilator_coverage_jinghua/run_results:/mnt/shared/gpfs/escad_verilog_dataset/verilator_coverage_jinghua/run_results ubuntu2404
   
# /mnt/shared/gpfs/tools/verilator/bin/verilator_coverage -write-info coverage.info coverage.dat

# /mnt/shared/gpfs/tools/verilator/bin/verilator_coverage --write coverage.txt coverage.dat

# /mnt/shared/gpfs/tools/verilator/bin/verilator_coverage –annotate /mnt/shared/gpfs/escad_verilog_dataset/verilator_coverage_jinghua/run_results/O100_manual_0/obj_dir/coverage.dat /mnt/shared/gpfs/escad_verilog_dataset/verilator_coverage_jinghua/run_results/O100_manual_0/obj_dir1 