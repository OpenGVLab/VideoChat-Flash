import json
import os
import builtins
import datetime
import time
import subprocess

import torch
import torch.distributed as dist






def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def setup_for_distributed(is_master):
    builtin_print = builtins.print
    
    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        # force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def init_distributed_mode(use_dynamic_port: bool = True):
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = rank % torch.cuda.device_count()

        world_size = int(os.environ["SLURM_NTASKS"])
        try:
            local_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
        except:
            local_size = int(os.environ.get("LOCAL_SIZE", 1))

        if "MASTER_PORT" not in os.environ:
            port = 10023  # + random.randint(0, 20)
            # if use_dynamic_port:
            #     for i in range(10042, 65535):
            #         cmd = f"netstat -aon|grep {i}"
            #         with os.popen(cmd, "r") as file:
            #             if file.read() == "":
            #                 port = i
            #                 break

            print(f"MASTER_PORT = {port}")
            os.environ["MASTER_PORT"] = str(port)

            time.sleep(3)

        node_list = os.environ["SLURM_STEP_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr

        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["LOCAL_WORLD_SIZE"] = str(local_size)
        os.environ["WORLD_SIZE"] = str(world_size)

    else:
        rank = int(os.environ["RANK"])

    setup_for_distributed(rank == 0)

    print(
        f"Rank {os.environ['RANK']} | Local Rank {os.environ['LOCAL_RANK']} | "
        f"World Size {os.environ['WORLD_SIZE']} | Local World Size {os.environ['LOCAL_WORLD_SIZE']} |",
        force=True
    )


