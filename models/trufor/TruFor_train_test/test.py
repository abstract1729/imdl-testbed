# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
#
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt

"""
Created in September 2022
@author: fabrizio.guillaro
"""

import sys, os
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
import csv
from datetime import datetime
from pathlib import Path

import torch
from torch.nn import functional as F

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

from lib.config import config, update_config
from lib.utils import get_model
from dataset.dataset_test import TestDataset

parser = argparse.ArgumentParser(description='Test TruFor')
parser.add_argument('-g',   '--gpu',     type=int, default=0, help='device, use -1 for cpu')
parser.add_argument('-in',  '--input',   type=str, default='../images', help='can be a single file, a directory or a glob statement')
parser.add_argument('-out', '--output',  type=str, default='../output', help='output folder')
parser.add_argument('-exp', '--experiment', type=str, default='trufor_ph3')
parser.add_argument('-save_np', '--save_np', action='store_true', help='whether to save the Noiseprint++ or not')
parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()
update_config(config, args)

input   = args.input
output  = args.output
gpu     = args.gpu
save_np = args.save_np

# device = 'cuda:%d' % gpu if gpu >= 0 else 'cpu'

if torch.backends.mps.is_available():
    device = "mps"
elif gpu >= 0 and torch.cuda.is_available():
    device = f"cuda:{gpu}"
else:
    device = "cpu"

if device.startswith("cuda"):
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

if device != 'cpu':
    # cudnn setting
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

if '*' in input:
    list_img = glob(input, recursive=True)
    list_img = [img for img in list_img if not os.path.isdir(img)]
elif os.path.isfile(input):
    list_img = [input]
elif os.path.isdir(input):
    list_img = glob(os.path.join(input, '**/*'), recursive=True)
    list_img = [img for img in list_img if not os.path.isdir(img)]
else:
    raise ValueError("input is neither a file or a folder")

test_dataset = TestDataset(list_img=list_img)

testloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1)   # 1 to allow arbitrary input sizes

if config.TEST.MODEL_FILE:
    model_state_file = config.TEST.MODEL_FILE
else:
    raise ValueError("Model file is not specified.")

print('=> loading model from {}'.format(model_state_file))
# checkpoint = torch.load(model_state_file, map_location=torch.device(device))
checkpoint = torch.load(model_state_file,map_location=device,weights_only=False)
print("Epoch: {}".format(checkpoint['epoch']))

model = get_model(config)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)


# ============================================================
# Diagnostic CSV Configuration (Added section)
# ============================================================

# Timestamp for this inference run
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# CSV file to store model input/output statistics
csv_path = os.path.join(output, f"official_pipeline_stats_{run_timestamp}.csv")
# Create output directory if required
os.makedirs(output, exist_ok=True)

CSV_COLUMNS = [ "image_path", "image_name",

    # ------------------------------
    # Input Tensor
    # ------------------------------
    "input_shape", "input_dtype", "input_min", "input_max", "input_mean", "input_std",

    # ------------------------------
    # Model Outputs
    # ------------------------------
    "score","map_shape","map_min","map_max","map_mean","map_std",
    "conf_shape","conf_min","conf_max","conf_mean","conf_std",
    "np_shape","np_min","np_max","np_mean","np_std"]

csv_file = open(csv_path,mode="w",newline="")
csv_writer = csv.DictWriter(csv_file,fieldnames=CSV_COLUMNS)
csv_writer.writeheader()
print(f"\nDiagnostic CSV: {csv_path}\n")

# ============================================================

with torch.no_grad():
    for index, (rgb, path) in enumerate(tqdm(testloader)):
        if os.path.splitext(os.path.basename(output))[1] == '':  # output is a directory
            path = path[0]
            root = input.split('*')[0]

            if os.path.isfile(input):
                sub_path = path.replace(os.path.dirname(root), '').strip()
            else:
                sub_path = path.replace(root, '').strip()

            if sub_path.startswith('/'):
                sub_path = sub_path[1:]

            filename_out = os.path.join(output, sub_path) + '.npz'
        else:  # output is a filename
            filename_out = output

        if not filename_out.endswith('.npz'):
            filename_out = filename_out + '.npz'

        # by default it does not overwrite
        if not (os.path.isfile(filename_out)):
            try:
                rgb = rgb.to(device)

                # ============================================================
                # Collect Input Tensor Statistics
                # ============================================================

                input_shape = tuple(rgb.shape)
                input_dtype = str(rgb.dtype)

                input_min = rgb.min().item()
                input_max = rgb.max().item()

                input_mean = rgb.mean().item()
                input_std = rgb.std().item()

                # ============================================================

                model.eval()

                det  = None
                conf = None

                pred, conf, det, npp = model(rgb, save_np=save_np)

                if conf is not None:
                    conf = torch.squeeze(conf, 0)
                    conf = torch.sigmoid(conf)[0]
                    conf = conf.cpu().numpy()

                if npp is not None:
                    npp = torch.squeeze(npp, 0)[0]
                    npp = npp.cpu().numpy()

                if det is not None:
                    det_sig = torch.sigmoid(det).item()

                pred = torch.squeeze(pred, 0)
                pred = F.softmax(pred, dim=0)[1]
                pred = pred.cpu().numpy()


                # ============================================================
                # Collect Output Statistics
                # ============================================================

                map_shape = pred.shape
                map_min = float(pred.min())
                map_max = float(pred.max())
                map_mean = float(pred.mean())
                map_std = float(pred.std())


                if conf is not None:

                    conf_shape = conf.shape
                    conf_min = float(conf.min())
                    conf_max = float(conf.max())
                    conf_mean = float(conf.mean())
                    conf_std = float(conf.std())

                else:

                    conf_shape = ""
                    conf_min = ""
                    conf_max = ""
                    conf_mean = ""
                    conf_std = ""


                if save_np and npp is not None:

                    np_shape = npp.shape
                    np_min = float(npp.min())
                    np_max = float(npp.max())
                    np_mean = float(npp.mean())
                    np_std = float(npp.std())

                else:

                    np_shape = ""
                    np_min = ""
                    np_max = ""
                    np_mean = ""
                    np_std = ""

                # ============================================================

                out_dict = dict()
                out_dict['map'    ] = pred
                out_dict['imgsize'] = tuple(rgb.shape[2:])
                if det is not None:
                    out_dict['score'] = det_sig
                if conf is not None:
                    out_dict['conf'] = conf
                if save_np:
                    out_dict['np++'] = npp

                from os import makedirs
                makedirs(os.path.dirname(filename_out), exist_ok=True)
                
                # ============================================================
                # Save Diagnostic Statistics
                # ============================================================

                csv_writer.writerow(
                    {
                        "image_path": path,
                        "image_name": Path(path).stem,

                        "input_shape": input_shape,
                        "input_dtype": input_dtype,
                        "input_min": input_min,
                        "input_max": input_max,
                        "input_mean": input_mean,
                        "input_std": input_std,

                        "score": det_sig if det is not None else "",

                        "map_shape": map_shape,
                        "map_min": map_min,
                        "map_max": map_max,
                        "map_mean": map_mean,
                        "map_std": map_std,

                        "conf_shape": conf_shape,
                        "conf_min": conf_min,
                        "conf_max": conf_max,
                        "conf_mean": conf_mean,
                        "conf_std": conf_std,

                        "np_shape": np_shape,
                        "np_min": np_min,
                        "np_max": np_max,
                        "np_mean": np_mean,
                        "np_std": np_std,
                    }
                )

                csv_file.flush()

                # ============================================================


                np.savez(filename_out, **out_dict)
            except:
                import traceback
                traceback.print_exc()
                pass

# ============================================================
# Close Diagnostic CSV
# ============================================================

csv_file.close()
print(f"\nDiagnostic statistics saved to:\n{csv_path}")
