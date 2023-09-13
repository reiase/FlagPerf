# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# DO NOT MODIFY THESE REQUIRED PARAMETERS

# Required parameters
vendor: str = None
data_dir: str = None
name: str = "transformer_xl"
cudnn_benchmark: bool = False
cudnn_deterministic: bool = True

# Optional parameters

# =========================================================
# loss scale
# =========================================================
lr: float = 5e-5
weight_decay = 0.0

# =========================================================
# train && evaluate
# =========================================================
train_batch_size: int = 4
eval_batch_size: int = 4

max_steps: int = None
max_epoch: int = 3
target_acc: float = 0.903

do_train = True
distributed: bool = True


# =========================================================
# utils
# =========================================================
seed: int = 0
num_epochs_to_generate_seeds_for: int = 2
dist_backend: str = 'nccl'
device: str = None

# =========================================================
# datasets
# =========================================================
dataloader_drop_last: bool = False
dataloader_num_workers: int = 8
dataset_config_name: str = "wikitext-103-raw-v1"

# =========================================================
# for driver
# =========================================================
local_rank: int = -1
use_env: bool = True
log_freq: int = 1000
print_freq: int = 1000
n_device: int = 1
sync_bn: bool = False
gradient_accumulation_steps: int = 1
