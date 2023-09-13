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

from dataclasses import dataclass
from time import time

from driver import Driver
from evaluator import Evaluator
from model import create_model
from optimizer import create_optimizer
from torch.types import Device
from train.training_state import TrainingState


def _process_batch(batch, device: Device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@dataclass
class Trainer:
    driver: Driver = (None,)
    evaluator: Evaluator = (None,)
    state: TrainingState = (None,)
    device: Device = (None,)

    def init(self, config):
        self.model, self.model_config, self.tokenizer = create_model(config)
        self.model = self.model.to(self.device)
        self.optimizer = create_optimizer(self.model, config)

    def train_one_epoch(self, dataloader):
        state = self.state
        self.model.train()
        with state.epoch_train_guard(self.driver):
            no_eval_start = time()
            for _, data in enumerate(dataloader):
                state.global_step += 1
                data = _process_batch(data, self.device)
                self.train_one_step(data)

                state.noevaltime += time() - no_eval_start
                no_eval_start = time()
        with state.epoch_eval_guard(self.driver):
            state.acc = self.evaluator.evaluate(self)
        self.detect_training_status(state)

    def train_one_step(self, data):
        with self.state.step_guard(self.driver):
            outputs = self.model(**data)
            loss = outputs["loss"]
            loss.backward()
            self.optimizer.step()
            self.state.loss = loss.item()

    def detect_training_status(self, state: TrainingState):
        if state.acc >= self.config.target_acc:
            state.converged_success()
            state.end_training = True

        if state.epoch >= self.config.max_epoch:
            state.end_training = True

        return state.end_training
