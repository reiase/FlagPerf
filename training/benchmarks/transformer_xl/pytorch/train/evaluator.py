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

import torch
from torch.types import Device

from .accuracy import Accuracy


class Evaluator:
    def __init__(self, config, dataloader):
        self.config = config
        self.dataloader = dataloader

        self.metric = Accuracy()

    def compute_metrics(preds, labels):
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return self.metric.compute(predictions=preds, references=labels)

    def process_batch(self, batch, device: Device):
        """Process batch and produce inputs for the model."""
        for k, v in batch.items():
            batch[k] = v.to(device, non_blocking=True)
        return batch

    def evaluate(self, trainer):
        model = trainer.model
        model.eval()

        total_eval_loss = 0.0
        total_examples = 0

        with torch.no_grad():
            for batch in self.dataloader:
                batch = self.process_batch(batch, trainer.device)
                loss = model(**batch)
                total_eval_loss += loss.loss
                total_examples += loss.losses.shape[0]
                torch.cuda.synchronize()
        trainer.model.train()

        if torch.distributed.is_initialized():
            # Collect total scores from all ranks
            torch.distributed.all_reduce(
                total_eval_loss, op=torch.distributed.ReduceOp.SUM
            )
        # Average by number of examples
        total_eval_loss /= total_examples

        return total_eval_loss.item()
