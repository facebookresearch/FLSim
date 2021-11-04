#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from typing import Dict, Iterator, Tuple, Union

import numpy as np
import portalai.speakerid.spkid.models as smod
import torch
import torch.nn as nn
from flsim.interfaces.model import IFLModel
from flsim.utils.simple_batch_metrics import FLBatchMetrics
from papaya.toolkit.simulation.experimental.speaker_id.speaker_id_fl_dataset_utils import (
    SpeakerIdFLDataset,
)
from papaya.toolkit.simulation.experimental.speaker_id.speaker_id_training_utils import (
    SpeakerIdFLEvalBatchMetrics,
)
from portalai.speakerid.spkid.losses import create_loss


class SpeakerIdFLModel(IFLModel):
    def __init__(self, model: smod.AudioClassifier, device=torch.device("cpu")):
        self.model = model
        self.device = device
        self.loss_fun = create_loss()

    # pyre-fixme[14]: `fl_create_training_batch` overrides method defined in
    #  `IFLModel` inconsistently.
    def fl_create_training_batch(
        self,
        batch: Dict[str, torch.Tensor]
        # pyre-fixme[11]: Annotation `tensor` is not defined as a type.
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Takes a dictionary containing an entire batch's worth of data and converts
        it into a form consumable by the fl_forward function of the model. Note that
        the tensor associated with each string key is already batched via torch's
        stack function. This is due to the implementation of fl_train_set in
        FLDatasetDataLoaderWithBatch. This may introduce an extra dimension, which
        can be removed in this method.

        For the speaker ID model, this involves converting the dictionary of form
        {
            'data': Tensor of dim (batch_size, 1, 100, 64)
            'user_id': Tensor of dim (batch_size, 1)
        }

        to a Tuple of form:
        (Tensor of dim (batch_size), Tensor of dim (batch_size, 1, 100, 64))

        This tuple specifies the correct user ids and audio features for all the
        samples in the batch. Note that this same input is what is consumed by the
        fl_forward method specified for this model.
        """
        labels_list = torch.squeeze(batch[SpeakerIdFLDataset.shard_col_id], 1).to(
            self.device
        )
        audio_features = batch[SpeakerIdFLDataset.data_id].to(self.device)
        return (labels_list, audio_features)

    def fl_forward(self, batch: Tuple[torch.tensor, torch.tensor]) -> FLBatchMetrics:
        labels = batch[0]
        audio_features = batch[1]
        preds = self.model(audio_features, labels)
        loss = self.loss_fun(preds, labels)
        num_examples = labels.shape[0]
        return FLBatchMetrics(loss, num_examples, preds, labels, [audio_features])

    def fl_get_module(self) -> nn.Module:
        return self.model

    def parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        return self.model.parameters()

    def fl_cuda(self) -> None:
        if torch.cuda.is_available:
            if self.device not in [None, torch.device("cpu")]:
                self.model = self.model.to(self.device)
            else:
                self.model = self.model.cuda()
            self.device = next(self.parameters()).device

    def get_eval_metrics(
        self,
        batch: Union[
            Tuple[torch.tensor, torch.tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.tensor, torch.tensor],
        ],
    ) -> SpeakerIdFLEvalBatchMetrics:
        """
        Note that this method will be invoked both on the central server which has access to the eval
        set, and on clients which only have access to their train set. In the case of the speaker ID
        model, evaluation logic is different from training logic, which requires differentiation between
        the two cases. This is relatively easy since the inputs are different in form.

        The training data is a tuple of two values containing:
        (
            Tensor of dim (batch_size,) containing labels,
            Tensor of dim (batch_size, 1, 100, 64) containing the audio inputs
        )

        The eval data is a tuple of 5 values containing:
        clips for user A candidates, in a tensor of dim (batch_size, 1, 100, 64)
        clips for user B candidates in a tensor of dim (batch_size, 1, 100, 64)
        User A candidate ids in a tensor of dim (batch_size,)
        User B candidate ids in a tensor of dim (batch_size,)
        Bools indicating if User A == User B for each sample in batch in a tensor of dim (batch_size,)
        """
        with torch.no_grad():
            # TODO : @akashb ensure this eval function works properly on batch inputs rather
            # individual samples that this function currently iterates through. The behavior
            # below reflects the original implementation by portal AI, as in the run_eer function
            # in portalai.speakerid.spkid.eval . In addition to standard authentication metrics
            # like equal error rate and AUC for the binary distinction of whether two given audio
            # clips are from the same user, we additionally also evaluation standard classification
            # accuracy (as done during training). Note that the actual evaluation of metrics is
            # done by the metrics reported. This function merely provides the necessary values to
            # calculate the error metrics.
            if len(batch) > 2:
                afn_clip, bfn_clip, a_user_id, b_user_id, is_same = batch
                tot_loss = None
                all_preds = []
                all_labels = []
                all_inputs = []
                all_user_preds = []
                all_user_labels = []
                for i in range(afn_clip.shape[0]):
                    # don't unsqueeze inputs to clip_embedding! Due to 1D batch norm used in the speaker
                    # ID model, and the correction for variance that divides squared deviations from mean
                    # by (batch_size-1)*batch_size, singleton batches create NaNs. Interestingly, this
                    # doesn't happen locally on a dev GPU using dev-nosan, but happens on FB Learner. Also,
                    # do not call clip_embedding on an entire batch, but only on a singleton tensor of dim
                    # (1, 100, 64)
                    emba = self.model.clip_embedding(afn_clip[i])
                    embb = self.model.clip_embedding(bfn_clip[i])
                    apreds = self.model(afn_clip[i].unsqueeze(0), a_user_id[i])
                    aloss = self.loss_fun(apreds, a_user_id[i])
                    bpreds = self.model(bfn_clip[i].unsqueeze(0), b_user_id[i])
                    bloss = self.loss_fun(bpreds, b_user_id[i])
                    y_score = np.dot(emba, embb) / (
                        np.linalg.norm(emba) * np.linalg.norm(embb)
                    )
                    y_true = 1 if is_same[i] else 0
                    all_preds.append(y_score)
                    all_labels.append(y_true)
                    all_inputs.append([emba, embb])
                    all_user_preds.append(apreds)
                    all_user_preds.append(bpreds)
                    all_user_labels.append(a_user_id[i])
                    all_user_labels.append(b_user_id[i])
                    if tot_loss is None:
                        tot_loss = (aloss + bloss) / 2.0
                    else:
                        tot_loss += (aloss + bloss) / 2.0
                return SpeakerIdFLEvalBatchMetrics(
                    tot_loss,
                    afn_clip.shape[0],
                    # pyre-fixme[6]: Expected `Tensor` for 3rd param but got
                    #  `List[typing.Any]`.
                    all_preds,
                    # pyre-fixme[6]: Expected `Tensor` for 4th param but got
                    #  `List[typing.Any]`.
                    all_labels,
                    all_inputs,
                    loss_only_eval=False,
                    eval_time_user_predictions=torch.cat(all_user_preds),
                    eval_time_user_labels=torch.cat(all_user_labels),
                )
            else:
                labels = batch[0]
                audio_features = batch[1]
                preds = self.model(audio_features, labels)
                loss = self.loss_fun(preds, labels)
                num_examples = labels.shape[0]
                return SpeakerIdFLEvalBatchMetrics(
                    loss,
                    num_examples,
                    preds,
                    labels,
                    [audio_features],
                    loss_only_eval=True,
                )

    def get_num_examples(self, batch: Tuple[torch.tensor, torch.tensor]) -> int:
        """
        Used to evaluate the number of train samples per client. It should consume the same
        type of input as the return value of fl_create_training_batch. Note that these samples
        are batched, so you'd have to accumulate the size of the batch.
        """
        return batch[0].shape[0]

    def fl_create_eval_batch(
        self,
        batch: Tuple[
            torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor
        ],
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        The input is assembled by a pytorch data loader already and corresponds to a batch. Specifically,
        the input is of form:
        (
            Tensor for clip As of dim (batch_size, 1, 100, 64),
            Tensor for clip Bs of dim (batch_size, 1, 100, 64),
            Tensor for gold labels for clip As of dim (batch_size, 1)
            Tensor for gold labels for clip Bs of dim (batch_size, 1),
            Tensor of boolean values establishing which corresponding clips are for the same user
        )

        The output of this function is fed to get_eval_metrics(...)
        """
        return (
            batch[0].to(self.device),
            batch[1].to(self.device),
            batch[2].to(self.device),
            batch[3].to(self.device),
            batch[4].to(self.device),
        )

    def fl_create_test_batch(
        self,
        batch: Tuple[
            torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor
        ],
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        The input is assembled by a pytorch data loader already and corresponds to a batch. Specifically,
        the input is of form:
        (
            Tensor for clip As of dim (batch_size, 1, 100, 64),
            Tensor for clip Bs of dim (batch_size, 1, 100, 64),
            Tensor for gold labels for clip As of dim (batch_size, 1)
            Tensor for gold labels for clip Bs of dim (batch_size, 1),
            Tensor of boolean values establishing which corresponding clips are for the same user
        )
        """
        return (
            batch[0].to(self.device),
            batch[1].to(self.device),
            batch[2].to(self.device),
            batch[3].to(self.device),
            batch[4].to(self.device),
        )
