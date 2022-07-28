
from speechbrain.nnet.losses import bce_loss, nll_loss
import torch

def ADAIE_loss(
        spk_loss_metric,
        log_probabilities_spk,
        log_probabilities_ag,
        log_probabilities_c,
        spk_targets,
        ag_targets,
        c_targets,
        length=None,
        label_smoothing=0.0,
        allowed_len_diff=3,
        reduction="mean",
):
    # process ag loss
    ag_loss = bce_loss(log_probabilities_ag, ag_targets, length=length,reduction=reduction)

    if reduction == "batch":
        return ag_loss

    def get_masked_nll_loss(mask,
                            targets,
                            log_probabilities,
                            length,
                            loss_func
                            ):
        mask = torch.squeeze(mask, 1)
        masked_log_probabilities = log_probabilities[mask]
        masked_targets = targets[mask]
        masked_length = None

        # if length is not None: masked_length = length[torch.squeeze(mask, 1)]
        if length is not None: masked_length = length[mask]

        # if the target exists in current batch
        if masked_targets.shape[0] == 0:
            loss = torch.zeros(ag_loss.shape, device='cuda:0')
            return loss
        loss = loss_func(masked_log_probabilities,
                         masked_targets,
                         length=masked_length,
                         # reduction=reduction
                         )

        if reduction == 'batch':
            temp = torch.zeros(ag_loss.shape, device='cuda:0')
            for i, (index, value) in enumerate(zip(mask, loss)):
                temp[index] =value
            loss = temp

        return loss

    # process c loss
    c_mask = torch.le(c_targets, 3)
    c_loss = get_masked_nll_loss(c_mask,
                                   c_targets,
                                   log_probabilities_c,
                                   length,
                                   nll_loss,
                                   )

    # process spk loss
    ag_mask = torch.ge(ag_targets, 1)
    spk_loss = get_masked_nll_loss(ag_mask, spk_targets, log_probabilities_spk, length, spk_loss_metric)

    dgasv_loss = ag_loss + spk_loss + 0.01*c_loss

    return dgasv_loss

