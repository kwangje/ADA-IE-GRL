import os
import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.parameter_transfer import Pretrainer

from models.ADA_IE import ADA_IE
from datasets.ADAIE_dataset import get_dataset


if __name__ == "__main__":

    torch.cuda.empty_cache()

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    sb.utils.distributed.ddp_init_group(run_opts)
    
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    pretrain = Pretrainer(loadables={'model': hparams['modules']['fbanks_encoder']},
                          paths={'model': "speechbrain/spkrec-ecapa-voxceleb/embedding_model.ckpt"})
    # local path
    # pretrain = Pretrainer(collect_in='model_checkpoints',
    #             loadables={'model': hparams['modules']['fbanks_encoder']},
    #             paths={'model': 'model_checkpoints/model.ckpt'})

    pretrain.collect_files()
    pretrain.load_collected()
    
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    

    datasets = get_dataset(hparams)
        
    grl_model = ADA_IE(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    grl_model.fit(
        epoch_counter=grl_model.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["dev"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

