import speechbrain as sb
import os
import json

LABEL_DIR = '/media/beck/ssd/datasets_opensource/sr/processed_data'
SPEAKER_ID_LIST = 'speaker_id_list_train_dc.json'
AG_LIST = 'AG_list.json'
C_ID_LIST = 'c_id_list.json'


def get_dataset(hparams):
    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'spk01': 0, 'spk02': 1, ..)
    label_encoder_spk = sb.dataio.encoder.CategoricalEncoder()
    label_encoder_ag = sb.dataio.encoder.CategoricalEncoder()
    label_encoder_c = sb.dataio.encoder.CategoricalEncoder()


    @sb.utils.data_pipeline.takes("path") # file_path
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(path):
        sig = sb.dataio.dataio.read_audio(path)
        return sig

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("VoxCeleb_ID") # speaker_id
    @sb.utils.data_pipeline.provides("speaker_id", "speaker_encoded")
    def speaker_label_pipeline(speaker_id):
        yield speaker_id
        speaker_encoded = label_encoder_spk.encode_label_torch(speaker_id, True)
        yield speaker_encoded

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("agegroup")
    @sb.utils.data_pipeline.provides("agegroup", "agegroup_encoded")
    def agegroup_label_pipeline(agegroup):
        yield agegroup
        agegroup_encoded = label_encoder_ag.encode_label_torch(agegroup, True)
        yield agegroup_encoded

    @sb.utils.data_pipeline.takes("child_id")
    @sb.utils.data_pipeline.provides("child_id", "c_encoded")
    def child_id_label_pipeline(child_id):
        yield child_id
        c_encoded = label_encoder_c.encode_label_torch(child_id, True)
        yield c_encoded


    datasets = {}

    for dataset in ["train", "dev", "eval"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[
                           audio_pipeline,
                           speaker_label_pipeline,
                           agegroup_label_pipeline,
                           child_id_label_pipeline,
                        #    triplet_label_pipeline
                           ],
            output_keys=["id", "sig",
                        "speaker_id", "speaker_encoded",
                        "agegroup", "agegroup_encoded",
                        "child_id", "c_encoded", 
                        # "triplet_encoded"
                         ],
        )

    def load_create_label_encoder(label_enc_file,
                                  label_encoder,
                                  label_list_file = None,
                                  ):
        label_list = None
        
        with open(os.path.join(LABEL_DIR, label_list_file), 'r') as f:
            label_list = [tuple(json.load(f))]

        lab_enc_file = os.path.join(hparams["save_folder"], label_enc_file)
        label_encoder.load_or_create(
            path=lab_enc_file,
            sequence_input=False,
            from_iterables=label_list,
        )

    load_create_label_encoder(label_enc_file="label_encoder_speaker.txt",
                              label_encoder=label_encoder_spk,
                              label_list_file=SPEAKER_ID_LIST
                              )
    label_encoder_spk.add_unk()

    load_create_label_encoder(label_enc_file="label_encoder_ag.txt",
                              label_encoder=label_encoder_ag,
                              label_list_file=AG_LIST
                              )
    label_encoder_ag.add_unk()

    load_create_label_encoder(label_enc_file="label_encoder_c.txt",
                              label_encoder=label_encoder_c,
                              label_list_file=C_ID_LIST,
                              )
    label_encoder_c.add_unk()  

    # load_create_label_encoder(label_enc_file="label_encoder_triplet.txt",
    #                           label_encoder=label_encoder_triplet,
    #                           label_list_file=TRIPLET_ID_LIST,
    #                           )

    return datasets