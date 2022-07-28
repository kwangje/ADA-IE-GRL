import speechbrain as sb
import os

SPEAKER_ID_LIST = 'speaker_id_list.json'

def get_spk_eval_dataset():
    @sb.utils.data_pipeline.takes("file_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(file_path):
        sig = sb.dataio.dataio.read_audio(file_path)
        return sig

    data_dir = '/media/beck/ssd/datasets_opensource/sr/processed_data'
    enrol_file = 'eval_enroll.json'
    trl_file = 'eval_trl.json'

    datasets = {}

    datasets['enrol'] = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path= os.path.join(data_dir, enrol_file),
        dynamic_items=[audio_pipeline,
                       ],
        output_keys=["id", "sig", "speaker_id"],
    )

    datasets['trl'] = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path= os.path.join(data_dir, trl_file),
        dynamic_items=[audio_pipeline,
                        ],
        output_keys=["id", "sig", "speaker_id"],
    )
    return datasets