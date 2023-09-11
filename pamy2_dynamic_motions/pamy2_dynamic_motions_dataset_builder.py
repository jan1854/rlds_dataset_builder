from pathlib import Path
from typing import Iterator, Tuple, Any

import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class Pamy2DynamicMotions(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        # 'image': tfds.features.Image(
                        #     shape=(0, 0, 0),
                        #     dtype=np.uint8,
                        #     encoding_format='png',
                        #     doc='Main camera RGB observation.',
                        # ),
                        'state': tfds.features.Tensor(
                            shape=(16,),
                            dtype=np.float32,
                            doc='Robot state, consists of 4x robot joint angles [rad], 4x robot joint velocities [rad/s], '
                                '8x observed muscle pressures (agonist 0, antagonist 0, agonist 1, ...).',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(8,),
                        dtype=np.float32,
                        doc='Robot action, consists of 8x desired muscle pressures (agonist 0, antagonist 0, agonist 1, ...).',
                    ),
                    # 'discount': tfds.features.Scalar(
                    #     dtype=np.float32,
                    #     doc='There is no discount factor in the data, set to 1.'
                    # ),
                    # 'reward': tfds.features.Scalar(
                    #     dtype=np.float32,
                    #     doc='There is no reward in the data, set to 0.'
                    # ),
                    # 'is_first': tfds.features.Scalar(
                    #     dtype=np.bool_,
                    #     doc='The data is not divided into episodes, set to False.'
                    # ),
                    # 'is_last': tfds.features.Scalar(
                    #     dtype=np.bool_,
                    #     doc='The data is not divided into episodes, set to False.'
                    # ),
                    # 'is_terminal': tfds.features.Scalar(
                    #     dtype=np.bool_,
                    #     doc='The data is not divided into episodes, set to False.'
                    # ),
                    # 'language_instruction': tfds.features.Text(
                    #     doc='There are no language instructions in the data, set to a dummy value.'
                    # ),
                    # 'language_embedding': tfds.features.Tensor(
                    #     shape=(512,),
                    #     dtype=np.float32,
                    #     doc='There are no language instructions in the data, set to a dummy value.'
                    # ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(Path(__file__).parent / "data" / "*"),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            data = pd.read_pickle(episode_path)

            state_columns = ([f"position_{i}" for i in range(4)]
                             + [f"velocity_{i}" for i in range(4)]
                             + [f"observed_pressure_{i}_{muscle}" for i in range(4) for muscle in ["ago", "antago"]])
            action_columns = [f"desired_pressure_{i}_{muscle}" for i in range(4) for muscle in ["ago", "antago"]]

            data_state_action_only = data[state_columns + action_columns].values

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for step in data_state_action_only:
                # dummy_language_instruction = 'move dynamically'
                # compute Kona language embedding
                # language_embedding = self._embed([dummy_language_instruction])[0].numpy()
                state = step[:len(state_columns)].astype(np.float32)
                action = step[len(state_columns):].astype(np.float32)

                episode.append({
                    'observation': {
                        # 'image': np.zeros((0, 0, 0), dtype=np.uint8),
                        'state': state,
                    },
                    'action': action,
                    # 'discount': 1.0,
                    # 'reward': 0.0,
                    # 'is_first': False,
                    # 'is_last': False,
                    # 'is_terminal': False,
                    # 'language_instruction': dummy_language_instruction,
                    # 'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path.name
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return str(episode_path), sample

        # create list of all examples
        episode_paths = glob.glob(str(path))
        # Sort by the number of the data file (there are no leading zeros, so extract the number first and sort by that)
        episode_paths.sort(key=lambda x: int(Path(x).name[10:-3]))

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(Path(sample))

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

