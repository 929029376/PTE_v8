import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os

# TODO: class-FeltDataset
class FELTDataset(BaseDataset):
    def __init__(self, split, base_path=None):
        super().__init__()
        self.split = split
        if base_path is not None:
            self.base_path = os.fspath(base_path)
        elif split == 'test':
            self.base_path = os.path.join(self.env_settings.felt_path,  split)
        else:
            self.base_path = os.path.join(self.env_settings.felt_path, 'train')
        self.sequence_list = self._get_sequence_list(split)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        with open('{}/list1k.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()
        if split == 'val' or split == 'train':
            split_path = os.path.join(self.base_path, '{}1k.txt'.format(split))
            with open(split_path) as f:
                seq_ids = f.read().splitlines()
            sequence_list = [sequence_list[int(x)] for x in seq_ids]
        sequence_list = self._filter_runnable_sequences(sequence_list)
        return sequence_list

    def _filter_runnable_sequences(self, sequence_list):
        runnable = []
        skipped = []
        for sequence_name in sequence_list:
            if self._has_sequence_files(self._resolve_sequence_path(sequence_name), sequence_name):
                runnable.append(sequence_name)
            else:
                skipped.append(sequence_name)
        if skipped:
            if self.split == 'val':
                raise FileNotFoundError(
                    'FELT val split has incomplete sequences: {}'.format(
                        ', '.join(skipped)))
            print('FELT: skipped {} incomplete sequences from {}'.format(len(skipped), self.base_path))
        return runnable

    def _sequence_leaf(self, sequence_name):
        return os.path.basename(sequence_name.rstrip(os.sep))

    def _resolve_sequence_path(self, sequence_name):
        seq_path = os.path.join(self.base_path, sequence_name)
        nested_path = os.path.join(seq_path, self._sequence_leaf(sequence_name))
        if self._has_sequence_files(seq_path, sequence_name):
            return seq_path
        if self._has_sequence_files(nested_path, sequence_name):
            return nested_path
        return seq_path

    def _has_sequence_files(self, seq_path, sequence_name):
        sequence_leaf = self._sequence_leaf(sequence_name)
        return (os.path.isfile(os.path.join(seq_path, 'groundtruth.txt')) and
                os.path.isfile(os.path.join(seq_path, 'absent.txt')) and
                os.path.isdir(os.path.join(seq_path, sequence_leaf + '_aps')) and
                os.path.isdir(os.path.join(seq_path, sequence_leaf + '_dvs')))

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        seq_path = self._resolve_sequence_path(sequence_name)
        anno_path = '{}/groundtruth.txt'.format(seq_path)
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64).reshape(-1, 4)
        presence = np.loadtxt(
            os.path.join(seq_path, 'absent.txt'), dtype=np.uint8).reshape(-1)
        if presence.shape[0] != ground_truth_rect.shape[0]:
            raise ValueError(
                '{} has {} boxes but {} presence flags'.format(
                    sequence_name, ground_truth_rect.shape[0], presence.shape[0]))
        if not np.isin(presence, (0, 1)).all():
            raise ValueError(
                '{} presence flags must use 1=present and 0=absent'.format(
                    sequence_name))

        aps_seq_path = '{}/{}'.format(seq_path, self._sequence_leaf(sequence_name)+'_aps')
        aps_frame_list = [frame for frame in os.listdir(aps_seq_path) if frame.endswith(".png") or frame.endswith(".bmp") ]
        aps_frame_list.sort(key=lambda f: int(f[-8:-4]))
        aps_frame_list = [os.path.join(aps_seq_path, frame) for frame in aps_frame_list]

        dvs_seq_path = '{}/{}'.format(seq_path, self._sequence_leaf(sequence_name)+'_dvs')
        dvs_frame_list = [frame for frame in os.listdir(dvs_seq_path) if frame.endswith(".png") or frame.endswith(".bmp") ]
        dvs_frame_list.sort(key=lambda f: int(f[-8:-4]))
        dvs_frame_list = [os.path.join(dvs_seq_path, frame) for frame in dvs_frame_list]

        frame_counts = {
            'APS': len(aps_frame_list),
            'DVS': len(dvs_frame_list),
            'ground_truth': ground_truth_rect.shape[0],
            'presence': presence.shape[0],
        }
        if len(set(frame_counts.values())) != 1:
            raise ValueError(
                '{} frame count mismatch: {}'.format(
                    sequence_name,
                    ', '.join('{}={}'.format(name, count)
                              for name, count in frame_counts.items()),
                ))

        return Sequence(
            sequence_name,
            aps_frame_list,
            'FELT',
            ground_truth_rect,
            target_visible=presence,
            dvs_frame_list=dvs_frame_list,
        )
