import random
import torch.utils.data
import torch
from lib.utils import TensorDict
from lib.train.data.srbt_labels import build_temporal_targets

def no_processing(data):
    return data


class TrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames, num_template_frames=1, processing=no_processing, frame_sample_mode='causal',
                 train_cls=False, pos_prob=0.5, cfg=None, training=True):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.datasets = datasets
        self.train_cls = train_cls
        self.pos_prob = pos_prob

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode
        self.cfg = cfg
        self.training = bool(training)
        data_cfg = getattr(cfg, "DATA", None)
        srbt_cfg = getattr(data_cfg, "SRBT", None)
        self.srbt_enabled = bool(getattr(srbt_cfg, "ENABLE", False))
        self.srbt_history_length = int(getattr(srbt_cfg, "HISTORY_LENGTH", 8))
        if self.srbt_history_length <= 0:
            raise ValueError("SRBT history length must be positive")
        self.srbt_max_hazard = int(getattr(srbt_cfg, "MAX_HAZARD", 128))
        anchor_weights = getattr(srbt_cfg, "ANCHOR_WEIGHTS", None)
        self.srbt_anchor_weights = {
            "visible_to_visible": float(getattr(anchor_weights, "VISIBLE", 0.25)),
            "visible_to_absent": float(getattr(anchor_weights, "PRESENT_TO_ABSENT", 0.25)),
            "absent_to_absent": float(getattr(anchor_weights, "ABSENT", 0.25)),
            "absent_to_present": float(getattr(anchor_weights, "REAPPEARING", 0.25)),
        }
        # self.batch_size = batch_size

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None,
                            allow_invisible=False, force_invisible=False):
        """ Samples num_ids frames between min_id and max_id for which target is visible
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        if force_invisible:
            valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        else:
            if allow_invisible:
                valid_ids = [i for i in range(min_id, max_id)]
            else:
                valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    @staticmethod
    def _to_bool_list(values):
        if hasattr(values, "detach"):
            values = values.detach().cpu().tolist()
        return [bool(item) for item in values]

    def _add_presence_transition_fields(self, data, seq_info_dict, template_frame_ids, search_frame_ids,
                                        template_anno=None, search_anno=None, sampler_event_type=None):
        data['template_frame_ids'] = torch.tensor(template_frame_ids, dtype=torch.long)
        data['search_frame_ids'] = torch.tensor(search_frame_ids, dtype=torch.long)
        data['sampler_event_type'] = sampler_event_type or "default"
        if template_anno is not None and 'absent' in template_anno:
            data['template_absent'] = template_anno['absent']
        if search_anno is not None and 'absent' in search_anno:
            data['search_absent'] = search_anno['absent']
        if 'absent' not in seq_info_dict:
            return data

        # FELT absent.txt is stored as a present flag in this codebase:
        # 1 means target present, 0 means target absent.
        present_flags = self._to_bool_list(seq_info_dict['absent'])
        previous_present = []
        current_present = []
        for frame_id in search_frame_ids:
            frame_id = int(frame_id)
            cur = present_flags[frame_id] if frame_id < len(present_flags) else True
            prev_id = max(frame_id - 1, 0)
            prev = present_flags[prev_id] if prev_id < len(present_flags) else cur
            previous_present.append(prev)
            current_present.append(cur)
        previous_present = torch.tensor(previous_present, dtype=torch.uint8)
        current_present = torch.tensor(current_present, dtype=torch.uint8)
        data['previous_present'] = previous_present
        data['current_present'] = current_present
        data['is_reappear'] = ((~previous_present.bool()) & current_present.bool()).to(torch.uint8)
        return data

    def _srbt_anchor_ids(self, seq_info_dict, event_type,
                         min_id=None, max_id=None):
        present = torch.as_tensor(seq_info_dict["absent"], dtype=torch.bool)
        if "valid" in seq_info_dict:
            present = present & torch.as_tensor(
                seq_info_dict["valid"], dtype=torch.bool)
        present = present.tolist()
        min_id = max(0, 0 if min_id is None else int(min_id))
        last_anchor = max(0, len(present) - 1)
        max_id = last_anchor if max_id is None else min(int(max_id), last_anchor)

        present_run = []
        run_length = 0
        for is_present in present:
            run_length = run_length + 1 if is_present else 0
            present_run.append(run_length)

        ids = []
        for frame_id in range(min_id, max_id):
            previous = present[frame_id - 1] if frame_id > 0 else present[frame_id]
            current = present[frame_id]
            following = present[frame_id + 1]
            run_length = present_run[frame_id]
            run_started_after_absence = current and run_length <= frame_id
            stable_visible = (
                current
                and run_length >= 3
                and not (run_started_after_absence and run_length <= 3)
            )
            if event_type == "visible_to_visible" and stable_visible and following:
                ids.append(frame_id)
            elif event_type == "visible_to_absent" and current and not following:
                ids.append(frame_id)
            elif event_type == "absent_to_absent" and not current:
                ids.append(frame_id)
            elif event_type == "absent_to_present" and not previous and current:
                ids.append(frame_id)
        return ids

    def _causal_frame_ids_for_anchor(self, visible, search_id, event_ids):
        base_id = self._previous_visible_id(visible, search_id)
        if base_id is None:
            return None
        prev_frame_ids = self._sample_visible_ids(
            visible,
            num_ids=self.num_template_frames - 1,
            min_id=base_id - self.max_gap,
            max_id=base_id,
        )
        if prev_frame_ids is None:
            return None
        search_frame_ids = [search_id]
        if self.num_search_frames > 1:
            later_ids = [idx for idx in event_ids if idx >= search_id]
            search_frame_ids += random.choices(
                later_ids or event_ids, k=self.num_search_frames - 1)
        return [base_id] + prev_frame_ids, search_frame_ids

    def _sample_srbt_event_causal_frame_ids(self, visible, seq_info_dict):
        if "absent" not in seq_info_dict:
            return None, None, None

        examples = {}
        for event_type in self.srbt_anchor_weights:
            event_ids = self._srbt_anchor_ids(
                seq_info_dict,
                event_type,
                min_id=self.num_template_frames,
            )
            random.shuffle(event_ids)
            for search_id in event_ids:
                frame_ids = self._causal_frame_ids_for_anchor(
                    visible, search_id, event_ids)
                if frame_ids is not None:
                    examples[event_type] = frame_ids
                    break

        if not examples:
            return None, None, None
        events = list(examples)
        weights = [max(0.0, self.srbt_anchor_weights[event]) for event in events]
        event_type = random.choices(
            events,
            weights=weights if sum(weights) > 0 else None,
            k=1,
        )[0]
        template_frame_ids, search_frame_ids = examples[event_type]
        return template_frame_ids, search_frame_ids, event_type

    def _previous_visible_id(self, visible, frame_id):
        visible = self._to_bool_list(visible)
        for idx in range(frame_id - 1, -1, -1):
            if visible[idx]:
                return idx
        return None

    def _replace_absent_search_boxes_for_crop(self, search_anno, search_frame_ids, seq_info_dict):
        if "absent" not in search_anno or "bbox" not in search_anno:
            return
        for item_id, frame_id in enumerate(search_frame_ids):
            absent_value = search_anno["absent"][item_id]
            if hasattr(absent_value, "item"):
                absent_value = absent_value.item()
            if int(absent_value) != 0:
                continue
            anchor_id = self._previous_visible_id(seq_info_dict["visible"], frame_id)
            if anchor_id is not None:
                search_anno["bbox"][item_id] = seq_info_dict["bbox"][anchor_id].clone()

    def _add_srbt_future_fields(self, data, dataset, seq_id, seq_info_dict,
                                anchor, horizon):
        present = seq_info_dict.get("absent", seq_info_dict["visible"]).to(torch.bool)
        if "valid" in seq_info_dict:
            present = present & seq_info_dict["valid"].to(torch.bool)

        targets = build_temporal_targets(
            present,
            anchor=anchor,
            horizon=horizon,
            max_hazard=self.srbt_max_hazard,
        )
        valid_length = int(targets["future_valid"].sum().item())
        future_frame_ids = list(range(anchor + 1, anchor + 1 + valid_length))

        if future_frame_ids:
            future_images, future_event_images, future_anno, _ = dataset.get_frames(
                seq_id, future_frame_ids, seq_info_dict)
            height, width = future_images[0].shape[:2]
            future_masks = future_anno.get(
                "mask", [torch.zeros((height, width))] * valid_length)
            future_boxes = future_anno["bbox"]
        else:
            future_images = []
            future_event_images = []
            future_boxes = []
            future_masks = []

        padded_ids = torch.full((int(horizon),), -1, dtype=torch.long)
        if future_frame_ids:
            padded_ids[:valid_length] = torch.tensor(future_frame_ids, dtype=torch.long)

        data.update({
            "future_images": future_images,
            "future_event_images": future_event_images,
            "future_anno": future_boxes,
            "future_masks": future_masks,
            "future_frame_ids": padded_ids,
            **targets,
        })

    def _add_srbt_history_fields(self, data, dataset, seq_id, seq_info_dict,
                                 anchor):
        start = max(0, anchor - self.srbt_history_length)
        history_frame_ids = list(range(start, anchor))
        if history_frame_ids:
            history_images, history_event_images, history_anno, _ = \
                dataset.get_frames(seq_id, history_frame_ids, seq_info_dict)
            height, width = history_images[0].shape[:2]
            history_boxes = history_anno["bbox"]
            history_masks = history_anno.get(
                "mask", [torch.zeros((height, width))] * len(history_frame_ids))
        else:
            history_images = []
            history_event_images = []
            history_boxes = []
            history_masks = []

        present = seq_info_dict.get("absent", seq_info_dict["visible"]).to(torch.bool)
        if "valid" in seq_info_dict:
            present = present & seq_info_dict["valid"].to(torch.bool)
        valid_length = len(history_frame_ids)
        offset = self.srbt_history_length - valid_length
        padded_ids = torch.full(
            (self.srbt_history_length,), -1, dtype=torch.long)
        history_valid = torch.zeros(
            self.srbt_history_length, dtype=torch.bool)
        history_present = torch.zeros(
            self.srbt_history_length, dtype=torch.long)
        padded_ids[offset:] = torch.tensor(history_frame_ids, dtype=torch.long)
        history_valid[offset:] = True
        history_present[offset:] = present[history_frame_ids].to(torch.long)

        data.update({
            "history_images": history_images,
            "history_event_images": history_event_images,
            "history_anno": history_boxes,
            "history_masks": history_masks,
            "history_frame_ids": padded_ids,
            "history_present": history_present,
            "history_valid": history_valid,
        })

    def __getitem__(self, index):
        horizon = None
        if isinstance(index, tuple):
            if len(index) != 2:
                raise ValueError("sample index tuples must contain (index, horizon)")
            _, horizon = index
        if self.train_cls:
            batch_data = self.getitem_cls()
        else:
            batch_data = self.getitem(horizon=horizon)
        return batch_data

    def getitem(self, horizon=None):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        while not valid:
            dataset = random.choices(self.datasets, self.p_datasets)[0]
            is_video_dataset = dataset.is_video_sequence()
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(
                dataset, is_video_dataset)
            if is_video_dataset:
                if self.frame_sample_mode == 'causal':
                    template_frame_ids, search_frame_ids = None, None
                    sampler_event_type = None
                    gap_increase = 0
                    # Sample test and train frames in a causal manner, i.e. search_frame_ids > template_frame_ids
                    if horizon is not None and self.srbt_enabled:
                        template_frame_ids, search_frame_ids, sampler_event_type = \
                            self._sample_srbt_event_causal_frame_ids(
                                visible, seq_info_dict)
                    while search_frame_ids is None:
                        base_frame_id = self._sample_visible_ids(visible, num_ids=1,
                                                                 min_id=self.num_template_frames - 1,
                                                                 max_id=len(visible) - self.num_search_frames)
                        prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                  min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                                  max_id=base_frame_id[0])
                        if prev_frame_ids is None:
                            gap_increase += 5
                            continue

                        template_frame_ids = base_frame_id + prev_frame_ids
                        search_frame_ids = self._sample_visible_ids(
                            visible,
                            num_ids=self.num_search_frames,
                            min_id=template_frame_ids[0] + 1,
                            max_id=template_frame_ids[0] + self.max_gap + gap_increase)
                        sampler_event_type = sampler_event_type or "default"
                        gap_increase += 5

                elif self.frame_sample_mode == "trident" or self.frame_sample_mode == "trident_pro":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                    sampler_event_type = "trident"
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                    sampler_event_type = "stark"
                else:
                    raise ValueError("Illegal frame sample mode")
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames
                sampler_event_type = "image"
            try:
                # rgb + event
                template_aps_frame_list, template_dvs_frame_list, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
                search_aps_frame_list, search_dvs_frame_list, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)
                self._replace_absent_search_boxes_for_crop(search_anno, search_frame_ids, seq_info_dict)

                H, W, _ = template_aps_frame_list[0].shape
                template_masks = template_anno['mask'] if 'mask' in template_anno else [torch.zeros((H, W))] * self.num_template_frames
                search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros((H, W))] * self.num_search_frames

                data = TensorDict({'template_images': template_aps_frame_list,
                                   'template_anno': template_anno['bbox'],
                                   'template_masks': template_masks,
                                   'search_images': search_aps_frame_list,
                                   'search_anno': search_anno['bbox'],
                                   'search_masks': search_masks,
                                   'dataset': dataset.get_name(),
                                   'test_class': meta_obj_test.get('object_class_name'),
                                    'template_event_images': template_dvs_frame_list,
                                   'search_event_images': search_dvs_frame_list,
                                })
                if self.srbt_enabled and is_video_dataset:
                    self._add_srbt_history_fields(
                        data,
                        dataset,
                        seq_id,
                        seq_info_dict,
                        anchor=search_frame_ids[-1],
                    )
                if horizon is not None:
                    self._add_srbt_future_fields(
                        data,
                        dataset,
                        seq_id,
                        seq_info_dict,
                        anchor=search_frame_ids[-1],
                        horizon=int(horizon),
                    )
                self._add_presence_transition_fields(
                    data, seq_info_dict, template_frame_ids, search_frame_ids,
                    template_anno, search_anno, sampler_event_type)
                # make data augmentation
                data = self.processing(data)

                # check whether data is vali
                valid = data['valid']
            except RuntimeError:
                print("Error loading sample.")
                valid = False

        return data

    def getitem_cls(self):
        """
        args:
            index (int): Index (Ignored since we sample randomly)
            aux (bool): whether the current data is for auxiliary use (e.g. copy-and-paste)

        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        label = None
        while not valid:
            # Select a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            # sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
            # sample template and search frame ids
            if is_video_dataset:
                if self.frame_sample_mode in ["trident", "trident_pro"]:
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("illegal frame sample mode")
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames
            try:
                # "try" is used to handle trackingnet data failure
                # get images and bounding boxes (for templates)
                template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids,
                                                                                    seq_info_dict)
                H, W, _ = template_frames[0].shape
                template_masks = template_anno['mask'] if 'mask' in template_anno else [torch.zeros(
                    (H, W))] * self.num_template_frames
                # get images and bounding boxes (for searches)
                # positive samples
                if random.random() < self.pos_prob:
                    label = torch.ones(1,)
                    search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)
                    search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros(
                        (H, W))] * self.num_search_frames
                # negative samples
                else:
                    label = torch.zeros(1,)
                    if is_video_dataset:
                        search_frame_ids = self._sample_visible_ids(visible, num_ids=1, force_invisible=True)
                        if search_frame_ids is None:
                            search_frames, search_anno, meta_obj_test = self.get_one_search()
                        else:
                            search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids,
                                                                                           seq_info_dict)
                            search_anno["bbox"] = [self.get_center_box(H, W)]
                    else:
                        search_frames, search_anno, meta_obj_test = self.get_one_search()
                    H, W, _ = search_frames[0].shape
                    search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros(
                        (H, W))] * self.num_search_frames

                data = TensorDict({'template_images': template_frames,
                                   'template_anno': template_anno['bbox'],
                                   'template_masks': template_masks,
                                   'search_images': search_frames,
                                   'search_anno': search_anno['bbox'],
                                   'search_masks': search_masks,
                                   'dataset': dataset.get_name(),
                                   'test_class': meta_obj_test.get('object_class_name')})
                # make data augmentation
                data = self.processing(data)
                # add classification label
                data["label"] = label
                # check whether data is valid
                valid = data['valid']
            except:
                valid = False

        return data

    def get_center_box(self, H, W, ratio=1/8):
        cx, cy, w, h = W/2, H/2, W * ratio, H * ratio
        return torch.tensor([int(cx-w/2), int(cy-h/2), int(w), int(h)])

    def sample_seq_from_dataset(self, dataset, is_video_dataset):

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_search_frames + self.num_template_frames) and len(visible) >= 20

            enough_visible_frames = enough_visible_frames or not is_video_dataset
        return seq_id, visible, seq_info_dict

    def get_one_search(self):
        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]

        is_video_dataset = dataset.is_video_sequence()
        # sample a sequence
        seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
        # sample a frame
        if is_video_dataset:
            if self.frame_sample_mode == "stark":
                search_frame_ids = self._sample_visible_ids(seq_info_dict["valid"], num_ids=1)
            else:
                search_frame_ids = self._sample_visible_ids(visible, num_ids=1, allow_invisible=True)
        else:
            search_frame_ids = [1]
        # get the image, bounding box and other info
        search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

        return search_frames, search_anno, meta_obj_test

    def get_frame_ids_trident(self, visible):
        # get template and search ids in a 'trident' manner
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                if self.frame_sample_mode == "trident_pro":
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id,
                                                    allow_invisible=True)
                else:
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids

    def get_frame_ids_stark(self, visible, valid):
        # get template and search ids in a 'stark' manner
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # get the dynamic template id

            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                """we require the frame to be valid but not necessary visible"""
                f_id = self._sample_visible_ids(valid, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids
