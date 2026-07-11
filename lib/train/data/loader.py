import torch
import torch.utils.data.dataloader
import importlib
import collections
import collections.abc as collections_abc
import math
import random
string_classes = (str, bytes)
from lib.utils import TensorDict, TensorList

if float(torch.__version__[:3]) >= 1.9 or len('.'.join((torch.__version__).split('.')[0:2])) > 3:
    int_classes = int
else:
    from torch._six import int_classes


def _check_use_shared_memory():
    if hasattr(torch.utils.data.dataloader, '_use_shared_memory'):
        return getattr(torch.utils.data.dataloader, '_use_shared_memory')
    collate_lib = importlib.import_module('torch.utils.data._utils.collate')
    if hasattr(collate_lib, '_use_shared_memory'):
        return getattr(collate_lib, '_use_shared_memory')
    return torch.utils.data.get_worker_info() is not None

def _new_shared_stack_output(batch, dim):
    numel = sum(x.numel() for x in batch)
    storage = batch[0]._typed_storage()._new_shared(numel, device=batch[0].device)
    out_shape = list(batch[0].size())
    out_shape.insert(dim, len(batch))
    return batch[0].new(storage).resize_(*out_shape)


class HorizonBatchSampler:
    """Group arbitrary sample indices into batches sharing one horizon."""

    def __init__(self, indices, batch_size, horizons, weights, drop_last):
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if not horizons or any(int(value) <= 0 for value in horizons):
            raise ValueError("horizons must contain positive values")
        if len(horizons) != len(weights):
            raise ValueError("horizons and weights must have the same length")
        if any(float(value) < 0 for value in weights) or sum(weights) <= 0:
            raise ValueError("weights must contain a positive total")
        self.indices = indices
        self.batch_size = int(batch_size)
        self.horizons = tuple(int(value) for value in horizons)
        self.weights = tuple(float(value) for value in weights)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.epoch)
        batch = []
        for index in self.indices:
            batch.append(index)
            if len(batch) == self.batch_size:
                yield self._with_horizon(batch, rng)
                batch = []
        if batch and not self.drop_last:
            yield self._with_horizon(batch, rng)

    def __len__(self):
        length = len(self.indices)
        if self.drop_last:
            return length // self.batch_size
        return math.ceil(length / self.batch_size)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        if hasattr(self.indices, "set_epoch"):
            self.indices.set_epoch(epoch)

    def _with_horizon(self, indices, rng):
        horizon = rng.choices(self.horizons, weights=self.weights, k=1)[0]
        return [(index, horizon) for index in indices]


_TEMPORAL_PAD_KEYS = {
    "history_images": ("history_valid", True),
    "history_event_images": ("history_valid", True),
    "history_anno": ("history_valid", True),
    "future_images": ("future_valid", False),
    "future_event_images": ("future_valid", False),
    "future_anno": ("future_valid", False),
}


def _pad_temporal_tensors(values, target_length, stack_dim, pad_left):
    padded = []
    for value in values:
        if value.shape[0] > target_length:
            raise ValueError("temporal tensor exceeds its declared length")
        output = value.new_zeros((target_length, *value.shape[1:]))
        offset = target_length - value.shape[0] if pad_left else 0
        output[offset:offset + value.shape[0]] = value
        padded.append(output)
    return torch.stack(padded, dim=stack_dim)


def _collate_tensor_dict(batch, collate, stack_dim):
    result = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        pad_spec = _TEMPORAL_PAD_KEYS.get(key)
        if pad_spec is not None:
            valid_key, pad_left = pad_spec
            target_length = max(
                int(item[valid_key].numel()) for item in batch)
            result[key] = _pad_temporal_tensors(
                values, target_length, stack_dim, pad_left)
        else:
            result[key] = collate(values)
    return TensorDict(result)



def ltr_collate(batch):
    """Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _check_use_shared_memory():
            out = _new_shared_stack_output(batch, 0)
        return torch.stack(batch, 0, out=out)
        # if batch[0].dim() < 4:
        #     return torch.stack(batch, 0, out=out)
        # return torch.cat(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if torch.utils.data.dataloader.re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return torch.utils.data.dataloader.numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], TensorDict):
        return _collate_tensor_dict(batch, ltr_collate, stack_dim=0)
    elif isinstance(batch[0], collections_abc.Mapping):
        return {key: ltr_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], TensorList):
        transposed = zip(*batch)
        return TensorList([ltr_collate(samples) for samples in transposed])
    elif isinstance(batch[0], collections_abc.Sequence):
        transposed = zip(*batch)
        return [ltr_collate(samples) for samples in transposed]
    elif batch[0] is None:
        return batch

    raise TypeError((error_msg.format(type(batch[0]))))


def ltr_collate_stack1(batch):
    """Puts each data field into a tensor. The tensors are stacked at dim=1 to form the batch"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        stack_dim = 0 if batch[0].ndim == 0 else 1
        out = None
        if _check_use_shared_memory():
            out = _new_shared_stack_output(batch, stack_dim)
        return torch.stack(batch, stack_dim, out=out)
        # if batch[0].dim() < 4:
        #     return torch.stack(batch, 0, out=out)
        # return torch.cat(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if torch.utils.data.dataloader.re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 1)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return torch.utils.data.dataloader.numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], TensorDict):
        return _collate_tensor_dict(batch, ltr_collate_stack1, stack_dim=1)
    elif isinstance(batch[0], collections_abc.Mapping):
        return {key: ltr_collate_stack1([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], TensorList):
        transposed = zip(*batch)
        return TensorList([ltr_collate_stack1(samples) for samples in transposed])
    elif isinstance(batch[0], collections_abc.Sequence):
        transposed = zip(*batch)
        return [ltr_collate_stack1(samples) for samples in transposed]
    elif batch[0] is None:
        return batch

    raise TypeError((error_msg.format(type(batch[0]))))


class LTRLoader(torch.utils.data.dataloader.DataLoader):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    Note: The only difference with default pytorch DataLoader is that an additional option stack_dim is available to
            select along which dimension the data should be stacked to form a batch.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, ``shuffle`` must be False.
        batch_sampler (Sampler, optional): like sampler, but returns a batch of
            indices at a time. Mutually exclusive with batch_size, shuffle,
            sampler, and drop_last.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            (default: 0)
        collate_fn (callable, optional): merges a list of samples to form a mini-batch.
        stack_dim (int): Dimension along which to stack to form the batch. (default: 0)
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors
            into CUDA pinned memory before returning them.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: 0)
        worker_init_fn (callable, optional): If not None, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: None)

    .. note:: By default, each worker will have its PyTorch seed set to
              ``base_seed + worker_id``, where ``base_seed`` is a long generated
              by main process using its RNG. However, seeds for other libraries
              may be duplicated upon initializing workers (w.g., NumPy), causing
              each worker to return identical random numbers. (See
              :ref:`dataloader-workers-random-seed` section in FAQ.) You may
              use ``torch.initial_seed()`` to access the PyTorch seed for each
              worker in :attr:`worker_init_fn`, and use it to set other seeds
              before data loading.

    .. warning:: If ``spawn`` start method is used, :attr:`worker_init_fn` cannot be an
                 unpicklable object, e.g., a lambda function.
    """

    __initialized = False

    def __init__(self, name, dataset, training=True, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, epoch_interval=1, collate_fn=None, stack_dim=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None, persistent_workers=False):

        # Select the collate function according to stack_dim.
        if collate_fn is None:
            if stack_dim == 0:
                collate_fn = ltr_collate
            elif stack_dim == 1:
                collate_fn = ltr_collate_stack1
            else:
                raise ValueError('Stack dim no supported. Must be 0 or 1.')

        super(LTRLoader, self).__init__(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle,
            sampler=sampler, batch_sampler=batch_sampler,
            num_workers=num_workers, collate_fn=collate_fn,
            pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
            worker_init_fn=worker_init_fn,
            persistent_workers=bool(persistent_workers and num_workers > 0),
        )

        self.name = name
        self.training = training
        self.epoch_interval = epoch_interval
        self.stack_dim = stack_dim
