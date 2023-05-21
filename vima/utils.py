import collections
import numpy as np
import copy
import tree
import collections.abc
import torch
import pprint
import fnmatch
from contextlib import contextmanager
from typing import List, Union, Optional, Dict, Callable
import functools

__all__ = [
    "any_concat",
    "any_stack",
    "any_to_torch_tensor",
    "any_to_numpy",
    "any_to_datadict",
    "stack_sequence_fields",
    "get_batch_size",
    "any_slice",
    "any_transpose_first_two_axes",
    "add_batch_dim",
]


def any_concat(xs: list, *, dim: int = 0):
    """
    Works for both torch Tensor and numpy array
    """

    def _any_concat_helper(*xs):
        x = xs[0]
        if isinstance(x, np.ndarray):
            return np.concatenate(xs, axis=dim)
        elif torch.is_tensor(x):
            return torch.cat(xs, dim=dim)
        elif isinstance(x, float):
            # special treatment for float, defaults to float32
            return np.array(xs, dtype=np.float32)
        else:
            return np.array(xs)

    return tree.map_structure(_any_concat_helper, *xs)


def any_stack(xs: list, *, dim: int = 0):
    """
    Works for both torch Tensor and numpy array
    """

    def _any_stack_helper(*xs):
        x = xs[0]
        if isinstance(x, np.ndarray):
            return np.stack(xs, axis=dim)
        elif torch.is_tensor(x):
            return torch.stack(xs, dim=dim)
        elif isinstance(x, float):
            # special treatment for float, defaults to float32
            return np.array(xs, dtype=np.float32)
        else:
            return np.array(xs)

    return tree.map_structure(_any_stack_helper, *xs)


def any_to_torch_tensor(
    x,
    dtype: Union[str, torch.dtype, None] = None,
    device: Union[str, int, torch.device, None] = None,
    copy=False,
    non_blocking=False,
    smart_optimize: bool = True,
):
    dtype = torch_dtype(dtype)
    device = torch_device(device)

    if not isinstance(x, (torch.Tensor, np.ndarray)):
        # x is a primitive python sequence
        x = torch.tensor(x, dtype=dtype)
        copy = False

    # This step does not create any copy.
    # If x is a numpy array, simply wraps it in Tensor. If it's already a Tensor, do nothing.
    x = torch.as_tensor(x)
    # avoid passing None to .to(), PyTorch 1.4 bug
    dtype = dtype or x.dtype
    device = device or x.device

    if not smart_optimize:
        # do a single stage type conversion and transfer
        return x.to(dtype=dtype, device=device, copy=copy, non_blocking=non_blocking)

    # we have two choices: (1) convert dtype and then transfer to GPU
    # (2) transfer to GPU and then convert dtype
    # because CPU-to-GPU memory transfer is the bottleneck, we will reduce it as
    # much as possible by sending the smaller dtype

    src_dtype_size = torch_dtype_size(x.dtype)

    # destination dtype size
    if dtype is None:
        dest_dtype_size = src_dtype_size
    else:
        dest_dtype_size = torch_dtype_size(dtype)

    if x.dtype != dtype or x.device != device:
        # a copy will always be performed, no need to force copy again
        copy = False

    if src_dtype_size > dest_dtype_size:
        # better to do conversion on one device (e.g. CPU) and then transfer to another
        return _convert_then_transfer(x, dtype, device, copy, non_blocking)
    elif src_dtype_size == dest_dtype_size:
        # when equal, we prefer to do the conversion on whichever device that's GPU
        if x.device.type == "cuda":
            return _convert_then_transfer(x, dtype, device, copy, non_blocking)
        else:
            return _transfer_then_convert(x, dtype, device, copy, non_blocking)
    else:
        # better to transfer data across device first, and then do conversion
        return _transfer_then_convert(x, dtype, device, copy, non_blocking)


def any_to_numpy(
    x,
    dtype: Union[str, np.dtype, None] = None,
    copy: bool = False,
    non_blocking: bool = False,
    smart_optimize: bool = True,
):
    if isinstance(x, torch.Tensor):
        x = any_to_torch_tensor(
            x,
            dtype=dtype,
            device="cpu",
            copy=copy,
            non_blocking=non_blocking,
            smart_optimize=smart_optimize,
        )
        return x.detach().numpy()
    else:
        # primitive python sequence or ndarray
        return np.array(x, dtype=dtype, copy=copy)


def _convert_then_transfer(x, dtype, device, copy, non_blocking):
    x = x.to(dtype=dtype, copy=copy, non_blocking=non_blocking)
    return x.to(device=device, copy=False, non_blocking=non_blocking)


def _transfer_then_convert(x, dtype, device, copy, non_blocking):
    x = x.to(device=device, copy=copy, non_blocking=non_blocking)
    return x.to(dtype=dtype, copy=False, non_blocking=non_blocking)


def torch_dtype(dtype: Union[str, torch.dtype, None]) -> torch.dtype:
    if dtype is None:
        return None
    elif isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        try:
            dtype = getattr(torch, dtype)
        except AttributeError:
            raise ValueError(f'"{dtype}" is not a valid torch dtype')
        assert isinstance(
            dtype, torch.dtype
        ), f"dtype {dtype} is not a valid torch tensor type"
        return dtype
    else:
        raise NotImplementedError(f"{dtype} not supported")


def torch_device(device: Union[str, int, None]) -> torch.device:
    if device is None:
        return None
    elif device == "auto":
        return torch.device("cuda")
    elif isinstance(device, int) and device < 0:
        return torch.device("cpu")
    else:
        return torch.device(device)


def torch_dtype_size(dtype: Union[str, torch.dtype]) -> int:
    return _TORCH_DTYPE_TABLE[torch_dtype(dtype)]


_TORCH_DTYPE_TABLE = {
    torch.bool: 1,
    torch.int8: 1,
    torch.uint8: 1,
    torch.int16: 2,
    torch.short: 2,
    torch.int32: 4,
    torch.int: 4,
    torch.int64: 8,
    torch.long: 8,
    torch.float16: 2,
    torch.half: 2,
    torch.float32: 4,
    torch.float: 4,
    torch.float64: 8,
    torch.double: 8,
}


_DATA_VAR = "_data_"


class StopTraverse:
    """
    Prevent DataDict from recursing into the primitive dict structure.
    When you access the value, it will auto-unwrap into the original primitive
    """

    def __init__(self, value):
        if isinstance(value, StopTraverse):
            # do not double-wrap StopTraverse
            value = value.value
        self.value = value

    def __repr__(self):
        return f"{self.__class__.__name__} {self.value}"


class DataDict(collections.abc.MutableMapping):
    _AUTO_UNWRAP_STOP_TRAVERSE = True  # hack
    # should be used with contextmanager DataDict.settings_context()
    _INTERNAL_SETTINGS__ = {
        # strict nested structure matching for slice assign, merge, etc.
        "strict_match": False,
    }

    def __init__(self, _data_: collections.abc.Mapping = None, **kwargs):
        """
        Constructor:
            * DataDict(mapping)
            * DataDict(iterable[(key, value)])
            * DataDict(**kwargs)
        """
        if _data_ is not None:
            is_iter = _is_iterable(_data_)
            assert _is_mapping(_data_) or is_iter, (
                f"data type {type(_data_)} is not supported. "
                f"Only Mapping and Iterable[(key, value)] are valid."
            )
            assert not kwargs, (
                "DataDict can only be constructed by either a single "
                "positional arg (of type Mapping) or **kwargs, but not both"
            )
            if is_iter:
                _data_ = dict(_data_)
        else:
            _data_ = kwargs

        # self._data_ should always be a top-level PLAIN dict
        if isinstance(_data_, DataDict):
            self._data_ = _data_._data_.copy()
        else:
            self._data_ = {}
            # hack to preserve StopTraverse during constructor
            DataDict._AUTO_UNWRAP_STOP_TRAVERSE = False
            for k, v in _data_.items():
                self[k] = v
            DataDict._AUTO_UNWRAP_STOP_TRAVERSE = True

    def __getattr__(self, name):
        if name not in self._data_:
            raise AttributeError(f'Missing key-attribute "{name}"')
        return self[name]

    def __setattr__(self, name, value):
        if name == _DATA_VAR:
            super().__setattr__(name, value)
        else:
            self[name] = value

    def __delattr__(self, name):
        if name == _DATA_VAR:
            raise RuntimeError(f'Cannot delete "_data_" attr from DataDict')
        del self[name]

    def __getitem__(self, key):
        if isinstance(key, str):
            if "." in key:
                parent_key, child_key = key.split(".", 1)
                # recurse through every `.`
                if parent_key not in self._data_:
                    raise KeyError(f'Missing parent key "{parent_key}" in "{key}"')
                subdict = self._data_[parent_key]
                if not hasattr(subdict, "__getitem__"):
                    raise KeyError(
                        f"Parent is not a subscriptable object, "
                        f'cannot access child key "{child_key}"'
                    )
                return subdict[child_key]
            else:  # base case, key does not have "."
                if key not in self._data_:
                    raise KeyError(f'Missing key "{key}"')
                obj = self._data_[key]
                if (
                    isinstance(obj, StopTraverse)
                    and DataDict._AUTO_UNWRAP_STOP_TRAVERSE
                ):
                    return obj.value
                else:
                    return obj
        else:  # array slicing
            return tree.map_structure(lambda x: x[key] if _is_sliceable(x) else x, self)

    def __setitem__(self, key, value) -> None:
        if isinstance(key, str):
            if "." in key:
                parent_key, child_key = key.split(".", 1)
                if parent_key not in self._data_:
                    self._data_[parent_key] = DataDict()
                self._data_[parent_key][child_key] = value
            else:  # base case, key does not have "."
                self._data_[key] = _wrap_datadict(value)
        else:  # array slice assign
            value = _wrap_datadict(value)
            strict_match = DataDict._INTERNAL_SETTINGS__["strict_match"]

            def _assign_slice(paths, ours):
                v = value
                try:
                    # if a single value, we assign it to all
                    if _is_mapping(value):
                        for path in paths:
                            v = v[path]
                except Exception as e:
                    if strict_match:
                        raise TypeError(f"Sub-structure mismatch, key path: {paths}")
                    else:
                        return
                ours[key] = v

            tree.map_structure_with_path(_assign_slice, self)

    def __delitem__(self, key) -> None:
        if isinstance(key, str):
            if "." in key:
                parent_key, child_key = key.split(".", 1)
                if parent_key not in self._data_:
                    raise KeyError(f'Missing parent key "{parent_key}" in "{key}"')
                del self._data_[parent_key][child_key]
            else:  # base case, key does not have "."
                del self._data_[key]
        else:
            raise NotImplementedError("__delitem__ cannot handle non-string keys.")

    def __contains__(self, key: str):
        if "." in key:
            parent_key, child_key = key.split(".", 1)
            if parent_key in self._data_:
                try:
                    return child_key in self._data_[parent_key]
                except TypeError:
                    return False
            else:
                return False
        else:
            return key in self._data_

    def __len__(self) -> int:
        return len(self._data_)

    def __iter__(self):
        return iter(self._data_)

    def __getstate__(self):
        return self._data_

    def __setstate__(self, state):
        self._data_ = state

    def keys(self):
        return self._data_.keys()

    def values(self):
        return self._data_.values()

    def items(self):
        return self._data_.items()

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def pop(self, key, default=None):
        try:
            v = self[key]
            del self[key]
        except KeyError:
            v = default
        return v

    def update(self, other=None, **kwargs):
        """
        Same behavior as dict.update. Does not recurse into nested dicts.
        If you need recursive update, use merge() instead
        """
        other_dict = DataDict(other, **kwargs)
        self._data_.update(other_dict._data_)
        return self

    def merge(self, other=None, **kwargs):
        """
        Recursively update dict.
        """
        other_dict = DataDict(other, **kwargs)
        strict_match = DataDict._INTERNAL_SETTINGS__["strict_match"]
        our_keys = set(self.keys())
        for k, v in other_dict.items():
            if k in our_keys:
                our_v = self._data_[k]
                if _is_mapping(v) and _is_mapping(our_v):
                    self._data_[k].merge(v)  # recurse
                else:
                    if strict_match:
                        raise TypeError(f"Sub-structure mismatch: key {k}")
                    self._data_[k] = v
            else:
                if strict_match:
                    raise TypeError(f"Sub-structure mismatch: key {k}")
                self._data_[k] = v

    def __ior__(self, other):
        """
        |= operator
        https://docs.python.org/3/library/operator.html
        Python 3.9: dict union
        """
        return self.update(other)

    def __or__(self, other):
        return self.copy().update(other)

    def _filter_key_dict(self, keys):
        return DataDict({k: self._data_[k] for k in keys})

    def __and__(self, other):
        """
        & operator
        Get the intersection of keys between this dict and other.keys()
        """
        return self._filter_key_dict(set(self.keys()) & _get_keys_set(other))

    def __sub__(self, other):
        """
        - operator
        Get the subtraction set of keys between this dict and other.keys()
        """
        return self._filter_key_dict(set(self.keys()) - _get_keys_set(other))

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        INDENT = 2
        s = cls_name + " {\n"
        flag = False
        for k, v in self.items():
            rpl = "\n" + " " * INDENT
            obj = pprint.pformat(v).replace("\n", rpl)
            s += " " * INDENT + f"{k}: {obj},\n"
            flag = True
        if flag:
            s += "}"
        else:
            s = cls_name + "{}"
        return s

    def __copy__(self):
        return DataDict(self)

    def __deepcopy__(self, memo):
        return DataDict(copy.deepcopy(self._data_))

    def copy(self):
        """
        Copies all nested levels except the numpy/torch tensors.
        """
        return copy.copy(self)

    def deepcopy(self):
        """
        Deepcopies everything, including the leaf tensors.
        """
        return copy.deepcopy(self)

    # ------------------ Mapping functions -------------------
    def map_structure(
        self,
        func: Callable,
        *other_datadicts,
        inplace: bool = False,
        with_path: bool = False,
    ):
        mapper = tree.map_structure_with_path if with_path else tree.map_structure
        results = mapper(func, self, *other_datadicts)
        if inplace:
            self._data_ = results._data_
            return self
        else:
            return results

    def map_structure_with_path(
        self,
        func: Callable,
        *other_datadicts,
        inplace: bool = False,
    ):
        """
        Args:
            func: lambda path, v: <your func>
        """
        return self.map_structure(
            func, *other_datadicts, inplace=inplace, with_path=True
        )

    def traverse(self, func, top_down: bool = True):
        return tree.traverse(func, self, top_down=top_down)

    def to_container(self, flatten_keys: bool = False) -> Dict:
        """
        Args:
            flatten_keys: True to return a one-level dict with nested keys flattened
                by joining ".": {"obs.rgb": ..., "obs.state.proprio": ... }
                False to return the nested dict
        """
        if flatten_keys:
            ret = {}

            def _fn(paths, x):
                if not isinstance(x, DataDict):
                    ret[".".join(map(str, paths))] = x

            self.map_structure(_fn, with_path=True)
            return ret
        else:
            return tree.traverse(
                lambda x: x.to_container() if isinstance(x, DataDict) else None,
                self._data_,
                top_down=False,
            )

    @staticmethod
    @contextmanager
    def settings_context(strict_match=False):
        original_settings = DataDict._INTERNAL_SETTINGS__.copy()
        DataDict._INTERNAL_SETTINGS__["strict_match"] = strict_match
        try:
            yield
        except:
            raise
        finally:
            DataDict._INTERNAL_SETTINGS__.update(original_settings)

    def to_numpy(
        self,
        dtypes: Union[Dict[str, Union[str, type]], str, type, None] = None,
        copy: bool = False,
        non_blocking: bool = False,
        inplace: bool = True,
        include_keys: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        """
        dtypes: one of None, np.dtype, or {key_name: np.dtype}
            1. None: use default dtype inferred from the data
            2. np.dtype: use this dtype for all values
            3. a dict that maps a key to desired dtype
               nested key should be specified with dots, e.g. `obs.rgb`
               special key `None` to be a "catch-all" key
               you can also use special value `None` to automatically infer dtype
               for a given key
        """

        dtypes = _preprocess_dtypes(dtypes, "numpy")

        def _convert_fn(paths, value):
            key = ".".join(map(str, paths))
            if not match_patterns(
                key, include_keys, exclude_keys, precedence="exclude"
            ):
                # key is not matched, we don't convert and return value as-is
                return value
            return any_to_numpy(
                value,
                dtype=_match_pattern(key, dtypes),
                copy=copy,
                non_blocking=non_blocking,
                smart_optimize=True,
            )

        return self.map_structure(_convert_fn, inplace=inplace, with_path=True)

    def to_torch_tensor(
        self,
        dtypes: Union[
            Dict[str, Union[str, torch.dtype]], str, torch.dtype, None
        ] = None,
        copy: bool = False,
        device: Union[torch.device, int, str, None] = None,
        non_blocking: bool = False,
        inplace: bool = True,
        include_keys: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        """
        dtypes: one of None, np.dtype, or {key_name: np.dtype}
            1. None: use default dtype inferred from the data
            2. np.dtype: use this dtype for all values
            3. a dict that maps a key to desired dtype
               nested key should be specified with dots, e.g. `obs.rgb`
               special key `None` to be a "catch-all" key
               you can also use special value `None` to automatically infer dtype
               for a given key

        device: one of None, str, or torch.device
            - "auto": use the current context device
        non_blocking: transfer to device asynchronously, see torch.Tensor.to()
        copy: force copy even when tensor dtype and device don't change, see torch.Tensor.to()
        """
        dtypes = _preprocess_dtypes(dtypes, "torch")

        def _convert_fn(paths, value):
            key = ".".join(map(str, paths))
            if not match_patterns(
                key, include_keys, exclude_keys, precedence="exclude"
            ):
                # key is not matched, we don't convert and return value as-is
                return value
            return any_to_torch_tensor(
                value,
                dtype=_match_pattern(key, dtypes),
                device=device,
                copy=copy,
                non_blocking=non_blocking,
                smart_optimize=True,
            )

        return self.map_structure(_convert_fn, inplace=inplace, with_path=True)


def any_to_datadict(D, resolve: bool = True) -> DataDict:
    from omegaconf import OmegaConf, DictConfig

    if isinstance(D, DataDict):
        return D
    elif isinstance(D, DictConfig):
        return DataDict(OmegaConf.to_container(D, resolve=resolve))
    elif _is_mapping(D):
        return DataDict(D)
    else:
        raise NotImplementedError(
            f"Data type {type(D)} cannot be converted to DataDict"
        )


# ========================================================
# =================== Helper functions ===================
# ========================================================


def _is_mapping(x):
    return isinstance(x, collections.abc.MutableMapping)


def _is_iterable(x):
    return hasattr(x, "__iter__")


def _is_sliceable(x):
    return isinstance(x, (np.ndarray, torch.Tensor))


def _get_keys_set(x):
    if isinstance(x, collections.abc.Mapping):
        return set(x.keys())
    elif isinstance(x, collections.abc.Set):
        return x
    else:
        raise NotImplementedError(
            f"Unsupported type {type(x)}, must be either dict or set."
        )


def _wrap_datadict(objs):
    return tree.traverse(
        lambda x: DataDict(x)
        if _is_mapping(x) and not isinstance(x, DataDict)
        else None,
        objs,
        top_down=False,
    )


def _unwrap_datadict(x):
    if isinstance(x, DataDict):
        return x._data_
    else:
        return None


def _match_pattern(key, pattern_dict):
    """
    Returns:
        matched value
    """
    for pattern, v in pattern_dict.items():
        if pattern is not None and fnmatch.fnmatch(key, pattern):
            return v

    if None in pattern_dict:  # catch-all key
        return pattern_dict[None]
    elif "__default__" in pattern_dict:  # catch-all key
        return pattern_dict["__default__"]
    else:
        return None  # use default dtype


def _preprocess_dtypes(dtypes, mode):
    if dtypes is None:
        dtypes = {}
    elif _is_mapping(dtypes):
        # make the keys nested
        dtypes_flattened = {}
        dtypes = dtypes.copy()
        none_dtype = dtypes.pop(None, None)  # tree lib cannot handle None key

        def _fn(paths, x):
            if not _is_mapping(x):
                dtypes_flattened[".".join(map(str, paths))] = x

        tree.map_structure_with_path(_fn, dtypes)
        dtypes = dtypes_flattened
        if none_dtype:
            dtypes[None] = none_dtype
    else:
        dtypes = {None: dtypes}

    if mode == "numpy":
        return dtypes  # numpy can accept string as type
    elif mode == "torch":
        for k, v in dtypes.items():
            # convert string "float32" to torch.float32
            if isinstance(v, str):
                dtypes[k] = getattr(torch, v)
        return dtypes
    else:
        raise NotImplementedError("INTERNAL", mode)


def match_patterns(
    item: str,
    include: Union[str, List[str], Callable, List[Callable], None] = None,
    exclude: Union[str, List[str], Callable, List[Callable], None] = None,
    *,
    precedence="exclude",
):
    assert precedence in ["include", "exclude"]
    if exclude is None:
        exclude = []
    if isinstance(exclude, (str, Callable)):
        exclude = [exclude]
    if isinstance(include, (str, Callable)):
        include = [include]
    if include is None:
        # exclude is the sole veto vote
        return not _match_patterns_helper(item, exclude)

    if precedence == "include":
        return _match_patterns_helper(item, include)
    else:
        if _match_patterns_helper(item, exclude):
            return False
        else:
            return _match_patterns_helper(item, include)


def _match_patterns_helper(element, patterns):
    for p in patterns:
        if callable(p) and p(element):
            return True
        if fnmatch.fnmatch(element, p):
            return True
    return False


def stack_sequence_fields(sequence):

    # Handle empty input sequences.
    if not sequence:
        raise ValueError("Input sequence must not be empty")

    # Default to asarray when arrays don't have the same shape to be compatible
    # with old behaviour.
    try:
        return fast_map_structure(lambda *values: np.stack(values), *sequence)
    except ValueError:
        return fast_map_structure(lambda *values: np.asarray(values), *sequence)


def fast_map_structure(func, *structure):
    """Faster map_structure implementation which skips some error checking."""
    flat_structure = (tree.flatten(s) for s in structure)
    entries = zip(*flat_structure)
    # Arbitrarily choose one of the structures of the original sequence (the last)
    # to match the structure for the flattened sequence.
    return tree.unflatten_as(structure[-1], [func(*x) for x in entries])


def get_batch_size(x, strict: bool = False) -> int:
    """
    Args:
        x: can be any arbitrary nested structure of np array and torch tensor
        strict: True to check all batch sizes are the same
    """

    def _get_batch_size(x):
        if isinstance(x, np.ndarray):
            return x.shape[0]
        elif torch.is_tensor(x):
            return x.size(0)
        else:
            return len(x)

    xs = tree.flatten(x)

    if strict:
        batch_sizes = [_get_batch_size(x) for x in xs]
        assert all(
            b == batch_sizes[0] for b in batch_sizes
        ), f"batch sizes must all be the same in nested structure: {batch_sizes}"
        return batch_sizes[0]
    else:
        return _get_batch_size(xs[0])


def meta_decorator(decor):
    single_callable = (
        lambda args, kwargs: len(args) == 1 and len(kwargs) == 0 and callable(args[0])
    )

    @functools.wraps(decor)
    def new_decor(*args, **kwargs):
        if single_callable(args, kwargs):
            # this is the double-decorated f.
            # It should not run on a single callable.
            return decor(args[0])
        else:
            # decorator arguments
            return lambda real_f: decor(real_f, *args, **kwargs)

    return new_decor


@meta_decorator
def make_recursive_func(fn, *, with_path=False):
    """
    Decorator that turns a function that works on a single array/tensor to working on
    arbitrary nested structures.
    """

    @functools.wraps(fn)
    def _wrapper(tensor_struct, *args, **kwargs):
        if with_path:
            return tree.map_structure_with_path(
                lambda paths, x: fn(paths, x, *args, **kwargs), tensor_struct
            )
        else:
            return tree.map_structure(lambda x: fn(x, *args, **kwargs), tensor_struct)

    return _wrapper


@make_recursive_func
def any_slice(x, slice):
    """
    Args:
        slice: you can use np.s_[...] to return the slice object
    """
    if is_array_tensor(x):
        return x[slice]
    else:
        return x


def is_array_tensor(obj):
    return isinstance(obj, (np.ndarray, torch.Tensor))


@make_recursive_func
def any_transpose_first_two_axes(x):
    """
    util to convert between (L, B, ...) and (B, L, ...)
    """
    if isinstance(x, np.ndarray):
        return np.swapaxes(x, 0, 1)
    elif torch.is_tensor(x):
        return torch.swapaxes(x, 0, 1)
    else:
        raise ValueError(f"Input ({type(x)}) must be either a numpy array or a tensor.")


@make_recursive_func
def add_batch_dim(x):
    if isinstance(x, np.ndarray):
        return np.expand_dims(x, axis=0)
    elif torch.is_tensor(x):
        return x.unsqueeze(0)
    else:
        raise NotImplementedError(f"Unsupported data structure: {type(x)}")
