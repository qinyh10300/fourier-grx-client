from typing import Union, Dict, Optional
import os
import math
import numbers
import zarr
import numcodecs
import numpy as np
from functools import cached_property

def check_chunks_compatible(chunks: tuple, shape: tuple):
    assert len(shape) == len(chunks)
    for c in chunks:
        assert isinstance(c, numbers.Integral)
        assert c > 0

def rechunk_recompress_array(group, name, 
        chunks=None, chunk_length=None,
        compressor=None, tmp_key='_temp'):
    old_arr = group[name]
    if chunks is None:
        if chunk_length is not None:
            chunks = (chunk_length,) + old_arr.chunks[1:]  # 
            """
            # 假设原数组的分块配置是：
            old_arr.chunks = (100, 84, 84, 3)
            # 其中：
            # 100: 时间维度的分块大小
            # 224,: 图像的高度和宽度
            # 3: 颜色通道数

            # 如果设置 chunk_length = 200
            chunks = (200,) + (84, 84, 3)
            # 结果为：(200, 84, 84, 3)
            """
        else:
            chunks = old_arr.chunks
    check_chunks_compatible(chunks, old_arr.shape)
    
    if compressor is None:
        compressor = old_arr.compressor
    
    if (chunks == old_arr.chunks) and (compressor == old_arr.compressor):
        # no change
        return old_arr

    # rechunk recompress
    """
    编程思想：
    1. 将原数组临时移动到一个新的键名
    2. 获取移动后的数组引用
    3. 使用 zarr.copy 进行复制操作，同时应用新的分块和压缩设置
    4. 删除临时数组
    """
    group.move(name, tmp_key)
    old_arr = group[tmp_key]
    n_copied, n_skipped, n_bytes_copied = zarr.copy(
        source=old_arr,     # 源数组
        dest=group,         # 目标组 destination
        name=name,          # 新数组的名称
        chunks=chunks,      # 新的分块设置
        compressor=compressor,  # 新的压缩器设置
    )
    del group[tmp_key]
    arr = group[name]
    return arr

def get_optimal_chunks(shape, dtype, 
        target_chunk_bytes=2e6,  # 目标块大小：2MB
        max_chunk_length=None):
    """
    计算最优的数据分块大小。
    
    例如对于形状 (1000, 224, 224, 3) 的图像数据：
    T=1000: 时间步数
    H=224, W=224: 图像高宽
    C=3: RGB通道数
    """
    # 1. 获取数据类型的字节大小（比如 float32 是 4 字节）
    itemsize = np.dtype(dtype).itemsize
    
    # 2. 反转形状以便从最内层维度开始计算
    # 如 (1000, 84, 84, 3) -> [3, 84, 84, 1000]
    rshape = list(shape[::-1])
    
    # 3. 如果指定了最大长度，则限制时间维度
    if max_chunk_length is not None:
        rshape[-1] = int(max_chunk_length)
    
    # 4. 寻找合适的分割点
    split_idx = len(shape)-1
    for i in range(len(shape)-1):
        # 计算当前维度的字节数
        this_chunk_bytes = itemsize * np.prod(rshape[:i])
        # 计算加入下一维度后的字节数
        next_chunk_bytes = itemsize * np.prod(rshape[:i+1])
        
        # 如果当前大小合适但加入下一维度会超出目标大小
        # 则在这里分割
        if this_chunk_bytes <= target_chunk_bytes \
            and next_chunk_bytes > target_chunk_bytes:
            split_idx = i

    # 5. 构建分块配置
    # 取分割点之前的维度
    rchunks = rshape[:split_idx]
    
    # 计算每个数据项的字节数
    item_chunk_bytes = itemsize * np.prod(rshape[:split_idx])
    
    # 获取分割点维度的大小
    this_max_chunk_length = rshape[split_idx]
    
    # 计算分割点维度应该取多大
    next_chunk_length = min(this_max_chunk_length, math.ceil(
            target_chunk_bytes / item_chunk_bytes))
    
    # 添加分割点维度的大小
    rchunks.append(next_chunk_length)
    
    # 6. 剩余维度都设为1
    len_diff = len(shape) - len(rchunks)
    rchunks.extend([1] * len_diff)
    
    # 7. 反转回原来的维度顺序并返回
    chunks = tuple(rchunks[::-1])
    return chunks


class ReplayBuffer:
    """
    基于 Zarr 的时序数据结构。
    用于存储和管理机器人训练数据，如状态、动作、观察等时序数据。
    假设数据的第一个维度是时间维度，并且只在时间维度上进行分块。

    数据组织结构:
    - root
        - data/  # 存储实际的时序数据
            - observations  # 形状: (T, ...)
            - actions      # 形状: (T, ...)
            - states      # 形状: (T, ...)
        - meta/  # 存储元数据
            - episode_ends  # 记录每个episode的结束位置
    """
    def __init__(self, root: Union[zarr.Group, Dict[str,dict]]):
        """
        构造函数。建议使用 copy_from* 和 create_from* 类方法来创建实例。

        Args:
            root: 可以是 Zarr Group 或包含 data 和 meta 的字典
        """
        assert('data' in root)
        assert('meta' in root)
        assert('episode_ends' in root['meta'])
        for key, value in root['data'].items():
            assert(value.shape[0] == root['meta']['episode_ends'][-1])
        self.root = root
    
    # ============= create constructors ===============
    """
    因为python没有默认的重载方法,所以常用类方法来实现重载，类方法一般也只用于这里
    overload: 重载 
    override: 重写
    # 类方法不用实例化


    root/
    ├── data/           # 存储实际的训练数据
    │   ├── observations  # 观察数据
    │   ├── actions      # 动作数据
    │   └── states       # 状态数据
    │
    └── meta/           # 存储元数据
        └── episode_ends  # 记录每个episode的结束位置
        
    """
    @classmethod
    def create_empty_zarr(cls, storage=None, root=None):
        if root is None:
            if storage is None:
                storage = zarr.MemoryStore() # 内存存储，数据大的话使用DirectoryStore
            root = zarr.group(store=storage)  
        data = root.require_group('data', overwrite=False)
        meta = root.require_group('meta', overwrite=False)
        if 'episode_ends' not in meta:
            episode_ends = meta.zeros('episode_ends', shape=(0,), dtype=np.int64,
                compressor=None, overwrite=False)
        return cls(root=root)
    
    @classmethod
    def create_empty_numpy(cls):
        root = {
            'data': dict(),
            'meta': {
                'episode_ends': np.zeros((0,), dtype=np.int64)
            }
        }
        return cls(root=root)
    
    @classmethod 
    def create_from_group(cls, group, **kwargs):
        # 构造函数的重载
        if 'data' not in group:
            # create from stratch
            buffer = cls.create_empty_zarr(root=group, **kwargs)
        else:
            # already exist
            buffer = cls(root=group, **kwargs)
        return buffer

    @classmethod
    def create_from_path(cls, zarr_path, mode='r', **kwargs):
        """
        直接从磁盘打开 Zarr 数据集（适用于大于内存的数据集）。
        速度较慢，但可以处理超大数据集。

        Args:
            zarr_path: Zarr 数据集的路径
            mode: 打开模式
                - 'r': 只读模式 (默认)
                - 'w': 写入模式
                - 'a': 加模式
            **kwargs: 额外参数
        """
        # 展开用户路径（比如 ~/data）并打开 Zarr 数据集
        group = zarr.open(os.path.expanduser(zarr_path), mode)
        # 使用已打开的 group 创建 ReplayBuffer
        return cls.create_from_group(group, **kwargs)
    
    # ============= copy constructors ===============
    @classmethod
    def copy_from_store(cls, src_store, store=None, keys=None, 
            chunks: Dict[str,tuple]=dict(), 
            compressors: Union[dict, str, numcodecs.abc.Codec]=dict(), 
            if_exists='replace',
            **kwargs):
        """
        将数据从源存储复制到新的存储中。
        支持两种模：
        1. 复制到内存 (store=None)
        2. 复制到新的 Zarr 存储 (store=zarr.Store)
        """
        src_root = zarr.group(src_store)
        root = None
        if store is None:
            # numpy backend
            meta = dict()
            for key, value in src_root['meta'].items():
                if len(value.shape) == 0:
                    meta[key] = np.array(value)
                else:
                    meta[key] = value[:]

            if keys is None:
                keys = src_root['data'].keys()
            data = dict()
            for key in keys:
                arr = src_root['data'][key]
                data[key] = arr[:]

            root = {
                'meta': meta,
                'data': data
            }
        else:
            root = zarr.group(store=store)
            # copy without recompression
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(source=src_store, dest=store,
                source_path='/meta', dest_path='/meta', if_exists=if_exists)
            data_group = root.create_group('data', overwrite=True)
            if keys is None:
                keys = src_root['data'].keys()
            for key in keys:
                value = src_root['data'][key]
                # 解析分块和压缩设置
                cks = cls._resolve_array_chunks(chunks=chunks, key=key, array=value)
                cpr = cls._resolve_array_compressor(compressors=compressors, key=key, array=value)
                
                if cks == value.chunks and cpr == value.compressor:
                    # 如果分块和压缩设置相同，直接复制
                    zarr.copy_store(source=src_store, dest=store,
                        source_path='/data/' + key,
                        dest_path='/data/' + key)
                else:
                    # 需要重新分块或压缩
                    zarr.copy(source=value, dest=data_group, name=key,
                        chunks=cks, compressor=cpr)
        buffer = cls(root=root)
        return buffer
    
    @classmethod
    def copy_from_path(cls, zarr_path, backend=None, store=None, keys=None, 
            chunks: Dict[str,tuple]=dict(), 
            compressors: Union[dict, str, numcodecs.abc.Codec]=dict(), 
            if_exists='replace',
            **kwargs):
        """
        Copy a on-disk zarr to in-memory compressed.
        Recommended
        """
        if backend == 'numpy':
            print('backend argument is deprecated!')
            store = None
        group = zarr.open(os.path.expanduser(zarr_path), 'r')
        return cls.copy_from_store(src_store=group.store, store=store, 
            keys=keys, chunks=chunks, compressors=compressors, 
            if_exists=if_exists, **kwargs)

    # ============= save methods ===============
    def save_to_store(self, store, 
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict(),
            if_exists='replace', 
            **kwargs):
        
        root = zarr.group(store)
        if self.backend == 'zarr':
            # recompression free copy
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                source=self.root.store, dest=store,
                source_path='/meta', dest_path='/meta', if_exists=if_exists)
        else:
            meta_group = root.create_group('meta', overwrite=True)
            # save meta, no chunking
            for key, value in self.root['meta'].items():
                _ = meta_group.array(
                    name=key,
                    data=value, 
                    shape=value.shape, 
                    chunks=value.shape)
        
        # save data, chunk
        data_group = root.create_group('data', overwrite=True)
        for key, value in self.root['data'].items():
            cks = self._resolve_array_chunks(
                chunks=chunks, key=key, array=value)
            cpr = self._resolve_array_compressor(
                compressors=compressors, key=key, array=value)
            if isinstance(value, zarr.Array):
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = '/data/' + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=self.root.store, dest=store,
                        source_path=this_path, dest_path=this_path, if_exists=if_exists)
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value, dest=data_group, name=key,
                        chunks=cks, compressor=cpr, if_exists=if_exists
                    )
            else:
                # numpy
                _ = data_group.array(
                    name=key,
                    data=value,
                    chunks=cks,
                    compressor=cpr
                )
        return store

    def save_to_path(self, zarr_path,             
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict(), 
            if_exists='replace', 
            **kwargs):
        """
        将数据保存指定路径。

        Args:
            zarr_path: 保存路径
            chunks: 数据分块配置
            compressors: 压缩器配置
            if_exists: 如果文件已存在的处理方式
        """
        store = zarr.DirectoryStore(os.path.expanduser(zarr_path))
        return self.save_to_store(store, chunks=chunks, 
            compressors=compressors, if_exists=if_exists, **kwargs)

    @staticmethod
    def resolve_compressor(compressor='default'):
        if compressor == 'default':
            compressor = numcodecs.Blosc(cname='lz4', clevel=5, 
                shuffle=numcodecs.Blosc.NOSHUFFLE)
        elif compressor == 'disk':
            compressor = numcodecs.Blosc('zstd', clevel=5, 
                shuffle=numcodecs.Blosc.BITSHUFFLE)
        return compressor

    @classmethod
    def _resolve_array_compressor(cls, 
            compressors: Union[dict, str, numcodecs.abc.Codec], key, array):
        # allows compressor to be explicitly set to None
        cpr = 'nil'
        if isinstance(compressors, dict):
            if key in compressors:
                cpr = cls.resolve_compressor(compressors[key])
            elif isinstance(array, zarr.Array):
                cpr = array.compressor
        else:
            cpr = cls.resolve_compressor(compressors)
        # backup default
        if cpr == 'nil':
            cpr = cls.resolve_compressor('default')
        return cpr
    
    @classmethod
    def _resolve_array_chunks(cls,
            chunks: Union[dict, tuple], key, array):
        cks = None
        if isinstance(chunks, dict):
            if key in chunks:
                cks = chunks[key]
            elif isinstance(array, zarr.Array):
                cks = array.chunks
        elif isinstance(chunks, tuple):
            cks = chunks
        else:
            raise TypeError(f"Unsupported chunks type {type(chunks)}")
        # backup default
        if cks is None:
            cks = get_optimal_chunks(shape=array.shape, dtype=array.dtype)
        # check
        check_chunks_compatible(chunks=cks, shape=array.shape)
        return cks
    
    # ============= properties =================
    """
    cached_property: 缓存属性 调用第一次的时候计算，之后直接返回缓存值。
    """
    @cached_property
    def data(self):
        return self.root['data']
    
    @cached_property
    def meta(self):
        return self.root['meta']

    def update_meta(self, data):
        # sanitize data
        np_data = dict()
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                np_data[key] = value
            else:
                arr = np.array(value)
                if arr.dtype == object:
                    raise TypeError(f"Invalid value type {type(value)}")
                np_data[key] = arr

        meta_group = self.meta
        if self.backend == 'zarr':
            for key, value in np_data.items():
                _ = meta_group.array(
                    name=key,
                    data=value, 
                    shape=value.shape, 
                    chunks=value.shape,
                    overwrite=True)
        else:
            meta_group.update(np_data)
        
        return meta_group
    
    @property
    def episode_ends(self):
        return self.meta['episode_ends']
    
    def get_episode_idxs(self):
        import numba
        numba.jit(nopython=True)
        def _get_episode_idxs(episode_ends):
            result = np.zeros((episode_ends[-1],), dtype=np.int64)
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                for idx in range(start, end):
                    result[idx] = i
            return result
        return _get_episode_idxs(self.episode_ends)
        
    
    @property
    def backend(self):
        backend = 'numpy'
        if isinstance(self.root, zarr.Group):
            backend = 'zarr'
        return backend
    
    # =========== dict-like API ==============
    def __repr__(self) -> str:
        if self.backend == 'zarr':
            return str(self.root.tree())
        else:
            return super().__repr__()

    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def items(self):
        return self.data.items()
    
    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    # =========== our API ==============
    @property
    def n_steps(self):
        """总时间步数"""
        if len(self.episode_ends) == 0:
            return 0
        return self.episode_ends[-1]
    
    @property
    def n_episodes(self):
        """数据片段（episode）的总数"""
        return len(self.episode_ends)

    @property
    def chunk_size(self):
        if self.backend == 'zarr':
            return next(iter(self.data.arrays()))[-1].chunks[0]
        return None

    @property
    def episode_lengths(self):
        """
        返回所有数据片段的长度数组

        Returns:
            numpy.ndarray: 包含每个episode长度的数组
        """
        ends = self.episode_ends[:]
        ends = np.insert(ends, 0, 0)
        lengths = np.diff(ends)
        return lengths

    def add_episode(self, data: Dict[str, np.ndarray], 
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict()):
        """
        添加一个新的数据片段（episode）到缓存中。

        Args:
            data: 包含时序数据的字典，每个值应该是 shape=(T, ...) 的数组
            chunks: 数据分块大小的置
            compressors: 数据压缩器的配置

        示例:
            buffer.add_episode({
                'observations': obs_array,  # shape=(100, 84, 84, 3)
                'actions': action_array,    # shape=(100, 8)
                'states': state_array       # shape=(100, 32)
            })
        """
        assert(len(data) > 0)
        is_zarr = (self.backend == 'zarr')

        # 获取当前数据长度和新数据长度
        curr_len = self.n_steps
        episode_length = None
        for key, value in data.items():
            assert(len(value.shape) >= 1)
            if episode_length is None:
                episode_length = len(value)
            else:
                assert(episode_length == len(value))
        new_len = curr_len + episode_length

        # 为每个据键添加新数据
        for key, value in data.items():
            new_shape = (new_len,) + value.shape[1:]
            # 创建或调整数组大小
            if key not in self.data:
                if is_zarr:
                    cks = self._resolve_array_chunks(
                        chunks=chunks, key=key, array=value)
                    cpr = self._resolve_array_compressor(
                        compressors=compressors, key=key, array=value)
                    arr = self.data.zeros(name=key, 
                        shape=new_shape, 
                        chunks=cks,
                        dtype=value.dtype,
                        compressor=cpr)
                else:
                    arr = np.zeros(shape=new_shape, dtype=value.dtype)
                    self.data[key] = arr
            else:
                arr = self.data[key]
                assert(value.shape[1:] == arr.shape[1:])
                if is_zarr:
                    arr.resize(new_shape)
                else:
                    arr.resize(new_shape, refcheck=False)
            # 复制新数据
            arr[-value.shape[0]:] = value
        
        # 更新 episode_ends
        episode_ends = self.episode_ends
        if is_zarr:
            episode_ends.resize(episode_ends.shape[0] + 1)
        else:
            episode_ends.resize(episode_ends.shape[0] + 1, refcheck=False)
        episode_ends[-1] = new_len

    def drop_episode(self):
        is_zarr = (self.backend == 'zarr')
        episode_ends = self.episode_ends[:].copy()
        assert(len(episode_ends) > 0)
        start_idx = 0
        if len(episode_ends) > 1:
            start_idx = episode_ends[-2]
        for key, value in self.data.items():
            new_shape = (start_idx,) + value.shape[1:]
            if is_zarr:
                value.resize(new_shape)
            else:
                value.resize(new_shape, refcheck=False)
        if is_zarr:
            self.episode_ends.resize(len(episode_ends)-1)
        else:
            self.episode_ends.resize(len(episode_ends)-1, refcheck=False)
    
    def pop_episode(self):
        assert(self.n_episodes > 0)
        episode = self.get_episode(self.n_episodes-1, copy=True)
        self.drop_episode()
        return episode

    def extend(self, data):
        self.add_episode(data)

    def get_episode(self, idx, copy=False):
        """
        获取指定索引的数据片段。

        Args:
            idx: episode的索引
            copy: 是否返回数据的副本

        Returns:
            包含该episode所有数据的字典
        """
        idx = list(range(len(self.episode_ends)))[idx]
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx-1]
        end_idx = self.episode_ends[idx]
        result = self.get_steps_slice(start_idx, end_idx, copy=copy)
        return result
    
    def get_episode_slice(self, idx):
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx-1]
        end_idx = self.episode_ends[idx]
        return slice(start_idx, end_idx)

    def get_steps_slice(self, start, stop, step=None, copy=False):
        _slice = slice(start, stop, step)

        result = dict()
        for key, value in self.data.items():
            x = value[_slice]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x
        return result
    
    # =========== chunking =============
    def get_chunks(self) -> dict:
        assert self.backend == 'zarr'
        chunks = dict()
        for key, value in self.data.items():
            chunks[key] = value.chunks
        return chunks
    
    def set_chunks(self, chunks: dict):
        assert self.backend == 'zarr'
        for key, value in chunks.items():
            if key in self.data:
                arr = self.data[key]
                if value != arr.chunks:
                    check_chunks_compatible(chunks=value, shape=arr.shape)
                    rechunk_recompress_array(self.data, key, chunks=value)

    def get_compressors(self) -> dict:
        assert self.backend == 'zarr'
        compressors = dict()
        for key, value in self.data.items():
            compressors[key] = value.compressor
        return compressors

    def set_compressors(self, compressors: dict):
        assert self.backend == 'zarr'
        for key, value in compressors.items():
            if key in self.data:
                arr = self.data[key]
                compressor = self.resolve_compressor(value)
                if compressor != arr.compressor:
                    rechunk_recompress_array(self.data, key, compressor=compressor)
