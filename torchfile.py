"""
Mostly direct port of the Lua and C serialization implementation to 
Python, depending only on `struct`, `array`, and numpy.

Supported types:
 * `nil` to Python `None`
 * numbers to Python floats
 * booleans
 * tables unconditionally to a special dict (*), regardless of whether they 
   are list-like)
 * Torch classes: supports Tensors and Storages, and most classes such as 
   modules. Trivially extensible much like the Torch serialization code.
   Trivial torch classes like most `nn.Module` subclasses become `TorchObject`
   `namedtuple`s.
 * functions: loaded into the `LuaFunction` `namedtuple`,
   which simply wraps the raw serialized data, i.e. upvalues and code.
   These are mostly useless, but exist so you can deserialize anything.

(*) Since Lua allows you to index a table with a table but not Python, we 
    replace dicts with a subclass that is hashable, and change its
    equality comparison behaviour to compare by reference.
    See `hashable_uniq_dict`.

Currently, the implementation assumes the system-dependent binary Torch 
format, but minor refactoring can give support for the ascii format as well.
"""

TYPE_NIL      = 0
TYPE_NUMBER   = 1
TYPE_STRING   = 2
TYPE_TABLE    = 3
TYPE_TORCH    = 4
TYPE_BOOLEAN  = 5
TYPE_FUNCTION = 6
TYPE_RECUR_FUNCTION = 8
LEGACY_TYPE_RECUR_FUNCTION = 7

import struct
from array import array
import re
import numpy as np
import sys
from collections import namedtuple

filename = sys.argv[1]

TorchObject = namedtuple('TorchObject', ['typename', 'obj'])
LuaFunction = namedtuple('LuaFunction',
                         ['size', 'dumped', 'upvalues'])

class hashable_uniq_dict(dict):
    """
    Subclass of dict with equality and hashing semantics changed:
    equality and hashing is purely by reference/instance, to match
    the behaviour of lua tables.

    This way, dicts can be keys of other dicts.
    """
    def __hash__(self):
        return id(self)
    def __eq__(self, other):
        return id(self) == id(other)
    # TODO: dict's __lt__ etc. still exist

torch_readers = {}

def add_tensor_reader(typename, dtype):
    def read_tensor_generic(f, version):
        # source: https://github.com/torch/torch7/blob/master/generic/Tensor.c#L1243
        ndim, = read('i')

        # read size:
        arr = array('l')
        arr.fromfile(f, ndim)
        size = arr.tolist()
        # read stride:
        arr = array('l')
        arr.fromfile(f, ndim)
        stride = arr.tolist()
        # storage offset:
        storage_offset = read('l')[0] - 1
        # read storage:
        storage = read_obj()

        # DEBUG:
        print(ndim, size, stride, storage_offset, storage)

        if storage is None or ndim == 0 or len(size) == 0 or len(stride) == 0:
            # empty torch tensor
            return np.empty((), dtype=dtype)

        # convert stride to numpy style (i.e. in bytes)
        stride = [storage.dtype.itemsize * x for x in stride]

        # create numpy array that indexes into the storage:
        return np.lib.stride_tricks.as_strided(
                storage[storage_offset:],
                shape=size,
                strides=stride)
    torch_readers[typename] = read_tensor_generic
add_tensor_reader('torch.ByteTensor', dtype=np.uint8)
add_tensor_reader('torch.CharTensor', dtype=np.int8)
add_tensor_reader('torch.ShortTensor', dtype=np.int16)
add_tensor_reader('torch.IntTensor', dtype=np.int32)
add_tensor_reader('torch.FloatTensor', dtype=np.float32)
add_tensor_reader('torch.DoubleTensor', dtype=np.float64)
add_tensor_reader('torch.CudaTensor', np.float32)  # float


def add_storage_reader(typename, dtype):
    def read_storage(f, version):
        # source: https://github.com/torch/torch7/blob/master/generic/Storage.c#L244
        size, = read('l')
        return np.fromfile(f, dtype=dtype, count=size)
    torch_readers[typename] = read_storage
add_storage_reader('torch.ByteStorage', dtype=np.uint8)
add_storage_reader('torch.CharStorage', dtype=np.int8)
add_storage_reader('torch.ShortStorage', dtype=np.int16)
add_storage_reader('torch.IntStorage', dtype=np.int32)
add_storage_reader('torch.FloatStorage', dtype=np.float32)
add_storage_reader('torch.DoubleStorage', dtype=np.float64)
add_storage_reader('torch.CudaStorage', dtype=np.float32)  # float


def add_trivial_class_reader(typename):
    def reader(f, version):
        obj = read_obj()
        return TorchObject(typename, obj)
    torch_readers[typename] = reader
for mod in ["nn.ConcatTable", "nn.SpatialAveragePooling",
"nn.TemporalConvolutionFB", "nn.BCECriterion", "nn.Reshape", "nn.gModule",
"nn.SparseLinear", "nn.WeightedLookupTable", "nn.CAddTable",
"nn.TemporalConvolution", "nn.PairwiseDistance", "nn.WeightedMSECriterion",
"nn.SmoothL1Criterion", "nn.TemporalSubSampling", "nn.TanhShrink",
"nn.MixtureTable", "nn.Mul", "nn.LogSoftMax", "nn.Min", "nn.Exp", "nn.Add",
"nn.BatchNormalization", "nn.AbsCriterion", "nn.MultiCriterion",
"nn.LookupTableGPU", "nn.Max", "nn.MulConstant", "nn.NarrowTable", "nn.View",
"nn.ClassNLLCriterionWithUNK", "nn.VolumetricConvolution",
"nn.SpatialSubSampling", "nn.HardTanh", "nn.DistKLDivCriterion",
"nn.SplitTable", "nn.DotProduct", "nn.HingeEmbeddingCriterion",
"nn.SpatialBatchNormalization", "nn.DepthConcat", "nn.Sigmoid",
"nn.SpatialAdaptiveMaxPooling", "nn.Parallel", "nn.SoftShrink",
"nn.SpatialSubtractiveNormalization", "nn.TrueNLLCriterion", "nn.Log",
"nn.SpatialDropout", "nn.LeakyReLU", "nn.VolumetricMaxPooling",
"nn.KMaxPooling", "nn.Linear", "nn.Euclidean", "nn.CriterionTable",
"nn.SpatialMaxPooling", "nn.TemporalKMaxPooling", "nn.MultiMarginCriterion",
"nn.ELU", "nn.CSubTable", "nn.MultiLabelMarginCriterion", "nn.Copy",
"nn.CuBLASWrapper", "nn.L1HingeEmbeddingCriterion",
"nn.VolumetricAveragePooling", "nn.StochasticGradient",
"nn.SpatialContrastiveNormalization", "nn.CosineEmbeddingCriterion",
"nn.CachingLookupTable", "nn.FeatureLPPooling", "nn.Padding", "nn.Container",
"nn.MarginRankingCriterion", "nn.Module", "nn.ParallelCriterion",
"nn.DataParallelTable", "nn.Concat", "nn.CrossEntropyCriterion",
"nn.LookupTable", "nn.SpatialSoftMax", "nn.HardShrink", "nn.Abs", "nn.SoftMin",
"nn.WeightedEuclidean", "nn.Replicate", "nn.DataParallel",
"nn.OneBitQuantization", "nn.OneBitDataParallel", "nn.AddConstant", "nn.L1Cost",
"nn.HSM", "nn.PReLU", "nn.JoinTable", "nn.ClassNLLCriterion", "nn.CMul",
"nn.CosineDistance", "nn.Index", "nn.Mean", "nn.FFTWrapper", "nn.Dropout",
"nn.SpatialConvolutionCuFFT", "nn.SoftPlus", "nn.AbstractParallel",
"nn.SequentialCriterion", "nn.LocallyConnected",
"nn.SpatialDivisiveNormalization", "nn.L1Penalty", "nn.Threshold", "nn.Power",
"nn.Sqrt", "nn.MM", "nn.GroupKMaxPooling", "nn.CrossMapNormalization",
"nn.ReLU", "nn.ClassHierarchicalNLLCriterion", "nn.Optim", "nn.SoftMax",
"nn.SpatialConvolutionMM", "nn.Cosine", "nn.Clamp", "nn.CMulTable",
"nn.LogSigmoid", "nn.LinearNB", "nn.TemporalMaxPooling", "nn.MSECriterion",
"nn.Sum", "nn.SoftSign", "nn.Normalize", "nn.ParallelTable", "nn.FlattenTable",
"nn.CDivTable", "nn.Tanh", "nn.ModuleFromCriterion", "nn.Square", "nn.Select",
"nn.GradientReversal", "nn.SpatialFullConvolutionMap", "nn.SpatialConvolution",
"nn.Criterion", "nn.SpatialConvolutionMap", "nn.SpatialLPPooling",
"nn.Sequential", "nn.Transpose", "nn.SpatialUpSamplingNearest",
"nn.SpatialFullConvolution", "nn.ModelParallel", "nn.RReLU",
"nn.SpatialZeroPadding", "nn.Identity", "nn.Narrow", "nn.MarginCriterion",
"nn.SelectTable", "nn.VolumetricFullConvolution",
"nn.SpatialFractionalMaxPooling", "fbnn.ProjectiveGradientNormalization",
"fbnn.Probe", "fbnn.SparseLinear", "cudnn._Pooling3D",
"cudnn.VolumetricMaxPooling", "cudnn.SpatialCrossEntropyCriterion",
"cudnn.VolumetricConvolution", "cudnn.SpatialAveragePooling", "cudnn.Tanh",
"cudnn.LogSoftMax", "cudnn.SpatialConvolution", "cudnn._Pooling",
"cudnn.SpatialMaxPooling", "cudnn.ReLU", "cudnn.SpatialCrossMapLRN",
"cudnn.SoftMax", "cudnn._Pointwise", "cudnn.SpatialSoftMax", "cudnn.Sigmoid",
"cudnn.SpatialLogSoftMax", "cudnn.VolumetricAveragePooling", "nngraph.Node",
"nngraph.JustTable", "graph.Edge", "graph.Node", "graph.Graph"]:
    add_trivial_class_reader(mod)


f = open(filename, 'rb')
objects = {}  # read objects so far
def read(fmt):
    sz = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(sz))

def read_obj():
    typeidx, = read('i')
    if typeidx == TYPE_NIL:
        return None
    elif typeidx == TYPE_NUMBER:
        return read('d')[0]
    elif typeidx == TYPE_BOOLEAN:
        return read('i')[0] == 1
    elif typeidx == TYPE_STRING:
        size, = read('i')
        return f.read(size)
    elif (typeidx == TYPE_TABLE or typeidx == TYPE_TORCH
            or typeidx == TYPE_FUNCTION or typeidx == TYPE_RECUR_FUNCTION 
            or typeidx == LEGACY_TYPE_RECUR_FUNCTION):
        # read the index
        index, = read('i')

        # check it is loaded already
        if index in objects:  # TODO: what is force?
            return objects[index]

        # otherwise read it
        if (typeidx == TYPE_FUNCTION or typeidx == TYPE_RECUR_FUNCTION 
                or typeidx == LEGACY_TYPE_RECUR_FUNCTION):
            size, = read('i')
            dumped = f.read(size)
            upvalues = read_obj()
            obj = LuaFunction(size, dumped, upvalues)
            objects[index] = obj
            return obj
        elif typeidx == TYPE_TORCH:
            version = f.read(read('i')[0])
            try:
                versionNumber = int(re.match(r'^V (.*)$', version).group(1))
            except:
                versionNumber = None
            if not versionNumber:
                className = version
                versionNumber = 0  # created before existence of versioning
            else:
                className = f.read(read('i')[0])
            if className not in torch_readers:
                raise Exception('unsupported torch class: <%s>' % className)
            print('reading type: '+className)  # TODO: remove line
            obj = torch_readers[className](f, version)
            objects[index] = obj
            return obj
        else:  # it is a table
            size, = read('i')
            obj = hashable_uniq_dict()  # custom hashable dict, can be a key
            objects[index] = obj
            for i in range(size):
                k = read_obj()
                v = read_obj()
                obj[k] = v
            return obj
    else:
        raise Exception("unknown object")

print(read_obj())

