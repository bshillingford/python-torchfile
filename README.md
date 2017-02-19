# Torch serialization reader for Python
[![Build Status](https://travis-ci.org/bshillingford/python-torchfile.svg?branch=master)](https://travis-ci.org/bshillingford/python-torchfile)

Mostly direct port of the torch7 Lua and C serialization implementation to 
Python, depending only on `numpy` (and the standard library: `array` 
and `struct`). Sharing of objects including `torch.Tensor`s is preserved.


## Installation:
Install from [PyPI](https://pypi.python.org/pypi/torchfile/0.0.2):
```sh
pip install torchfile
```
or clone this repository, then:
```sh
python setup.py install
```

Supports Python 2.7, 3.4, 3.5, 3.6. Probably others too.

## Example:
### Write from torch, read from Python:
Lua:
```lua
+th> torch.save('/tmp/test.t7', {hello=123, world=torch.rand(1,2,3)})
```
Python:
```python
In [3]: o = torchfile.load('/tmp/test.t7')
In [4]: print o['world'].shape
(1, 2, 3)
In [5]: o
Out[5]: 
{'hello': 123, 'world': array([[[ 0.52291083,  0.29261517,  0.11113465],
         [ 0.01017287,  0.21466237,  0.26572137]]])}
```

### Arbitary torch classes supported:
```python
In [1]: import torchfile

In [2]: o = torchfile.load('testfiles_x86_64/gmodule_with_linear_identity.t7')

In [3]: o.forwardnodes[3].data.module
Out[3]: TorchObject(nn.Identity, {'output': array([], dtype=float64), 'gradInput': array([], dtype=float64)})

In [4]: for node in o.forwardnodes: print(repr(node.data.module))                                                                                                            
None
None
None
TorchObject(nn.Identity, {'output': array([], dtype=float64), 'gradInput': array([], dtype=float64)})
None
TorchObject(nn.Identity, {'output': array([], dtype=float64), 'gradInput': array([], dtype=float64)})
TorchObject(nn.Linear, {'weight': array([[-0.0248373 ],
       [ 0.17503954]]), 'gradInput': array([], dtype=float64), 'gradWeight': array([[  1.22317168e-312],
       [  1.22317168e-312]]), 'bias': array([ 0.05159848, -0.25367146]), 'gradBias': array([  1.22317168e-312,   1.22317168e-312]), 'output': array([], dtype=float64)})
TorchObject(nn.CAddTable, {'output': array([], dtype=float64), 'gradInput': []})
None

In [5]: o.forwardnodes[6].data.module.weight
Out[5]: 
array([[-0.0248373 ],
       [ 0.17503954]])

In [6]: o.forwardnodes[6].data.module.bias
Out[6]: array([ 0.05159848, -0.25367146])
```

### More complex writing from torch:
Lua:
```lua
+th> f = torch.DiskFile('/tmp/test.t7', 'w'):binary()
+th> f:writeBool(false)
+th> f:writeObject({hello=123})
+th> f:writeInt(456)
+th> f:close()
```
Python:
```python
In [1]: import torchfile
In [2]: with open('/tmp/test.t7','rb') as f:
   ...:     r = torchfile.T7Reader(f)
   ...:     print(r.read_boolean())
   ...:     print(r.read_obj())
   ...:     print(r.read_int())
   ...: 
False
{'hello': 123}
456
```


## Supported types:
 * `nil` to Python `None`
 * numbers to Python floats, or by default a heuristic changes them to ints or
   longs if they are integral
 * booleans
 * strings: read as byte strings (Python 3) or normal strings (Python 2), like
   lua strings which don't support unicode, and that can contain null chars
 * tables converted to a special dict (*); if they are list-like (i.e. have
   numeric keys from 1 through n) they become a python list by default
 * Torch classes: supports Tensors and Storages, and most classes such as 
   modules. Trivially extensible much like the Torch serialization code.
   Trivial torch classes like most `nn.Module` subclasses become 
   `TorchObject`s. The `torch_readers` dict contains the mapping from class
   names to reading functions.
 * functions: loaded into the `LuaFunction` `namedtuple`,
   which simply wraps the raw serialized data, i.e. upvalues and code.
   These are mostly useless, but exist so you can deserialize anything.
 * tds.Hash, tds.Vec

(*) Since Lua allows you to index a table with a table but Python does not, we 
    replace dicts with a subclass that is hashable, and change its
    equality comparison behaviour to compare by reference.
    See `hashable_uniq_dict`.


### Test files demonstrating various features:
```python
In [1]: import torchfile

In [2]: torchfile.load('testfiles_x86_64/list_table.t7')
Out[2]: ['hello', 'world', 'third item', 123]

In [3]: torchfile.load('testfiles_x86_64/doubletensor.t7')
Out[3]: 
array([[ 1. ,  2. ,  3. ],
       [ 4. ,  5. ,  6.9]])

# ...also other files demonstrating various types.
```

The example `t7` files will work on any modern Intel or AMD 64-bit CPU, but the
code will use the native byte ordering etc. Currently, the implementation 
assumes the system-dependent binary Torch format, but minor refactoring can 
give support for the ascii format as well.

