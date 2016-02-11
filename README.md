# WIP: refactor into library soon
# Torch serialization reader for Python

Mostly direct port of the Lua and C serialization implementation to 
Python, depending only on `struct`, `array`, and numpy.

## Supported types:
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

