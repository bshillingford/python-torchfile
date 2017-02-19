# t7 test files

These `.t7` files are all created on Linux x86_64. In practice, this is relevant because of the differing size of a long (see [this PR](https://github.com/bshillingford/python-torchfile/pull/1)) in VC++ and hence win32 Python, and almost all saved `t7` files are little-endian.

Non-exhaustive explanation of test files:

* `recursive_class.t7`: instance of a class created as `torch.class("A")`, containing a reference to itself
* `recursive_kv_table.t7`: a table containing one element, with key and value both referencing the table they are contained in

Some other test files check for the correctness of the heuristics. See `tests.py`.

