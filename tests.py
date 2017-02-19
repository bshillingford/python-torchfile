import unittest
import torchfile
import os.path
import sys
import numpy as np


unicode_type = str if sys.version_info > (3,) else unicode


def make_filename(fn):
    TEST_FILE_DIRECTORY = 'testfiles_x86_64'
    return os.path.join(TEST_FILE_DIRECTORY, fn)


def load(fn, **kwargs):
    return torchfile.load(make_filename(fn), **kwargs)


class TestBasics(unittest.TestCase):

    def test_dict(self):
        obj = load('hello=123.t7')
        self.assertEqual(dict(obj), {b'hello': 123})

    def test_custom_class(self):
        obj = load('custom_class.t7')
        self.assertEqual(obj.torch_typename(), b"Blah")

    def test_classnames_never_decoded(self):
        obj = load('custom_class.t7', utf8_decode_strings=True)
        self.assertNotIsInstance(obj.torch_typename(), unicode_type)

        obj = load('custom_class.t7', utf8_decode_strings=False)
        self.assertNotIsInstance(obj.torch_typename(), unicode_type)

    def test_basic_tensors(self):
        f64 = load('doubletensor.t7')
        self.assertTrue((f64 == np.array([[1, 2, 3, ], [4, 5, 6.9]],
                                         dtype=np.float64)).all())

        f32 = load('floattensor.t7')
        self.assertAlmostEqual(f32.sum(), 12.97241666913, delta=1e-5)

    def test_function(self):
        func_with_upvals = load('function_upvals.t7')
        self.assertIsInstance(func_with_upvals, torchfile.LuaFunction)

    def test_dict_accessors(self):
        obj = load('hello=123.t7',
                   use_int_heuristic=True,
                   utf8_decode_strings=True)
        self.assertIsInstance(obj['hello'], int)
        self.assertIsInstance(obj.hello, int)

        obj = load('hello=123.t7',
                   use_int_heuristic=True,
                   utf8_decode_strings=False)
        self.assertIsInstance(obj[b'hello'], int)
        self.assertIsInstance(obj.hello, int)


class TestRecursiveObjects(unittest.TestCase):

    def test_recursive_class(self):
        obj = load('recursive_class.t7')
        self.assertEqual(obj.a, obj)

    def test_recursive_table(self):
        obj = load('recursive_kv_table.t7')
        # both the key and value point to itself:
        key, = obj.keys()
        self.assertEqual(key, obj)
        self.assertEqual(obj[key], obj)


class TestTDS(unittest.TestCase):

    def test_hash(self):
        obj = load('tds_hash.t7')
        self.assertEqual(len(obj), 3)
        self.assertEqual(obj[1], 2)
        self.assertEqual(obj[10], 11)

    def test_vec(self):
        # Should not be affected by list heuristic at all
        vec = load('tds_vec.t7', use_list_heuristic=False)
        self.assertEqual(vec, [123, 456])


class TestHeuristics(unittest.TestCase):

    def test_list_heuristic(self):
        obj = load('list_table.t7', use_list_heuristic=True)
        self.assertEqual(obj, [b'hello', b'world', b'third item', 123])

        obj = load('list_table.t7',
                   use_list_heuristic=False,
                   use_int_heuristic=True)
        self.assertEqual(
            dict(obj),
            {1: b'hello', 2: b'world', 3: b'third item', 4: 123})

    def test_int_heuristic(self):
        obj = load('hello=123.t7', use_int_heuristic=True)
        self.assertIsInstance(obj[b'hello'], int)

        obj = load('hello=123.t7', use_int_heuristic=False)
        self.assertNotIsInstance(obj[b'hello'], int)

        obj = load('list_table.t7',
                   use_list_heuristic=False,
                   use_int_heuristic=False)
        self.assertEqual(
            dict(obj),
            {1: b'hello', 2: b'world', 3: b'third item', 4: 123})
        self.assertNotIsInstance(list(obj.keys())[0], int)


if __name__ == '__main__':
    unittest.main()
