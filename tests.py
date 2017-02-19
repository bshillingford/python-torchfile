import unittest
import torchfile
import os.path


def make_filename(fn):
    TEST_FILE_DIRECTORY = 'testfiles_x86_64'
    return os.path.join(TEST_FILE_DIRECTORY, fn)


class TestRecursiveObjects(unittest.TestCase):
    def test_recursive_class(self):
        obj = torchfile.load(make_filename('recursive_class.t7'))
        self.assertEqual(obj.a, obj)

    def test_recursive_table(self):
        obj = torchfile.load(make_filename('recursive_kv_table.t7'))
        # both the key and value point to itself:
        key, = obj.keys()
        self.assertEqual(key, obj)
        self.assertEqual(obj[key], obj)

class TestHeuristics(unittest.TestCase):
    pass  # TODO

if __name__ == '__main__':
    unittest.main()

