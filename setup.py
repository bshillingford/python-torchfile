from distutils.core import setup

version = '0.0.1'

setup(
    name = 'torchfile',
    version = version,
    description = "Torch7 binary serialized file parser",
    classifiers = [
        'Topic :: Software Development :: Libraries',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5'
    ],
    author = 'Brendan Shillingford',
    author_email = 'brendan.shillingford@cs.ox.ac.uk',
    url = 'https://github.com/bshillingford/python-torchfile',
    license = 'MIT',
    py_modules=['torchfile']
)

