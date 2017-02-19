from distutils.core import setup

version = '0.0.2'

setup(
    name = 'torchfile',
    version = version,
    description = "Torch7 binary serialized file parser",
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    author = 'Brendan Shillingford',
    author_email = 'brendan.shillingford@cs.ox.ac.uk',
    url = 'https://github.com/bshillingford/python-torchfile',
    license = 'BSD',
    py_modules=['torchfile']
)

