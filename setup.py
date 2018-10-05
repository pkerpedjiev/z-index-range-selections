from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext

install_requires = [
        'numba',
        'numpy'
]

setup(
    name='zindex',
    version='0.1.0',
    description='Z-Index based sorting and querying of sparse matrices',
    author='Peter Kerpedjiev',
    author_email='pkerpedjiev@gmail.com',
    url='',
    install_requires=install_requires,
    packages=['zindex']
)
