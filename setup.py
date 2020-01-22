from setuptools import setup, find_packages
setup(
    name = 'QFTSampler',
    version = '0.0.0',
    url = 'https://github.com/IntenF/QFTSampler',
    license = 'MIT',
    author = 'Taichi Nakamura, Katsuhiro Endo',
    author_email = 'taichi.nakamura@keio.jp',
    description = 'QFTSampler',
    install_requires = ['numpy', ],
    packages = find_packages(),

)
