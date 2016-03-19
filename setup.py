import os
import setuptools


setuptools.setup(
    name='ARENA',
    version="0.1.dev0",
    author="SHI Xingjian, Canceng Zeng",
    author_email="xshiab@cse.ust.hk, czeng@connect.ust.hk",
    packages=setuptools.find_packages(),
    description='A Distributed Asynchronous Reinforcement Learning Toolbox',
    long_description=open(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'README.md')).read(),
    license='MIT',
    url='https://github.com/peterzcc/dist_train/tree/arena',
    install_requires=['numpy', 'scipy', 'pillow', 'matplotlib'],
    classifiers=['Development Status :: 2 - Pre-Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Natural Language :: English',
                 'Operating System :: OS Independent',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
)
