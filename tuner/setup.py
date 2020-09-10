# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setuptools

setuptools.setup(
    name = 'CUHKPrototypeTuner',
    version = '1.2',
    packages = setuptools.find_packages(exclude=['*test*']),

    python_requires = '>=3.6',
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: ',
        'NNI Package :: tuner :: CUHKPrototypeTuner :: cuhk_prototype_tuner.CUHKPrototypeTuner :: cuhk_prototype_tuner.CUHKPrototypeClassArgsValidator'
    ],

    author = 'CUHKPrototypeTuner Team',
    author_email = 'nni@microsoft.com',
    description = 'NNI control for Neural Network Intelligence project',
    license = 'MIT',
    url = 'https://github.com/Microsoft/nni'
)
