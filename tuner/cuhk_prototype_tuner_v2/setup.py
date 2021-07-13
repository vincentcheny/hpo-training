# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setuptools

setuptools.setup(
    name = 'CUHKPrototypeTunerV2',
    version = '2.1.4',
    packages = setuptools.find_packages(exclude=['*test*']),

    python_requires = '>=3.6',
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: ',
        'NNI Package :: advisor :: CUHKPrototypeTunerV2 :: cuhk_prototype_tuner_v2.cuhk_prototype_tuner_v2_advisor.CUHKPrototypeTunerV2 :: cuhk_prototype_tuner_v2.cuhk_prototype_tuner_v2_advisor.CUHKPrototypeTunerV2ClassArgsValidator'
    ],
    install_requires = [
        'ConfigSpace==0.4.7'
    ],
    author = 'CUHK',
    author_email = 'cyliu@cse.cuhk.edu.hk',
    description = 'NNI control for Neural Network Intelligence project',
    license = 'MIT',
    url = 'https://github.com/vincentcheny/hpo-training'
)
