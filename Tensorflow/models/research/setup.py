"""Setup script for object_detection with TF2.0."""
import os
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'apache-beam==2.57.0',
    'avro-python3==1.10.2',
    'click==8.1.7',
    'contextlib2==21.6.0',
    'Cython==0.29.28',
    'GDAL==3.9.2',
    'keras==2.15.0',
    'lvis==0.5.3',
    'lxml==5.2.2',
    'matplotlib==3.9.1',
    'pandas==2.2.2',
    'pillow==10.4.0',
    'progressbar2==4.4.2',
    'pycocotools==2.0.8',
    'pyparsing==2.4.7',
    'sacrebleu==2.2.0',
    'scipy==1.14.0',
    'six==1.16.0',
    'tensorflow==2.15.0',
    'tensorflow-io==0.37.1',
    'tf-models-official==2.15.0',
]


setup(
    name='object_detection',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=(
        [p for p in find_packages() if p.startswith('object_detection')] +
        find_packages(where=os.path.join('.', 'slim'))),
    package_dir={
        'datasets': os.path.join('slim', 'datasets'),
        'nets': os.path.join('slim', 'nets'),
        'preprocessing': os.path.join('slim', 'preprocessing'),
        'deployment': os.path.join('slim', 'deployment'),
        'scripts': os.path.join('slim', 'scripts'),
    },
    description='Tensorflow Object Detection Library',
    python_requires='>3.6',
)
