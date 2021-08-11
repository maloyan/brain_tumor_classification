from setuptools import find_packages, setup


with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setup(
    name='brain_tumor_classification',
    packages=find_packages(),
    version='0.1.0',
    description='RSNA-MICCAI Brain Tumor Radiogenomic Classification',
    author='Narek Maloyan',
    license='MIT',
)
