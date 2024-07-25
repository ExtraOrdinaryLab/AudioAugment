from setuptools import find_packages, setup


setup(
    name='audio_augment', 
    version='0.0.3', 
    description='Open-source library of AudioAugment', 
    author='Yang Wang', 
    author_email='yangwang4work@gmail.com', 
    package_dir={'': 'src'}, 
    packages=find_packages('src'), 
    python_requires='>=3.9.0', 
    keywords='audio augment augmentation'
)