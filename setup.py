from setuptools import find_packages, setup


# Function to read the requirements.txt file and return a list of dependencies
def parse_requirements(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        # Filter out any commented or empty lines
        requirements = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    return requirements

# Read requirements from requirements.txt
requirements = parse_requirements('requirements.txt')


setup(
    name='audio_augment', 
    version='0.0.6', 
    description='Open-source library of AudioAugment', 
    author='Yang Wang', 
    author_email='yangwang4work@gmail.com', 
    package_dir={'': 'src'}, 
    packages=find_packages('src'), 
    install_requires=requirements, 
    python_requires='>=3.9.0', 
    keywords='audio augment augmentation'
)