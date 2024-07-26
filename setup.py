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

# Separate Git dependencies from PyPI dependencies
install_requires = [req for req in requirements if not req.startswith('git+')]
dependency_links = [req for req in requirements if req.startswith('git+')]

setup(
    name='audio_augment', 
    version='0.0.7', 
    description='Open-source library of AudioAugment', 
    author='Yang Wang', 
    author_email='yangwang4work@gmail.com', 
    package_dir={'': 'src'}, 
    packages=find_packages('src'), 
    install_requires=install_requires,  # Use the PyPI dependencies from requirements.txt
    dependency_links=dependency_links,  # Use the Git dependencies from requirements.txt
    python_requires='>=3.9.0', 
    keywords='audio augment augmentation'
)