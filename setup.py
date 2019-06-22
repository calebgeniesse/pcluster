from setuptools import find_packages, setup

# parse requirements.txt
with open('requirements.txt') as f:
    install_requires = [_ for _ in f.read().split('\n') 
                        if len(_) and _[0].isalpha()]

# parse README.md
with open('README.md') as f:
    long_description = f.read()

# run setup
setup(
    name='pcluster',
    version='0.0.1',
    description='A partial clusterer for Mapper',
    long_description=long_description,
    long_description_content_type="text/markdown",	
    author='Caleb Geniesse',
    author_email='geniesse@stanford.edu',
    #url='https://calebgeniesse.github.io/pcluster',
    license='BSD-3',
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    python_requires='>=3.6',
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords='clustering, mapper, topological data analysis',
)
