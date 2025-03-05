from setuptools import setup, find_packages

setup(
    name="synthlongread",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "torch>=1.7.0",
        "biopython>=1.78",
        "pysam>=0.16.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.23.0",
        "tqdm>=4.50.0",
        "scipy>=1.5.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A framework for generating synthetic long-read scRNA-seq data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/SynthLongRead",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Intended Audience :: Science/Research"
    ],
    python_requires=">=3.8",
)
