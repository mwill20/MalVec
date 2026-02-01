"""
MalVec - Malware Detection via Embedding-Space Analysis

A secure, production-ready tool for identifying polymorphic malware variants
using embedding geometry and K-NN classification.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="malvec",
    version="0.1.0",
    author="MalVec Team",
    description="Malware detection via embedding-space analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/malvec",
    packages=find_packages(exclude=["tests", "tests.*", "research", "scripts"]),
    python_requires=">=3.11",
    install_requires=[
        "ember-ml>=0.1.0",
        "sentence-transformers>=2.2.0",
        "torch>=2.0.0",
        "faiss-cpu>=1.7.4",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "pefile>=2023.2.7",
        "lief>=0.13.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "plotly>=5.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "malvec-train=scripts.train:main",
            "malvec-classify=scripts.classify:main",
            "malvec-visualize=scripts.visualize:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="malware detection embeddings machine-learning security",
)
