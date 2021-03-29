import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

INSTALL_REQUIRES = [
    'scikit-learn>=0.24',
    'numpy',
    'scipy',
    'pandas'
]

EXTRAS_REQUIRE = {
    'code-check': [
        'pytest',
        'mypy',
        'flake8',
        'pylint'
    ],
    'tests': [
        'pytest',
        'pytest-cov',
        'pytest-xdist',
    ],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
    ]
}

setuptools.setup(
    name="NoiseFiltersPy",
    version="0.0.8",
    author="Juliana Hosoume and Luis Faina",
    author_email="ju.hosoume@gmail.com",
    description="Python implementation of NoiseFiltersR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jhosoume/NoiseFiltersPy.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=INSTALL_REQUIRES,
    python_requires='>=3.5',
)
