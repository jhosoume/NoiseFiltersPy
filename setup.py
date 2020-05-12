import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NoiseFiltersPy",
    version="0.0.5",
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
    python_requires='>=3.5',
)
