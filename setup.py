import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nurbspy",
    version="0.1.0",
    author="Roberto Agromayor",
    author_email="roberto.agromayor@ntnu.no",
    description="A lightweight library for NURBS curves and surfaces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RoberAgro/nurbspy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy', 'scipy', 'matplotlib'],
    python_requires='>=3.6',
)
