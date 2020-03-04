import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nurbs-library", # Replace with your own username
    version="0.0.1",
    author="Roberto Agromayor",
    author_email="roberto.agromayor@gmail.com",
    description="A light library for NURBS curves and surfaces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=" ",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
