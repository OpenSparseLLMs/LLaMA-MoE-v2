import os

import setuptools

readme_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
with open(readme_filepath, "r", encoding="utf8") as fh:
    long_description = fh.read()

version_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VERSION")
with open(version_filepath, "r", encoding="utf8") as fh:
    version = fh.read().strip()

setuptools.setup(
    name="smoe",
    version=version,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=setuptools.find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    extras_require={
        "dev": [
            "pytest",
            "coverage",
            "black",
            "isort",
            "flake8",
            "pre-commit",
        ]
    },
    include_package_data=True,
    entry_points={},
)
