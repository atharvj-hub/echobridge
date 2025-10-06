"""Setup script for EchoBridge package"""

from setuptools import setup, find_packages

setup(
    name="echobridge",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.9",
)
