"""Package metadata and installation settings for the TF2.9 multi-backbone engine."""

from setuptools import setup,find_packages
setup(name="materon-tf29-engine",version="3.6.0",packages=find_packages(),include_package_data=True,package_data={"materon_tf29_engine.features":["*.csv","*.json"]},python_requires=">=3.9,<3.10")
