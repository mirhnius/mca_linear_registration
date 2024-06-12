from setuptools import setup, find_packages

setup(
    name="lnrgst_mca",
    version="0.1",
    author="Niusha",
    description=("A package for studying numerical stability of widely used linear registration techniques."),
    packages=find_packages(),
    install_requires=["numpy", "pandas", "nibabel", "nilearn", "matplotlib", "seaborn", "scikit-learn", "scipy", "gif", "pillow"],
)
