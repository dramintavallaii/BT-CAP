from setuptools import setup, find_packages

setup(
    name="BT-CAP",
    version="1.1.0",
    author="Amin Tavallaii",
    author_email="dr.amin.tavallaii@gmail.com",
    description="Brain Tumor Compositional Augmentation Pipeline",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dramintavallaii/BT-CAP",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "SimpleITK",
        "scipy",
        "scikit-learn",
        "tqdm",
        "scikit-image",
    ],
    entry_points={
        "console_scripts": [
            "BT-CAP=BT_CAP.main:main",
        ],
    },
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)