from setuptools import setup, find_packages

setup(
    name="attention_swapper",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["torch"],
    description="An easy way to swap out all the newest attention blocks in pytorch",
    author="Your Name",
    author_email="your.email@example.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
) 