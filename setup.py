from setuptools import setup, find_packages

setup(
    name="hmarl-grf",
    version="0.1.0",
    description="Hierarchical Multi-Agent Reinforcement Learning for Google Research Football",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "torch",
        "numpy",
    ],
)
