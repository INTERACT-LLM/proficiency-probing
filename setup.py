from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="proficiency-probing",
    version="0.1.0",
    author="INTERACT-LLM",
    description="A pipeline for embedding texts, fitting a linear probe, and testing generalizability of the probe on other distributions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/INTERACT-LLM/Proficiency_Probing",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
