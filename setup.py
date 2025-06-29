from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="polyglot-cpp",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A polyglot type system for cross-language compatibility",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/polyglot-type-system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "clang>=14.0",
        "libclang>=14.0.6",
        "dataclasses-json>=0.6.1",
        "llama-index>=0.9.48",
        "qdrant-client>=1.7.0",
        "pydantic>=2.5.3",
        "chromadb>=0.4.22",
        "click>=8.1.7",
        "rich>=13.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "extract-cpp-types=examples.extract_types:main",
        ],
    },
)
