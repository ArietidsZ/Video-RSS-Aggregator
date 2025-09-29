from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="video-aggregator-sdk",
    version="1.0.0",
    author="Video Aggregator Team",
    author_email="support@video-aggregator.com",
    description="Python SDK for Video RSS Aggregator API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/video-aggregator/sdk-python",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "aiohttp>=3.8.0",
        "pydantic>=2.0.0",
        "backoff>=2.0.0",
        "pyjwt>=2.0.0",
        "websockets>=11.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "pylint>=2.17.0",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/video-aggregator/sdk-python/issues",
        "Source": "https://github.com/video-aggregator/sdk-python",
        "Documentation": "https://docs.video-aggregator.com/sdks/python",
    },
)