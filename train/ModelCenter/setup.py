from setuptools import setup, find_packages
import os

def main():
    setup(
        name='model-center',
        version='1.0.3',
        description="example codes for big models using bmtrain",
        author="Weilin Zhao",
        author_email="acha131441373@gmail.com",
        packages=find_packages(),
        url="https://github.com/OpenBMB/ModelCenter",
        install_requires=[
            "bmtrain",
            "transformers>=4.28.0",
            "jieba",
        ],
        keywords="CPM, cuda, AI, model, transformer",
        license='Apache 2.0',
    )

if __name__ == '__main__':
    main()
