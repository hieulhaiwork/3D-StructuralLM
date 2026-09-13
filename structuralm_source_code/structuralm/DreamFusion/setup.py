from setuptools import find_packages, setup

setup(
    name="threestudio",
    version="0.2.3",
    packages=find_packages(),
    url="https://github.com/threestudio-project/threestudio",
    author="Yuan-Chen Guo and Ruizhi Shao and Ying-Tian Liu and Christian Laforte and Vikram Voleti and Guan Luo and Chia-Hao Chen and Zi-Xin Zou and Chen Wang and Yan-Pei Cao and Song-Hai Zhang",  # replace with your name
    author_email="shaorz20@mails.tsinghua.edu.cn",
    description="threestudio is a unified framework for 3D content creation from text prompts, single images, and few-shot images, by lifting 2D text-to-image generation models.",  # replace with a brief description of your project
    install_requires=[],
    classifiers=[
        "License :: Apache-2.0",
        "Programming Language :: Python :: 3",
    ],
)
