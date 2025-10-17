from setuptools import setup, find_packages

setup(
    name='TCMRepurposing',          # 项目名称
    version='0.1.0',                # 版本号
    description='A project for TCM formula knowledge graph and repurposing using PyTorch',
    packages=find_packages(),       # 自动查找所有包
    python_requires='3.6',        # Python 版本要求
    install_requires=[
        'torch=1.0',               # PyTorch 版本要求
        'numpy',                     # 可以根据项目实际依赖增加其他包
        'pandas',
        'scipy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
        ],
    },
)

