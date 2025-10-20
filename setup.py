from setuptools import setup, find_packages

setup(
    name='TCMRepurposing',          
    version='0.1.0',               
    description='A project for TCM formula knowledge graph and repurposing using PyTorch',
    packages=find_packages(),       
    python_requires='3.6',       
    install_requires=[
        'torch=1.0',               
        'numpy',                     
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

