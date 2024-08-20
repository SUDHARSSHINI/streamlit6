from setuptools import setup, find_packages

setup(
    name='your_app_name',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'streamlit==1.26.0',
        'pandas==2.0.3',
        'numpy==1.24.3',
        'scikit-learn==1.3.0',
        'responsibleai==0.5.0',
        'aif360==0.5.0'
    ],
)
