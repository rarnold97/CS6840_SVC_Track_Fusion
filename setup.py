from setuptools import setup, find_packages
from pathlib import Path


PACKAGE_LIST = (
    'astropy==5.1.1',
    'astroquery==0.4.6',
    'attrs==22.1.0',
    'beautifulsoup4==4.11.1',
    'cachetools==5.2.0',
    'certifi==2022.9.24',
    'charset-normalizer==2.1.1',
    'contourpy==1.0.6',
    'cycler==0.11.0',
    'czml3==0.7.0',
    'fonttools==4.38.0',
    'html5lib==1.1',
    'idna==3.4',
    'importlib-metadata==5.1.0',
    'jaraco.classes==3.2.3',
    'joblib==1.2.0',
    'jplephem==2.18',
    'keyring==23.11.0',
    'kiwisolver==1.4.4',
    'llvmlite==0.39.1',
    'matplotlib==3.6.2',
    'more-itertools==9.0.0',
    'numba==0.56.4',
    'numpy==1.23.5',
    'packaging==21.3',
    'pandas==1.5.2',
    'Pillow==9.3.0',
    'plotly==5.11.0',
    'poliastro==0.17.0',
    'pyerfa==2.0.0.1',
    'pygeoif==0.7',
    'pyparsing==3.0.9',
    'python-dateutil==2.8.2',
    'pytz==2022.6',
    'pyvo==1.4',
    'pywin32-ctypes==0.2.0',
    'PyYAML==6.0',
    'requests==2.28.1',
    'scikit-learn==1.1.3',
    'scipy==1.9.3',
    'six==1.16.0',
    'soupsieve==2.3.2.post1',
    'tenacity==8.1.0',
    'threadpoolctl==3.1.0',
    'typing-extensions==4.4.0',
    'urllib3==1.26.13',
    'vscodedebugvisualizer==0.1.0',
    'w3lib==2.0.1',
    'webencodings==0.5.1',
    'zipp==3.11.0',
)

    
setup(
    name='rarnold_cs6840_final_project',
    versopm='0.1.0',
    author='Ryan Arnold',
    author_email='arnold.227@wright.edu',
    description=("code for final project of CS 6840: Intro to Machine Learning"
                 "Uses a SVC to distinguish overlapping satellite tracks and perform track fusion."),
    packages=find_packages(include=[
        'project_entry',
        'config',
        'data_generation',
        'shallow_learning',
        'visual',
        'cache'
    ]),
    install_requires = PACKAGE_LIST,
    long_description=open(Path(__file__).parent.resolve()/"README.md").read(),
    include_package_data=True,
    package_data={"cache":['*.pkl']},
    entry_points= {
        'console_scripts':[
            'run_cs6840_final = project_entry:run'
        ]
    }
    
)