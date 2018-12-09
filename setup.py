from setuptools import find_packages
from setuptools import setup

setup(
    name='vp',
    version='0.0.1',
    description='Vanishing point toolbox.',
    license='MIT',
    keywords='vanishing points horizon ransac j-linkage computer vision',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 2 - Pre-Alpha'
        'Environment :: Other Environment',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    install_requires=['opencv-python', 'numpy', 'ransac'],
    dependency_links=[
        "git+ssh://git@github.com/romack77/ransac.git@173ef6f#egg=ransac"
    ],
    packages=find_packages(exclude=['notebooks']),
    test_suite='vp',
)
