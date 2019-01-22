from setuptools import setup, find_packages

setup(
    name = 'abfs',
    author = 'Christian Di Lorenzo',
    author_email = 'rcddeveloper@icloud.com',
    setup_requires = ['pytest-runner'],
    install_requires = [
        # 'scikit-image',
        # 'toolz',
        # 'osgeo',
        # 'numpy',
        # 'spacenetutilities',
        # 'matplotlib'
    ],
    tests_require = ['pytest', 'pytest-watch'],
    packages = find_packages()
)
