from setuptools import setup

APP = ['gui.py']
OPTIONS = {
    'argv_emulation': True,
    'packages': ['numpy'],  # Include numpy package
    'excludes': ['numpy.typing.tests','fcntl'],  # Exclude the problematic test modules
}

setup(
    app=APP,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)

