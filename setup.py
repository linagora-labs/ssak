import os

from setuptools import find_packages, setup

required_packages_filename = os.path.join(os.path.dirname(__file__), "requirements/requirements.txt")
if os.path.exists(required_packages_filename):
    install_requires = [l.strip() for l in open(required_packages_filename).readlines()]

version = None
license = None
with open(os.path.join(os.path.dirname(__file__), "ssak", "version.py")) as f:
    for line in f:
        if line.strip().startswith("__version__"):
            version = line.split("=")[1].strip().strip("\"'")
            if version and license:
                break
        if line.strip().startswith("__license__"):
            license = line.split("=")[1].strip().strip("\"'")
            if version and license:
                break
assert version and license

description = "Multi-lingual Automatic Speech Recognition (ASR) based on Whisper models, with accurate word timestamps, access to language detection confidence, several options for Voice Activity Detection (VAD), and more."

setup(
    name="ssak",
    py_modules=["ssak"],
    version=version,
    description=description,
    long_description=description + "\nSee https://github.com/linto-ai/ssak for more information.",
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    author="linto-ai",
    url="https://github.com/linto-ai/ssak",
    license=license,
    packages=find_packages(exclude=["tests*"]),
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "sak_infer=ssak.infer.transformers_infer:cli",
            "sak_infer_speechbrain=ssak.infer.speechbrain_infer:cli",
        ],
    },
    include_package_data=True,
    # extras_require={
    #     "full": [
    #         "soxbindings",
    #         "pypi-kenlm",
    #         "PyAudio",
    #     ],
    # },
)
