# Setup script for IBIS repo
from setuptools import setup, find_packages
from subprocess import Popen, PIPE

version = "1.0.0"
sha = None
git_describe_process = Popen(
    ("git",
     "describe",
     "--tags"),
    stdout=PIPE,
    stderr=PIPE)
try:
    out, _ = git_describe_process.communicate()
    version = out.decode("utf-8")
    sp = version.split("-")
    version = sp[0]
    # Clean tag?
    if len(sp) != 0:
        commits = sp[1]
        sha = sp[2]
        version += "."+commits
    else:
        sha = None
except Exception:
    pass

# Provide better description
description="IBIS"
if sha is not None:
    description += " (sha: {})".format(sha)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name="ibis",
      version=version,
      description=description,
      url="https://github.com/LLNL/IBIS",
      author="Sarah El-Jurf",
      author_email="eljurf1@llnl.gov",
      long_description=long_description,
      long_description_content_type="text/markdown",
      license="BSD 3-Clause",
      packages=find_packages(),
      zip_safe=False,
      install_requires=[
            "numpy",#>=1.15,<1.19",
            "scikit-learn",
            "scipy",
            "matplotlib",
      ],
      classifiers=[
            "Programming Language :: Python :: 3",
            #"Programming Language :: Python :: 2.7",
            "Operating System :: OS Independent",
      ],
      python_requires=">=3.6, >=2.7.16, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*",
      )
# Only when logo is made
# Popen(("scripts/render_logos.py",)).communicate()
