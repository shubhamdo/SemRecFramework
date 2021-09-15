from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='jobrec',
      version='0.0.1',  # 0.0.x Indicating Unstable version
      description='Project Description',
      py_modules=['DocumentationReference'],  # Code that should get installed
      packages=find_packages(exclude=('build', 'dist', 'docs')),
      # package_dir={'': 'scripts'},
      classifiers=[
          "Programming Language :: Python 3",
          "Programming Language :: Python 3.6",
          "Programming Language :: Python 3.7",
          "Programming Language :: Python 3.8",
          "Operating System :: OS Independent"
      ],
      long_description=long_description,
      install_requires=[            # Production - Compatible Version i.e. Can be compatible with newer versions > sign

      ],
      extras_require={              # Requires to be more specific for the development purpose
          "dev": [
              "pytest >= 3.7",
          ]
      }
      )
