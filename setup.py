import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='lppls',
      version='0.6.18',
      description='A Python module for fitting the LPPLS model to data.',
      packages=['lppls'],
      author='Josh Nielsen',
      author_email='josh@boulderinvestment.tech',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/Boulder-Investment-Technologies/lppls',
      python_requires='>=3.7',
      install_requires=[
          'pandas',
          'matplotlib',
          'scipy',
          'xarray',
          'cma',
          'tqdm',
          'numba'
      ],
      zip_safe=False,
      include_package_data=True,
      package_data={'': ['data/*.csv']},
)
