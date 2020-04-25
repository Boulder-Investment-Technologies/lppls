import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='lppls',
      version='0.1.13',
      description='A Python module for fitting the LPPLS model to data.',
      packages=['lppls'],
      author='Josh Nielsen',
      author_email='josh@boulderinvestment.tech',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/Boulder-Investment-Technologies/lppls',
      python_requires='>=3.6',
      zip_safe=False)