from setuptools import setup

setup(name='PixelSorter',
      version='0.1',
      description='Sort images by pixel properties',
      url='https://github.com/squaregoldfish/PixelSorter',
      author='Unathi-Skosana, Steve Jones',
      author_email='steve@squaregoldfish.co.uk',
      license='MIT',
      install_requires=['opencv-python', 'numpy'],
      packages=['PixelSorter'],
      zip_safe=False)