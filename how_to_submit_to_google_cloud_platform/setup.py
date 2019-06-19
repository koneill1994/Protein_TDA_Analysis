from setuptools import setup, find_packages

setup(name='trainer',
      version='1.0',
      packages=find_packages(),
      description='IEEE ICMLA 2019 Challenge - Protein Inter-Residue Distance Prediction',
      author='Badri Adhikari',
      author_email='badri.com.np@gmail.com',
      license='MIT',
      install_requires=['keras', 'h5py'],
      zip_safe=False)
