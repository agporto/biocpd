from setuptools import setup


def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()


setup(name='biocpd',
      version='0.1.1',
      description='Coherent Point Drift variants (rigid, affine, deformable, PCA/SSM) in NumPy/SciPy',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/your-org/biocpd',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12',
          'Topic :: Scientific/Engineering',
      ],
      keywords='point cloud registration CPD',
      author='Arthur Porto',
      author_email='agporto@gmail.com',
      license='MIT',
      packages=['biocpd'],
      python_requires='>=3.9',
      install_requires=['numpy', 'scipy', 'scikit-learn', 'matplotlib'],
      zip_safe=False)
