import setuptools

setuptools.setup(
    name='biasprobe',
    version=eval(open('biasprobe/_version.py').read().strip().split('=')[1]),
    author='Raphael Tang',
    license='MIT',
    url='https://github.com/castorini/sortprobe',
    author_email='r33tang@uwaterloo.ca',
    description='Bias probes for interpreting biases in LLMs',
    install_requires=open('requirements.txt').read().strip().splitlines(),
    packages=setuptools.find_packages(),
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
        ]
    }
)