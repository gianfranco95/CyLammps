from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
	Extension("ANALISI",
		sources=["analisi.pyx"],
		libraries=["m"]
		)
]

setup(name='ANALISI',
	ext_modules=cythonize(ext_modules, annotate=True,compiler_directives={'language_level' : "3"}) )



#compilare con   " python setup.py build_ext --inplace "
