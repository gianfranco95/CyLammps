from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
	Extension("CyLammps",
		sources=["library.pyx"],
		libraries=["m"]
		)
]

setup(name='CyLammps',
	ext_modules=cythonize(ext_modules, annotate=True,compiler_directives={'language_level' : "3"}) )
	