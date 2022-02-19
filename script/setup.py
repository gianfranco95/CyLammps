from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
	Extension("STRESS",
		sources=["stress_opt.pyx"],
		libraries=["m"]
		)
]

setup(name='STRESS',
	ext_modules=cythonize(ext_modules, annotate=True,compiler_directives={'language_level' : "3"}) )
