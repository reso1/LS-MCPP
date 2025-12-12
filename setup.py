from setuptools import setup


setup(
    name="lsmcpp",
    version="0.1",
    description="Large-Scale Multi-Robot Coverage Path Planning with Path Deconfliction",
    author="Jingtao Tang",
    author_email="todd.j.tang@gmail.com",
    packages=["lsmcpp.benchmark", "lsmcpp.conflict_solver", "lsmcpp", "lsmcpp.turn_minimization"],
    package_dir={
        "lsmcpp.benchmark": "benchmark",
        "lsmcpp.conflict_solver": "conflict_solver",
        "lsmcpp": "mcpp",
        "lsmcpp.turn_minimization": "turn_minimization"
    },
    install_requires=[
        "matplotlib",
        "networkx",
        "numpy",
        "PyYAML",
        "scipy",
        "gurobipy"
    ],
    
)
