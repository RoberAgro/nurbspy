#!/usr/bin/env python3
import pytest

# Define the list of tests
tests_list = ["test_basis_functions.py", "test_nurbs_curve.py", "test_nurbs_surface.py"]

# Run pytest when the python script is executed
pytest.main(tests_list)
