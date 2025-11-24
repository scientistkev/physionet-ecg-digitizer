"""
Text file utilities for loading, saving, and parsing text files.
"""


# Load a text file as a string.
def load_text(filename):
    with open(filename, 'r') as f:
        string = f.read()
    return string


# Save a string as a text file.
def save_text(filename, string):
    with open(filename, 'w') as f:
        f.write(string)


# Get a variable from a string.
def get_variable(string, variable_name):
    variable = ''
    has_variable = False
    for l in string.split('\n'):
        if l.startswith(variable_name):
            variable = l[len(variable_name):].strip()
            has_variable = True
    return variable, has_variable


# Get variables from a string.
def get_variables(string, variable_name, sep=','):
    variables = list()
    has_variable = False
    for l in string.split('\n'):
        if l.startswith(variable_name):
            variables += [variable.strip() for variable in l[len(variable_name):].strip().split(sep)]
            has_variable = True
    return variables, has_variable

