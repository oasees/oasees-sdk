def code_cell(source, readonly=False, editable=True,tags=[]):
    """Create a code cell with common metadata patterns"""
    if readonly:
        metadata = {
            "editable": False,
            "deletable": False, 
            "tags": tags
        }
    elif not editable:
        metadata = {
            "editable": False,
            "deletable": True,
            "tags": tags
        }
    else:
        metadata = {
            "editable": True,
            "deletable": True,
            "tags": tags
        }
    
    if isinstance(source, str):
        import textwrap
        source = textwrap.dedent(source).strip()
        source = [line + '\n' for line in source.split('\n')]
    else:
        source = [line + '\n' if not line.endswith('\n') else line for line in source]
    
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": source
    }

def markdown_cell(source):
    """Create a markdown cell"""
    if isinstance(source, str):
        import textwrap
        source = textwrap.dedent(source).strip()
        source = [line + '\n' for line in source.split('\n')]
    else:
        source = [line + '\n' if not line.endswith('\n') else line for line in source]
    
    return {
        "cell_type": "markdown", 
        "metadata": {},
        "source": source
    }

def notebook(cells):
    """Create a complete notebook"""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }