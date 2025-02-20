"""
This file serves as a mock API for the SedTRAILS project. 
It is used to demonstrate the use of the `autodoc` extension in Sphinx.
"""

def sum(x: int, y: int) -> int:
    """
    Add two numbers together.

    Parameters
    ----------
    x : int
        The first number.
    y : int
        The second number.

    Returns
    -------
    int
        The sum of the two numbers.

    Examples
    --------
    You can include examples in RST
    
    .. code-block:: python

        from sedtrails import sum
        sum(1, 2)
        # Output: 3
    
    """
    return x + y


class Calculator:
    """
    A simple calculator class.
    """
    def __init__(self):
        pass
    def add(self, x: int, y: int) -> int:
        """
        Add two numbers together.

        Parameters
        ----------
        x : int
            The first number.
        y : int
            The second number.

        Returns
        -------
        int
            The sum of the two numbers.
        """
        return x + y
    
    def subtract(self, x: int, y: int) -> int:
        """
        Subtract two numbers.

        Parameters
        ----------
        x : int
            The first number.
        y : int
            The second number.

        Returns
        -------
        int
            The difference between the two numbers.
        """
        return x - y