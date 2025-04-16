"""
This is an example of how to create a custom exception in Python.

'Programs may name their own exceptions by creating a new exception class (see Classes for more about Python classes).
Exceptions should typically be derived from the Exception class, either directly or indirectly.

Exception classes can be defined which do anything any other class can do, but are usually kept simple,
often only offering a number of attributes that allow information about the error to be extracted by handlers
for the exception.

Most exceptions are defined with names that end in “Error”, similar to the naming of the standard exceptions.

Many standard modules define their own exceptions to report errors that may occur in functions they define.
' -Python documentation

"""


# Custom exceptions inherit from the built-in Exception class.
class ValueTooLargeError(Exception):
    """
    Exception raised when the input value is too large.
    """

    pass  # This is enought to define the exception class. However, you can do more. See below.


class ValueTooSmallError(Exception):
    """
    Exception raised when the input value is too small.
    """

    def __init__(self, value, limit):  # this add behavior to the exception.
        # In this example, the exception always require a value and a limit.
        """
        Initialize the exception with a value.
        """
        self.value = value
        self.limit = limit
        super().__init__(f'Value {value} is too small. It should be at least {limit}.')


# Example of how to use the custom exceptions
if __name__ == '__main__':
    try:
        x = 10
        if x > 5:
            raise ValueTooLargeError('Value is too large')
    except ValueTooLargeError as e:
        print(e)

    try:
        x = 1
        if x < 5:
            raise ValueTooSmallError(x, 5)
    except ValueTooSmallError as e:
        raise ValueTooSmallError(e.value, e.limit) from e  # the use of 'from e' will show the original
        # exception in the traceback. This is useful to understand the context of the error.
