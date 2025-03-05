# Code Review Checklist

When reviewing code contributions keep in mind the following checklist. Not all items are relevat for every case. Please contact the SedTRAILS team if you have question about this list or items not consideres on this list.

## Requirements

   - [ ] Have the requirements been met?

## Code Formatting

    - [ ] Is the code formatted correctly?
    - [ ] Are docstrings using Numpy style?
    - [ ] Unecessary whitespace removed?

Best Practices

    - [ ] Follow Single Responsibility principle?
    - [ ] Are different errors handled correctly?
    - [ ] Are errors and warnings logged?
    Magic values avoided?
    - [ ] No unnecessary comments?
    - [ ] Minimal nesting used?

Maintainability

    - [ ] Is the code easy to read?
    - [ ] Is the code not repeated (DRY Principle)?
    - [ ] Is the code method/class not too long?

Architecture

    - [ ] Are separations of concerned followed?
    - [ ] Relevant Parameters are configurable?

Testing

    - [ ] Do unit tests pass?
    - [ ] Have edge cases been tested?
    - [ ] Are invalid inputs validated?
    - [ ] Are inputs sanitised?

Documentation

    - [ ] Is there sufficient documentation?
    - [ ] Is the ReadMe.md file up to date?

> Adapted from: https://www.codereviewchecklist.com