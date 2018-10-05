
"""
Created on October 4, 2018

@author: mae-ma
@attention: tests for continuous integration
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.0.0

#############################################################################################

This class is mainly to test if the "pytest" module is working correctly

History:
- v1.0.1: print explanation if program is executed
- v1.0.0: first init
"""


class TestFunction:
    # Important to start the class name with "Test"!
    def test_one(self):
        assert 1 == True

    def test_two(self):
        assert 1 != False


if __name__ == '__main__':
    tf = TestFunction()
    s = 'This package contains all the automated CI tests using "pytest".\nIncluding the following:'
    functions = [a for a in dir(tf) if not a.startswith('__')]
    print('–' * len(s.split('\n')[0]))
    print(s)
    for func in functions:
        print('-', func)
    print('–' * len(s.split('\n')[0]))
