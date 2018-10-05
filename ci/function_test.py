
"""
Created on October 4, 2018

@author: mae-ma
@attention: tests for continuous integration
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.0.0

#############################################################################################

This class is mainly to test if the "pytest" module is working correctly

History:
- v1.0.0: first init
"""

# Important to start the class name with "Test"!
class TestFunction:
    def test_one(self):
        assert 1 == True

    def test_two(self):
        assert 1 != False
