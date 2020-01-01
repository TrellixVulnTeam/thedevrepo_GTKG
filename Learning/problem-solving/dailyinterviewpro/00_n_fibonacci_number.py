'''
#Apple
Find the nth Fibonacci number.

0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144...

Input: n = 3
Output: 2

Input: n = 7
Output: 13
'''

class Solution():
  def fibonacci(self, n):
    if n == 0:
        return 0
    elif n < 3:
        return 1
    else:
        return Solution().fibonacci(n-1) + Solution().fibonacci(n-2)

n = 9
print(Solution().fibonacci(n))
# 34