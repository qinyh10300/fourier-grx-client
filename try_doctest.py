def add(a, b):
    """
    返回两个数的和。

    >>> add(2, 3)
    5
    >>> add(-1, 1)
    0
    """
    return a + b

if __name__ == "__main__":
    '''
    如果所有测试用例都通过，脚本将静默退出。如果有任何测试用例失败，doctest 将输出失败的详细信息。
    '''
    import doctest
    doctest.testmod()