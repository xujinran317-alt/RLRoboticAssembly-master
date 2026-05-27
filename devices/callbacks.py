#! usr/bin/env python
"""

callbacks.py - 回调函数处理工具

提供两个功能：
  1. handle()  - 安全地执行回调函数
  2. validate() - 验证函数是否符合要求（参数/返回值约束）
"""

import inspect


def handle(func, *args, **kwargs):
    """
    安全地调用回调函数
    - 如果 func 不为 None，就调用它
    - 如果 func 不接受参数，会尝试无参调用
    """
    if func is not None:
        try:
            return func(*args, **kwargs)
        except TypeError:
            return func()
    return


def validate(func, allow_args=False, allow_return=False, require_self=False):
    """
    验证函数是否符合框架的要求
    allow_args:    是否允许函数有参数
    allow_return:  是否允许函数有返回值
    require_self:  是否要求第一个参数是 self
    """
    assert callable(func) or func is None, '函数必须可调用或为 None'
    if callable(func):
        if not allow_args and not require_self:
            signature = inspect.signature(func)
            assert len(signature.parameters) == 0, '函数不能有输入参数'
        if not allow_return:
            lines, _ = inspect.getsourcelines(func)
            assert not lines[-1].lstrip().startswith('return'), '函数不能有 return 语句'
        if require_self:
            signature = inspect.signature(func)
            assert next(iter(signature.parameters)) == 'self', '函数的第一个参数必须是 self'

