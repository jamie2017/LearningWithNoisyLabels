#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Project: "A Study of Learning with Noisy Labels"

Copyright (c) 2017 Jianmei Ye

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

#  Task
#  Profiling method call and memory cost

from SequentialRun import single_run
from memory_profiler import profile
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

def method_call_profiling():
    graphviz = GraphvizOutput()
    graphviz.output_file = 'src/ProfilerOutput/method_call.png'
    with PyCallGraph(output=graphviz):
        single_run(True)


@profile
def memory_profiling():
    '''
    Commend line 54 and uncommend line 55
    Then, in terminal, under dir of "src" ,run following commend
    mprof run Profiling.py
    mprof plot
    '''
    single_run(True)

if __name__ == "__main__":
    method_call_profiling()
    # memory_profiling()