#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
from os.path import dirname, realpath

dir_path = dirname(realpath(__file__))
project_path = realpath(dir_path + '/..')

libs_dir_path = project_path + '/deps'

# Adding where to find libraries and dependencies
sys.path.append(libs_dir_path)
