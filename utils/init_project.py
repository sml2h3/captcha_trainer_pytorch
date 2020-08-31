#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time :    2020/8/23 18:09
@Author:   sml2h3
@File:     init_project
@Software: PyCharm
"""
from config import Config
import os
import shutil


class InitProject(object):
    def __init__(self, project_name):
        self.project_name = project_name
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "projects")

    def make(self, force: bool = False):
        project_path = os.path.join(self.base_path, self.project_name)
        if os.path.exists(project_path):
            if not force:
                while True:
                    force_delete = input(
                        "Found {} exists! Are you sure you want to delete this item? [Y/n]".format(self.project_name))
                    if force_delete.lower() == "n":
                        exit(0)
                    elif force_delete.lower() != "y":
                        continue
                    else:
                        break
            shutil.rmtree(project_path)
        os.mkdir(project_path)
        config = Config(self.project_name)
        config.make_config()
        os.mkdir(os.path.join(project_path, "graphs"))
        os.mkdir(os.path.join(project_path, "models"))
        os.mkdir(os.path.join(project_path, "datas"))
        print("Project named {} has been created!".format(self.project_name))


if __name__ == '__main__':
    InitProject("a").make(True)
