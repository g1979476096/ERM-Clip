import json
import time
from tkinter import *
from PIL import Image, ImageTk
import wget
import copy
import os
import openpyxl
import pandas as pd
import numpy as np
# import paramiko
# from scp import SCPClient
from tqdm import tqdm
import datetime
import random
from tkinter.filedialog import askdirectory, askopenfilename
import shutil
# import options.options as option
# import photohash
import cv2

results, record, error_json = [], {}, 0
Weibo_21_path = '/home/yanwang_nuist/dev_workspace/prj_python/fnd-bootstrap/dataset/Weibo_21'
json_files = [f'{Weibo_21_path}/fake_release_all.json', f'{Weibo_21_path}/real_release_all.json']
folders = [f'{Weibo_21_path}/rumor_images/', f'{Weibo_21_path}/nonrumor_images/']
# json_files = ['E:/Weibo_21/fake_release_all.json','E:/Weibo_21/real_release_all.json']
# folders = ['E:/Weibo_21/rumor_images/','E:/Weibo_21/nonrumor_images/']
conduct_download = False

for i in range(2):
    json_file = json_files[i]
    f = open(json_file,encoding='utf-8')
    line = f.readline()  # id,content,comments,timestamp,piclists,label,category
    while line:
        try:
            folder = folders[i]
            images_set = set(os.listdir(folder))
            item = json.loads(line)
            record = {}
            # yan: "piclists":
            # nan
            # http://ww4.sinaimg.cn/large/7d201ad4jw1dyisb4u36uj.jpg
            # ['//s2.sinaimg.cn/thumb180/001nGfzozy6OnOITrq141']
            piclists = item["piclists"]
            image_name = ""
            if isinstance(piclists,float): piclists = []
            if not isinstance(piclists,list): piclists = [piclists]
            for full_image_name in piclists:
                if full_image_name[:2]=='//': full_image_name = "http:"+full_image_name
                short_name = full_image_name[full_image_name.rfind('/') + 1:]
                print(full_image_name)
                print(short_name)
                if short_name[-4:]!='gif':
                    if short_name not in images_set:
                        if conduct_download:
                            try:
                                wget.download(full_image_name, folder + short_name)
                                image_name = image_name + "rumor_images/" + short_name + "|"
                                print("Download ok. {}".format(full_image_name))
                            except Exception:
                                print("Download error. {}".format(full_image_name))
                        else:
                            print("Do not download. {}".format(full_image_name))
                    else:
                        image_name = image_name + "rumor_images/" + short_name + "|"
                        print("Already Downloaded. {}".format(full_image_name))
                else:
                    print("Gif Skipped. {}".format(full_image_name))

            record['source'] = ""
            record['images'] = image_name
            record['content'] = item["content"]
            record['label'] = i
            results.append(record)
            line = f.readline()
        except Exception:
            print("Load JSON error!")
            error_json += 1
    f.close()

# print(results)
df = pd.DataFrame(results)
df.to_excel(f'{Weibo_21_path}/real_datasets.xlsx')
# df.to_excel('./dataset/Weibo_21/real_datasets.xlsx')
