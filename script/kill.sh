#!/bin/bash

ps -ef|grep 'path=tests/'|grep -v grep|awk '{print $2}'|xargs -r kill -9
ps -ef|grep 'path=./tests/'|grep -v grep|awk '{print $2}'|xargs -r kill -9
ps -ef|grep 'miniconda3/bin/'|grep -v grep|awk '{print $2}'|xargs -r kill -9