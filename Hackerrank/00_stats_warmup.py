'''
problem: https://www.hackerrank.com/challenges/stat-warmup/problem

Sample input:
10
64630 11735 14216 99233 14470 4978 73429 38120 51135 67060
'''

import math
from collections import OrderedDict

def get_mean(num_list):
    return round(sum(num_list)/len(num_list), 1)

def get_median(num_list):
    num_list = sorted(num_list)
    n = len(num_list)
    
    if n % 2 == 0:
        return sum(num_list[int(n/2)-1:int(n/2)+1])/2
    else:
        return num_list[int(n/2)-1]

def get_mode(num_list):
    num_list = sorted(num_list)
    num_set = set(num_list)
    mode_dict = OrderedDict()
    for i,v in enumerate(num_list):
        mode_dict[v] = 0
        if i != len(num_list)-1 and v == num_list[i+1]:
            mode_dict[v] += 1
    
    return int(max(mode_dict, key=mode_dict.get))

def get_sd(num_list):
    mean = get_mean(num_list)
    sd = math.sqrt(sum([(x - mean)**2 for x in num_list]) / len(num_list))
    return round(sd, 1)

def get_CIs(num_list, t_val):
    s = get_sd(num_list)
    root_n = math.sqrt(len(num_list))
    
    std_error = s/root_n
    margin_error = t_val * std_error
        
    x_bar = get_mean(num_list)
    return (round(x_bar - margin_error, 1), round(x_bar + margin_error, 1))

t_val = 1.96 # 2.262 for 95% confidence
N = input()
num_list = list(map(float,input().split(' ')))

lb, ub = get_CIs(num_list, t_val)

print("{}\n{}\n{}\n{}\n{} {}".format(get_mean(num_list), get_median(num_list), get_mode(num_list), get_sd(num_list), lb, ub))