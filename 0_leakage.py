# -*- coding: utf-8 -*-
"""
Created on Mon May 23 08:14:56 2016

@author: subhajit
"""

# coding: utf-8
__author__ = 'Ravi: https://kaggle.com/company'

import datetime
from heapq import nlargest
from operator import itemgetter
import os

os.chdir('D:\Data Science Competitions\Kaggle\Expedia Hotel Recommendations\codes')


def prepare_arrays_match():
    f = open("../input/train.csv", "r")
    f.readline()
    
    best_hotels_od_ulc = dict()
    best_hotels_uid_miss = dict()
    best_s00 = dict()
    best_s01 = dict()
    total = 0

    # Calc counts
    while 1:
        line = f.readline().strip()
        total += 1

        if total % 2000000 == 0:
            print('Read {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        book_year = int(arr[0][:4])
        book_month = int(arr[0][5:7])
        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        user_id = arr[7]
        srch_destination_id = arr[16]
        hotel_country = arr[21]
        hotel_market = arr[22]
        is_booking = float(arr[18])
        hotel_cluster = arr[23]

        append_0 = ((book_year - 2012)*12 + (book_month - 12))
        append_1 = append_0 * append_0 * (3 + 17.60*is_booking)
        append_2 = 3 + 5.56*is_booking

        if user_location_city != '' and orig_destination_distance != '' and user_id !='' and srch_destination_id != '' and hotel_country != '':
            s00 = hash(str(user_id)+':'+str(user_location_city)+':'+str(srch_destination_id)+':'+str(hotel_country)+':'+str(hotel_market))
            if s00 in best_s00:
                if hotel_cluster in best_s00[s00]:
                    best_s00[s00][hotel_cluster] += append_1
                else:
                    best_s00[s00][hotel_cluster] = append_1
            else:
                best_s00[s00] = dict()
                best_s00[s00][hotel_cluster] = append_1

        if user_location_city != '' and orig_destination_distance != '' and user_id !='' and srch_destination_id != '':
            s01 = hash(str(user_id)+':'+str(srch_destination_id)+':'+str(hotel_country)+':'+str(hotel_market))
            if s01 in best_s01:
                if hotel_cluster in best_s01[s01]:
                    best_s01[s01][hotel_cluster] += append_1
                else:
                    best_s01[s01][hotel_cluster] = append_1
            else:
                best_s01[s01] = dict()
                best_s01[s01][hotel_cluster] = append_1


        if user_location_city != '' and orig_destination_distance == '' and user_id !='' and srch_destination_id != '' and hotel_country != '':
            s0 = hash(str(user_id)+':'+str(user_location_city)+':'+str(srch_destination_id)+':'+str(hotel_country)+':'+str(hotel_market))
            if s0 in best_hotels_uid_miss:
                if hotel_cluster in best_hotels_uid_miss[s0]:
                    best_hotels_uid_miss[s0][hotel_cluster] += append_1
                else:
                    best_hotels_uid_miss[s0][hotel_cluster] = append_1
            else:
                best_hotels_uid_miss[s0] = dict()
                best_hotels_uid_miss[s0][hotel_cluster] = append_1

        if user_location_city != '' and orig_destination_distance != '':
            s1 = hash(str(user_location_city)+':'+str(orig_destination_distance))

            if s1 in best_hotels_od_ulc:
                if hotel_cluster in best_hotels_od_ulc[s1]:
                    best_hotels_od_ulc[s1][hotel_cluster] += append_0
                else:
                    best_hotels_od_ulc[s1][hotel_cluster] = append_0
            else:
                best_hotels_od_ulc[s1] = dict()
                best_hotels_od_ulc[s1][hotel_cluster] = append_0

    f.close()
    return best_s00,best_s01, best_hotels_od_ulc, best_hotels_uid_miss


def gen_submission(best_s00, best_s01, best_hotels_od_ulc, best_hotels_uid_miss):
    now = datetime.datetime.now()
    path = '../output/match_pred.csv'
    out = open(path, "w")
    f = open("../input/test.csv", "r")
    f.readline()
    total = 0
    total0 = 0
    total00 = 0
    total1 = 0
    total2 = 0
    total3 = 0
    total4 = 0
    out.write("id,hotel_cluster\n")
    
    while 1:
        line = f.readline().strip()
        total += 1

        if total % 100000 == 0:
            print('Write {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        id = arr[0]
        user_location_city = arr[6]
        orig_destination_distance = arr[7]
        user_id = arr[8]
        srch_destination_id = arr[17]
        hotel_country = arr[20]
        hotel_market = arr[21]

        out.write(str(id) + ',')
        filled = []

        s1 = hash(str(user_location_city)+':'+str(orig_destination_distance))
        if s1 in best_hotels_od_ulc:
            d = best_hotels_od_ulc[s1]
            topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                total1 += 1

        if orig_destination_distance == '':
            s0 = hash(str(user_id)+':'+str(user_location_city)+':'+str(srch_destination_id)+':'+str(hotel_country)+':'+str(hotel_market))
            if s0 in best_hotels_uid_miss:
                d = best_hotels_uid_miss[s0]
                topitems = nlargest(4, sorted(d.items()), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])
                    total0 += 1

        s00 = hash(str(user_id)+':'+str(user_location_city)+':'+str(srch_destination_id)+':'+str(hotel_country)+':'+str(hotel_market))
        s01 = hash(str(user_id)+':'+str(srch_destination_id)+':'+str(hotel_country)+':'+str(hotel_market))
        if s01 in best_s01 and s00 not in best_s00:
            d = best_s01[s01]
            topitems = nlargest(4, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                total00 += 1

        out.write("\n")
    out.close()
    print('Total 1: {} ...'.format(total1))
    print('Total 0: {} ...'.format(total0))
    print('Total 00: {} ...'.format(total00))
    print('Total 2: {} ...'.format(total2))
    print('Total 3: {} ...'.format(total3))
    print('Total 4: {} ...'.format(total4))


best_s00,best_s01, best_hotels_od_ulc, best_hotels_uid_miss = prepare_arrays_match()
gen_submission(best_s00, best_s01, best_hotels_od_ulc, best_hotels_uid_miss)