#encoding utf-8
import math
import sys
from collections import defaultdict

def distance(a, b, gauss=True):
    """
        Euclidean distance or gauss kernel
    """
    dim = len(a)
    
    _sum = 0.0
    for dimension in xrange(dim):
        difference_sq = (a[dimension] - b[dimension]) ** 2
        _sum += difference_sq

    if gauss :
        dis = 1 - math.exp(_sum * -1/2)
    else :
        dis = math.sqrt(_sum)

    return dis

def load_file(file_name) :
    """
        Load dataset, for iris dataset
    """
    data = []
    label = []
    with open(file_name) as fp :
        for line in fp :
            line = line.strip()
            if len(line) < 1 :
                continue

            tokens = line.split(',')
            if len(tokens) < 2 :
                continue
            #this last column is label
            data.append([float(x) for x in tokens[:-1]])
            label.append(tokens[-1])

    return data,label

def pair_dis(data, gauss=True) :
    """
        Calculate all pair distances between records
    """
    distances = {}

    length = len(data)
    max_dis = 0.0
    min_dis = sys.float_info.max

    for i in xrange(length - 1) :
        for j in xrange(i + 1, length) :
            dis = distance(data[i], data[j], gauss)
            distances[frozenset((i,j))] = dis

            if dis > max_dis :
                max_dis = dis

            if dis < min_dis :
                min_dis = dis

    return distances, max_dis, min_dis

def select_dc(allnum, distances, max_dis, min_dis) :
    """
        Auto-tune dc
    """
    dc = (max_dis + min_dis) / 2.0

    for i in xrange(100) :
        sumnum = 0
        for k, v in distances.iteritems() :
            if v < dc : 
                sumnum += 2

        mean_dc = float(sumnum) / (allnum ** 2)

        if mean_dc > 0.01 and mean_dc < 0.02 :
            break

        if mean_dc > 0.02 :
            max_dis = dc
            dc = (min_dis + dc) / 2
        elif mean_dc < 0.01 :
            min_dis = dc
            dc = (max_dis + dc) / 2

    return dc, mean_dc

def density(allnum, distances, density_distance_threshold, max_dis) :
    """
        generate candidate data, for each record generate the following data:

        'q' : record id
        'm' : local density (rho)
        'd' : the minimum distance to the neighbour of higher density (delta)
        's' : it's nearest neighbour of higher density
    """
    dens = defaultdict(lambda : 0)

    for k, v in distances.iteritems() :
        if v < density_distance_threshold : 
            for i in k :
                dens[i] += 1

    for i in xrange(allnum) :
        dens[i] += 0

    sortdens = sorted(dens.items(), key=lambda v : v[1], reverse=True)
    length = len(sortdens)

    candidate = [{'q' : sortdens[0][0], 'm' : sortdens[0][1], 'd' : max_dis, 's' : -1}]

    for i in range(1, length) :
        min_distance = sys.float_info.max
        min_record = 0

        for j in range(i) :
            sub = frozenset((sortdens[i][0], sortdens[j][0]))
            dis = distances[sub]

            if dis < min_distance :
                min_distance = dis
                min_record = sortdens[j][0]

        one = {'q' : sortdens[i][0], 'm' : sortdens[i][1], 'd' : min_distance, 's' : min_record}
        candidate.append(one)

    return candidate

def clustering(candidate_file, center_density_threshold, center_distance_threshold) :
    """
        clustering
    """
    center = {}
    cluster = {}

    for item, density, distance, nearest_item in load_candidate_file(candidate_file) :

        if density >= center_density_threshold and distance >= center_distance_threshold :
            center[item] = item
            cluster[item] = item
        else :
            if nearest_item in cluster :
                cluster[item] = cluster[nearest_item]
            else :
                cluster[item] = -1

    return center, cluster

def dump_candidate_file(candidate, out_file) :
    """
        dump_candidate_file, make for tuning params
    """
    fw = open(out_file, 'w')
    for one in candidate :
        fw.write("%d\t%d\t%.4f\t%d\n" % (one['q'], one['m'], one['d'], one['s']))
    fw.close()

def load_candidate_file(cand_file) :
    """
        load_candidate_file, make for tuning params
    """
    with open(cand_file) as fp :
        for line in fp :
            line = line.strip()
            if len(line) < 1 :
                continue

            tokens = line.split('\t')
            if len(tokens) < 4 :
                continue

            item = int(tokens[0])
            density = float(tokens[1])
            distance = float(tokens[2])
            nearest_item = int(tokens[3])

            yield item, density, distance, nearest_item

def confidence(cluster, distances, dc) :
    """
        Calculate the confidence for each record
        confidence = s / n
        n : the num of records with a distance dc
        s : the num of records with a distance dc and having the same cluster
    """
    dens = defaultdict(lambda : {'n' : 0, 's' : 0})

    for k, v in distances.iteritems() :
        if v < dc : 
            issame = False
            templist = list(k)
            if templist[0] in cluster and templist[1] in cluster and cluster[templist[0]] == cluster[templist[1]] :
                issame = True

            for i in k :
                dens[i]['n'] += 1
                if issame :
                    dens[i]['s'] += 1

    reliable = {}
    for k,  v in dens.iteritems() :
        reliable[k] = float(v['s']) / v['n']

    return reliable

def evaluate(center, cluster, label, reliable, file_name) :
    """
        evaluate precision and recall
    """
    allcount = 0
    norecall = 0
    correct = 0

    fw = open(file_name, 'w')

    for k , v in cluster.iteritems() :
        
        cluster_str = ""
        if v == -1 :
            norecall += 1
            cluster_str += "NORECALL\t"
        elif cmp(label[v], label[k]) == 0 :
            correct += 1
            cluster_str += "CORRECT\t"
        else :
            cluster_str += "ERROR\t"

        cluster_str += "%d\t%s\t%s\t" % (k, label[v], label[k])
        if k in reliable :
            cluster_str += "%.4f" % reliable[k]
        else :
            cluster_str += "-"

        allcount += 1
        fw.write(cluster_str + '\n')

    fw.close()

    print 'allcount: ', allcount
    print 'correct : ', correct
    print 'precision: ', float(correct) / (allcount - norecall)
    print 'recall: ', float(allcount - norecall) / allcount

if __name__ == '__main__':
    
    data,label = load_file('data/iris.data')

    distances, max_dis, min_dis = pair_dis(data, True)

    dc, mean_dc = select_dc(len(data), distances, max_dis, min_dis)
    print "Max distance: %.4f Min distance: %.4f DC: %.4f" % (max_dis, min_dis, dc)

    candidate = density(len(data), distances, dc, max_dis)

    dump_candidate_file(candidate, 'debug_iris_candidate')

    center, cluster = clustering('debug_iris_candidate', 1, 0.38)
    print "CENTER:", center

    reliable = confidence(cluster, distances, dc)

    evaluate(center, cluster, label, reliable, 'debug_iris_cluster')
