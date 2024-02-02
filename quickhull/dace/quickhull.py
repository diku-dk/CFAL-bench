import numpy as np
import os

def define_line(p1, p2):
    """Returns the line defined by two points."""
    a = p1[1] - p2[1]
    b = p2[0] - p1[0]
    c = p1[0]*p2[1] - p2[0]*p1[1]
    return a, b, c


def distance_point_line(p, l):
    """Returns the distance between a point and a line."""
    a, b, c = l
    return (a*p[0] + b*p[1] + c) / np.sqrt(a**2 + b**2)


def distance_point_line_vec(p, l):
    """Returns the distances between a set of points and a line."""
    a, b, c = l
    return (a*p[:, 0] + b*p[:, 1] + c) / np.sqrt(a**2 + b**2)


def findDistance(a: list, b: list, p: list):
    #rewriting coordinates for simply geometric syntax
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    px, py = p[0], p[1]
    d = 0
    d = (abs(((bx - ax) * (ay - py)) - ((ax - px) * (by - ay)))) / np.sqrt((pow((bx - ax), 2)) + (pow((by - ay), 2)))
    return d


def isLeft(a: list, b: list, c: list) -> bool:
    #rewriting coordinates for simply geometric syntax
    ax, ay, bx, by, cx, cy = a[0], a[1], b[0], b[1], c[0], c[1]

    #we will take point a and point b and do the cross product of these points
    z = ((bx - ax) * (cy - ay)) - ((cx - ax) * (by - ay))

    if z > 0:
        return True
    else:
        return False


def quickhull_2d(points: np.ndarray):
    """Returns the convex hull of a set of points in 2D."""

    L = np.copy(points)
    L_label = np.zeros(len(L), dtype=np.int64)
    L_dist = np.empty(len(L), dtype=np.float64)
    R = []
    
    # A_idx = np.argmin(points[:, 0])
    # B_idx = np.argmax(points[:, 0])
    pointList = points.tolist()
    A, B = min(pointList), max(pointList)
    A_idx = pointList.index(A)
    B_idx = pointList.index(B)
    print(L[A_idx], L[B_idx])

    R.append(L[A_idx].tolist())
    R.append(L[B_idx].tolist())

    L = np.delete(L, [A_idx, B_idx], axis=0)
    L_label = np.delete(L_label, [A_idx, B_idx], axis=0)
    L_dist = np.delete(L_dist, [A_idx, B_idx], axis=0)

    partitions = [(0, len(L)), (len(L), 2 * len(L))]
    len_L = len(L)
    bounds = [(A, B), (B, A)]
    # L = np.reshape(np.concatenate((L, L), axis=0), (2, len_L, 2))
    # print(partitions)
    print(bounds)
    print()

    prev_len_R = 0
    while len(R) > prev_len_R:
        # print(R)
        prev_len_R = len(R)

        new_L = []
        new_partitions = []
        new_bounds = []

        total_idx = 0
        for i in range(len(partitions)):
        # for i in range(len(bounds)):

            fidx, lidx = partitions[i]
            if lidx - fidx <= 1:
                continue

            partition = L[fidx:lidx]

            X, Y = bounds[i]

            line = define_line(X, Y)
            dist = distance_point_line_vec(partition, line)
            tmp = partition[dist > 0]
            max_dist = np.max(dist)

            if max_dist > 0:
                idx = np.argmax(dist)
                new_point = partition[idx].tolist()
                R.append(new_point)

                new_L.extend(tmp)
                new_partitions.append((total_idx, total_idx + len(tmp)))
                new_bounds.append((X, new_point))
                total_idx += len(tmp)
                new_L.extend(tmp)
                new_partitions.append((total_idx, total_idx + len(tmp)))
                new_bounds.append((new_point, Y))
                total_idx += len(tmp)
        
        L = np.array(new_L)
        partitions = new_partitions
        bounds = new_bounds
        # print(partitions)
        # print(bounds)
        print()

        
        # # Segmented scan
        # # M = np.maximum.reduceat(L_dist, partitions)[::2]
        # M = []
        # for i in range(len(partitions) - 1):
        #     partition = L_dist[partitions[i]:partitions[i+1]]
        #     M.extend(np.maximum.accumulate(partition))
        # # Update R
        # new_R_idx = []
        # new_R_labels = {}
        # for i in range(len(partitions) - 1):
        #     partition = M[partitions[i]:partitions[i+1]]
        #     if partition[-1] > 0:
        #         pivot = partitions[i] + np.argmax(partition)
        #         new_R_idx.append(pivot)

        #     else:
        #         new_R_idx.append(-1)
        # for idx in new_R_idx:
        #     if idx == -1:
        #         continue
        #     C = L[idx].tolist()
        #     C_label = L_label[idx]
        #     R.insert(C_label + 1, C)  # ???
        #     new_R_labels[C_label] = C
        # L = np.delete(L, new_R_idx, axis=0)
        # L_label = np.delete(L_label, new_R_idx, axis=0)
        # L_dist = np.delete(L_dist, new_R_idx, axis=0)
        
        # # Remove negative distance points
        # mark = L_dist >= 0
        # L = L[mark]
        # L_label = L_label[mark]
        # L_dist = L_dist[mark]

        # # Update partitions
        # new_L = []
        # new_L_label = []
        # new_partitions = [0]
        # for label in range(prev_len_R - 1):
        #     points = L[L_label == label]
        #     if len(points) == 0:
        #         continue
        #     pivot = new_R_labels[label]
        #     pivot_idx = R.index(pivot)
        #     left, right = [], []
        #     for p in points:
        #         if isLeft(R[pivot_idx - 1], R[pivot_idx], p.tolist()):  # Fix
        #             left.append(p)
        #         else:
        #             right.append(p)
        #     new_L.extend(left)
        #     new_L_label.extend([pivot_idx - 1 for _ in range(len(left))])
        #     if len(left) > 0:
        #         new_partitions.append(len(new_L))
        #     new_L.extend(right)
        #     new_L_label.extend([pivot_idx for _ in range(len(right))])
        #     if len(right) > 0:
        #         new_partitions.append(len(new_L))

        # L = np.array(new_L)
        # L_label = np.array(new_L_label, dtype=np.int64)
        # partitions = new_partitions
    
    return R
    

if __name__ == "__main__":
    # points = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5], [0.5, -0.5], [0, -1], [1, -1]])

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file = os.path.join(dir_path, '..', 'input', '1M_rectangle_16384.dat')
    # file = os.path.join(dir_path, '..', 'input', '1M_circle_16384.dat')
    # file = os.path.join(dir_path, '..', 'input', '1M_quadratic_2147483648.dat')
    arr = np.loadtxt(file)
    print(arr.shape)
    points = np.reshape(arr, (-1, 2))
    print(points.shape)

    hull = quickhull_2d(points)
    print(hull)
    print(len(hull))
