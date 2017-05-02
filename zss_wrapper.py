import zss


def zss_with_descriptor(tree1, tree2, tree_descriptor):
    if tree_descriptor == 'v1':
        return zss.simple_distance(tree1.root, tree2.root)
    else:
        penalty = 10
        weights = (1, 1, 1)

        desc = tree_descriptor.split(',')[1:]

        if len(desc) > 3:
            penalty = int(desc[3])
        if len(desc) > 2:
            weights = tuple(map(int, desc[:3]))

        def dist_method(label_a, label_b):
            return enhanced_label_dist(label_a, label_b, weights)

        return zss.distance(
            tree1.root, tree2.root,
            insert_cost=lambda node: label_dist([], node.enhanced_label, penalty, dist_method),
            remove_cost=lambda node: label_dist(node.enhanced_label, [], penalty, dist_method),
            update_cost=lambda node1, node2: label_dist(
                node1.enhanced_label, node2.enhanced_label, penalty, dist_method
            ),
            get_children=lambda node: node.children
        )


def label_dist(a_label, b_label, empty_distance, dist_method):
    # label = list of nt_labels

    if len(a_label) == 0 or len(b_label) == 0:
        return empty_distance * max(len(a_label), len(b_label))

    if len(a_label) == len(b_label) and len(a_label) == 1:
        return dist_method(a_label[0], b_label[0])

    if len(a_label) < len(b_label):
        oo = dist_method(a_label[0], b_label[0])
        oi = dist_method(a_label[0], b_label[1])
        return min(oo + empty_distance, oi + empty_distance)

    if len(b_label) < len(a_label):
        oo = dist_method(a_label[0], b_label[0])
        io = dist_method(a_label[1], b_label[0])
        return min(oo + empty_distance, io + empty_distance)

    oo = dist_method(a_label[0], b_label[0])
    oi = dist_method(a_label[0], b_label[1])
    io = dist_method(a_label[1], b_label[0])
    ii = dist_method(a_label[1], b_label[1])

    return min(oo + ii, oi + io)


def enhanced_label_dist(a_label, b_label, weights):
    # nt_label = list
    #   nucleotide_type (str)                       e.g. 'A', 'C'
    #   eta, theta (tuple of floats)                e.g. ((10.3, -40.3)), ((-140, 30), (-20, 60))
    #   counts in near area (list of ints)          e.g. [4, 8, 10, 30, 50]

    # 1. nucleotide_types
    penalty_1 = 0 if a_label[0] == b_label[0] else 1

    # 2. eta, theta
    eta_a, theta_a = tuple(map(lambda angle: angle + 180 if angle else angle, a_label[1]))  # numbers in (-180, 180)
    eta_b, theta_b = tuple(map(lambda angle: angle + 180 if angle else angle, b_label[1]))  # after transform (0, 360)
    penalty_2 = 0
    if not eta_a or not eta_b:
        penalty_2 += 0.5
    else:
        penalty_2 += abs(eta_b - eta_a) / 720.0  # I want eta and theta to have 0.5 max penalty each

    if not theta_a or not theta_b:
        penalty_2 += 0.5
    else:
        penalty_2 += abs(theta_b - theta_a) / 720.0

    # 3. counts in near area
    penalty_3 = 0

    neighborhood_a = a_label[2]
    neighborhood_b = b_label[2]

    penalty_ratio = 1.0 / len(neighborhood_a)

    for n_a, n_b in zip(neighborhood_a, neighborhood_b):
        max_n = float(max(n_a, n_b))
        penalty_3 += penalty_ratio * ((max_n - min(n_a, n_b)) / max_n)

    # Total: ...
    return weights[0] * penalty_1 + weights[1] * penalty_2 + weights[2] * penalty_3
