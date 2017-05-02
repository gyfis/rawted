def herpin_init_predicate(node):
    return node.child_count < 2 or all(child.is_leaf for child in node.children)


def herpin_result_predicate(node):
    return node.child_count == 1 or (not node.is_leaf and all(child.is_leaf for child in node.children))


def true_predicate(_):
    return True


def inner_node_predicate(node):
    return not node.is_leaf
