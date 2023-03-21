class DecisionTreeNode:

    def __init__(self, attribute=None, name=None, probability=None, left_child=None, right_child=None, size=None):
        self.attribute = attribute
        self.name = name
        self.probability = probability
        self.left_child = left_child
        self.right_child = right_child
        self.size = size

    def __str__(self):
        if self.left_child == None:
            return 'Leaf[' + str(self.name) + ', ' + str(self.probability) + ']\n\n'

        return 'Node:' + str(self.attribute) + \
               '\n\t' + self.left_child.__str__() + '\n\t' + self.right_child.__str__()

    def print_node(self, depth):
        indent = ''
        for i in range(depth):
            indent += "  "

        s = ''
        for i in range(depth-1):
            s += "  "

        if self.left_child is None:
            return 'Class: ' + str(self.name) + ' prob: ' + str(self.probability) + '\n'
        # remove last one indent for false child
        return str(self.attribute) + '=True: \n' + indent + self.left_child.print_node(depth+1) + \
               s + str(self.attribute) + '=False: \n' + indent + self.right_child.print_node(depth+1)
