"""
------------------------------------------------------------------------------------------------------------
Question 1: Determine whether some anagram of t is a substring of s.
------------------------------------------------------------------------------------------------------------
"""
def question1(s, t):
    """ 
        Check if some anagram of t is present in s
        Function is case Insensitive
        Average case: O(l) 
        Worst case: O(l) 
        Where l is greater of length of s or t.
    """
    
    # Define useful functions
    def getDict(str):
        """ 
            Generate a dictionary from a string 
            Complexity: 
                Avg-case: O(n)*O(1)
                Worst-case: O(n) * O(m)
                where, n is the size of the string & m is size of the dict.
                
                Since for us, m is max 26 (number of alphabets),
                both average and worst case becomes O(n)
        """
        d={}
        for x in str:
            if x in d:
                d[x]+=1
            else:
                d[x]=1
        return d
    def compareDict(d1, d2):
        """
            Compare if two dictionaries are the same.
            That is, the have same keys and same counts for those keys.
            Complexity:
                Avg-case: O(m)* (O(1)+O(1))
                Worst-case: O(m) * (O(m) + O(m))
                where, m is the size of the dictionaries
            
                Since for us, m is max 26 (number of alphabets),
                both average and worst case becomes O(1)
        """
        for key in d1:
            if key not in d2:
                return False
            if (d1[key]!=d2[key]):
                return False
        return True    
    
    # Trivial Case: t is empty    
    if(len(t)==0):
        return True
        
    # Make case insensitive
    t = t.lower()
    s = s.lower()
    
    # Get the dictionary for t
    dictT=getDict(t) # Avg: O(1) / Worst: O(t)

    """ 
        We will repeat through substrings 
        of s of size t and form dictionaries.
        If t is an anagram of some substring of s,
        then the two dictionaries will be the same
        Time complexity:
            Running through the outer loop is O(s), and the most
            expensive inside function has Average and Worst case as O(1)
            Average case: O(s) * O(1) = O(s)
            Worst case: O(s) * O(1)  = O(s)
            where, s is length of string s.
            
    """
    
    # let's maintain a dictionary iteratively as we slide a window
    # of length len(t) through s.
    x={}
    for i in range(0,len(s)): # O(s) for the loop
        if x.has_key(s[i]):
            x[s[i]] += 1
        else:
            x[s[i]] = 1
        if i >= len(t):
            # Decrement the count for the character at s[i-len(t)]
            # if we have moved past it.
            key = s[i - len(t)]
            if x[key] == 1:
                x.pop(key)
            else:
                x[key] -= 1;
            
            if compareDict(x, dictT) is True: # O(1) (see function definition)
                return True
                
    return False 
    
# Test cases for question 1
print "Question 1:"
print "\tTest Case 1 - Expected: True\t Got : " + str(question1("who is your", "yruo"))  
print "\tTest Case 2 - Expected: True\t Got : " + str(question1("who is your", ""))  
print "\tTest Case 3 - Expected: True\t Got : " + str(question1("Dabble", "Babel"))  
print "\tTest Case 4 - Expected: False\t Got : " + str(question1("Dabble", "bob"))  
print "\tTest Case 5 - Expected: False\t Got : " + str(question1("", "yruo"))  

"""
------------------------------------------------------------------------------------------------------------
Question 2: Find the longest palindromic substring contained in a.
------------------------------------------------------------------------------------------------------------
"""

def question2(a):
    """ 
        Function returns the longest palindromic substring contained in a.
        Function is case insensitive
        Time complexity:
            Worst case: O(n)*O(n) = O(n^2)
    """
    # Trivial case
    if not len(a):
        return ""
    palindrome = ""
    a = a.lower()
    # Rational is to expand around each index and see if there is a potential palindrome there
    # Time complexity: This outer loop has complexity O(n), where n is length of a.
    for i in range(0, len(a)):
        # Start checking characters around ith position on either side in increasing length 
        
        # Check for palindrome of even length, where first half ends at i
        foundLength=0
        j=1
        # Increase foundLength while characters on both side are equal
        # This loop goes max of n/2 length in the worst case
        # Time complexity: O(n/2) = O(n)
        while ((i - (j -1))>= 0) and ((i+j) < len(a)):
            if a[i-(j-1)] == a[i+j]: 
                foundLength+=1
            else:
                break
            j+=1
        #Compare the string with the length of the longest palindrome
        #Time complexity is O(n) in the worst case
        if (foundLength>0):
            substring = a[(i+1-foundLength):(i+foundLength+1)]
            if(len(substring)>len(palindrome)):
                palindrome = substring
                
        # Check for palindrome of odd length, where first half ends before i
        foundLength=0
        j=1
        # Time complexity is similar to above O(n)
        while ((i - j)>= 0) and ((i+j) < len(a)):
            if a[i-j] == a[i+j]: 
                foundLength+=1
            else:
                break
            j+=1
            
        if (foundLength>0):
            substring = a[(i-foundLength):(i+foundLength+1)]
            if(len(substring)>len(palindrome)):
                palindrome = substring
    return palindrome


# Test cases for question 2
print "\n\nQuestion 2:"
print "\tTest Case 1 - Expected: abaaba \tGot : " + str(question2("CABAABADE"))  
print "\tTest Case 2 - Expected: yxsxy \tGot : " + str(question2("BYXSXYX"))  
print "\tTest Case 3 - Expected: abba \tGot : " + str(question2("ABBACDE"))  
print "\tTest Case 4 - Expected: (entire) \tGot : " + str(question2("Able was I ere I saw Elba"))  
print "\tTest Case 5 - Expected: nothing \tGot : " + str(question2("nopalindrome"))  

"""
------------------------------------------------------------------------------------------------------------
Question 3: Find the minimum spaning tree
------------------------------------------------------------------------------------------------------------
"""
def question3(graph):
    """
        Function takes a graph and returns the minimum spanning tree
        using Prim's algorithm
        Assumption 1: If an edge exists between any two vertices, both vertices should be present in the graph data
                   2: Assumes that graph is a connected graph, else tree will not be spanning
                   
        Total Time Complexity: 
            We repeat the while loop for all vertices of the graph 
            and most expensive inner function is O(e) * O(log(v))
            where, e is number of edges of one vertex and v is the number of verties in the hypotheses
            If we count all interations, the max complexity is 
                O(|V| + |E| * log(|V|))
                which is O(|E| log(|V|))
                where |V| is number of vertices and |E| is number of edges.
                This is because the loop repeats for all vertices and edges exactly once.
    """
    
    def addHypotheses(vertex, edges):
        """
            Function to add the edges and it's distances to the vertex to hypotheses.
            In our hypotheses, for the vertex on the other side of the edge,
            we see if there is any other edge that connects to the MST with a shorter distance. 
            If not, then this edge is the shortest for that vertex. So we update 
            the distanceFromVerticesToMST dict with the lower cost, and 
            nearestVertexInMST with the current vertex
            Time Complexity: 
                Runs for O(e) * log(v) 
                where, e is number of edges for this vertex, 
                and v is number of vertices in the hypotheses
        """
        for edge in edges: 
            #Get the cost and the other vertex from the edge
            otherVertex = edge[0]
            cost = edge[1]
            if(otherVertex in mst):
                # If the vertex is already in the MST then skip
                continue
            if otherVertex not in distanceFromVerticesToMST:
                # If other vertex not part of the hypotheses yet,
                # Add its cost and current vertex as it is nearest to the otherVertex
                distanceFromVerticesToMST[otherVertex] = cost
                nearestVertexInMST[otherVertex] = vertex
            else:
                # Else, 
                # Check if the cost of edge is lower than the best hypothesis 
                # for joining otherVertex to the MST. 
                if(cost<distanceFromVerticesToMST[otherVertex]):
                    # If yes, then cost and vertex become the best hypothesis for 
                    # connecting otherVertex to the MST
                    distanceFromVerticesToMST[otherVertex] = cost
                    nearestVertexInMST[otherVertex] = vertex
                
    def addToMst(vertex, edge):
        """
            Function to symmetrically add the vertex and edge to the MST
            Time Complexity: O(1)
        """
        mst[vertex] = []
        mst[vertex].append(edge)
        tup = (vertex, edge[1])
        mst[edge[0]].append(tup)
        
    def pickFromHypotheses():
        """
            Function to get the vertex that is nearest to the MST and the cost for it
            Time Complexity: O(v), where v is the number of vertexes in the hypotheses
        """
        if len(distanceFromVerticesToMST) == 0:
            return None, 0  # Check that hypotheses are not empty
            
        # Iterate throgh all the vertices in the hypotheses 
        # and find the vertex that has least cost and note it
        minCost = float('inf')
        minVertext = ''
        for vertex in distanceFromVerticesToMST:
            cost = distanceFromVerticesToMST[vertex]
            if cost < minCost: 
                minCost = cost
                minVertex = vertex
                
        # Remove that vertex from the hypotheses
        distanceFromVerticesToMST.pop(minVertex)        
        nearestNode = nearestVertexInMST.pop(minVertex)
        # Make the tuple for the edge and return the vertex and the edge
        return minVertex, (nearestNode,minCost)
        
    # Trivial case, handle empty or graph with only one vertex
    if len(graph) <= 1:
        return graph
    
    # The minimum spanning tree 
    mst={}
    
    # Empty hypotheses 
    distanceFromVerticesToMST = {}
    nearestVertexInMST = {}
    
    # Initialize the MST with one element from the graph
    vertex, edges = graph.popitem()    
    mst[vertex]=[]
    
    # Expand the hypotheses for that vertex for initial hypotheses
    addHypotheses(vertex, edges)
    
    # Repeat while graph is not empty
    while len(graph)>0:
        # Pick the nearest vertex to the MST and the edge
        vertex, edge = pickFromHypotheses()
        # Add it to the MST symmetrically
        addToMst(vertex, edge)
        # Get all the edges for that vertex by removing it from the graph
        edges = graph.pop(vertex)
        # Expand all hypotheses for that vertex and its edges
        addHypotheses(vertex, edges)
        
    return mst

print "\n\nQuestion 3:"
graph1 = {'a': [('b', 1)], 'b': [('a', 1), ('c', 5)], 'c': [('b', 5)]}
mst1=question3(graph1)
print "\tTest Case 1 - MST : {}".format(mst1)  

graph2 = {'a': [('b', 3), ('c', 1)], 'b': [('a', 3), ('c', 6)], 'c': [('a', 1), ('b', 6)]}
mst2=question3(graph2)
print "\tTest Case 2 - MST : {}".format(mst2)  

graph3 = {'a': [('b', 1), ('d', 1)], 'b': [('a', 1), ('c', 1)], 'c': [('d', 1), ('b', 1)], 'd': [('c', 1), ('a', 1)]}
mst3=question3(graph3)
print "\tTest Case 3 (should return linear tree) - MST : {}".format(mst3)  

graph4 = {'a': []}
mst4=question3(graph4)
print "\tTest Case 4 (should return itself) - MST : {}".format(mst4)  

"""
------------------------------------------------------------------------------------------------------------
Question 4: Least common ancestor between two nodes on a binary search tree
------------------------------------------------------------------------------------------------------------
"""
def question4(T, r, n1, n2):
    """
    Function for finding the least common ancestor. 
    Assumption: The tree is correct in its data including BST and nodes are in the tree
    We start from the root and do BST traversal twice. Once for finding the path to n1.
    And once to finding the least common ancestor.
    Time complexity:
        In an actual binary tree, with nodes : this would be O(2 * log(n)) average case and O (2 * n) worst case.
        But as we are moving through an adjacent matrix, the cost of finding the child node is O(n/2). 
        Cost of maintaining the path, to search for n2 in the path also takes O (log(n))
        So the total complexity is O(n * log(n)^2) in average case and O (n ^ 2 * log(n) ) worst case.
    """
    # We store the path to n1 in this variable
    path = []
    # We start from r and traverse BST to reach search n1. We note down the path traversed
    current = r
    for i in range(0, 5):
        path.append(current)
        if(current is n1):
            break
        next=-1 # To handle where no value is 1
        #This is standard BST search
        if(n1>current):
            for j in range(current+1,len(T[r])):
                if(T[r][j] is not 0):
                    next = j
        else:
            for j in range(0,current-1):
                if(T[r][j] > 0):
                    next = T[r][j]
        if(next<0):
            break
        current=next
    # We start from r and traverse BST to reach search n2. 
    # We verify if we are following the same path and break when the path are different
    current = r
    leastCommonAncestor = -1
    for i in range(0, 5):
        # While we are following the same path from the root, note the steps
        if(current in path):
            leastCommonAncestor = current
        else:
            #We are past the least common ancestor 
            break
        if(current is n2):
            break
        next=-1 # To handle where no value is 1
        #This is standard BST search
        if(n2>current):
            for j in range(current+1,len(T[r])):
                if(T[r][j] is not 0):
                    next = j
        else:
            for j in range(0,current-1):
                if(T[r][j] > 0):
                    next = T[r][j]
        if(next<0):
            break
        current=next
        
    return leastCommonAncestor 

print "\n\nQuestion 4:"
lca = question4([[0, 1, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[1, 0, 0, 0, 1],[0, 0, 0, 0, 0]], 3, 1, 4)   
print "\tTest Case 1 - Expected : {}, Got : {}".format(3, lca)  
print "\tTest Case 2 - Expected : {}, Got : {}".format(0, question4([ [0, 1], [0,0]], 0,0,1))
lca = question4([[0, 0, 0, 0, 0],[1, 0, 1, 0, 0],[0, 0, 0, 0, 0],[0, 1, 0, 0, 1],[0, 0, 0, 0, 0]], 3, 0, 2)   
print "\tTest Case 3 - Expected : {}, Got : {}".format(1, lca)  
print "\tTest Case 4 - Expected : {}, Got : {}".format(1, question4([ [0, 0], [1,0]], 1,1,0))
"""
------------------------------------------------------------------------------------------------------------
Question 5: Find the element in a singly linked list that's m elements from the end
------------------------------------------------------------------------------------------------------------
"""
class Node(object):
    """
        Class for the node as provided
    """
    def __init__(self, data):
        self.data = data
        self.next = None
        
def question5(ll, m):
    """ 
        Function returns the element in linked list ll that is m elements from the end.
        Idea here is to traverse to the end of the list, reversing it.
        Then we move back to the front, and count the nodes and note the element at m counts.
        We also reverse back the links to the original
        Assumptions:
            There is no linked list class that holds size of the list
    """
    
    #First we move to the end of the list, and reverse the links while doing so.
    current=ll
    previous=None
    while(True):
        nextNode = current.next
        current.next = previous
        if(nextNode is None):
            break
        previous=current
        current = nextNode

    count = m
    element = None
    # Then we move back to the front, counting the number of elements
    # We note the mth element.
    # While moving to the front, we also reverse the links to point the links in the original direction
    previous=None
    while(True):
        count=count-1
        if(count==0):
            element=current
        nextNode = current.next
        current.next = previous
        if(nextNode is None):
            break
        previous=current
        current=nextNode

    return element
    
def question5_create(arr):
    """
        Helper function to create a linked list from an array 
    """
    previous=None
    # Create the list in a reverse manner
    for i in reversed(range(0,len(arr))):
        node = Node(arr[i])
        node.next=previous
        previous = node
    head = previous
    return head

def question5_print(ll):
    """
        Helper function to print the list
    """
    current = ll
    arr = []
    #Move to the end of the list
    while current is not None:
        arr.append(current.data)
        current=current.next
    print '{}'.format(arr)
    
print "\n\nQuestion 5:"
head = question5_create([1,"hi",3,4.0,-35])
print "\tTest Case 1 - (ll, 3) - Expected : {}, Got : {}".format(3, question5(head, 3).data)  
print "\tTest Case 2 - (ll, 0) - Expected : {}, Got : {}".format(None, question5(head, 0))  
print "\tTest Case 3 - (ll, 1) - Expected : {}, Got : {}".format(-35, question5(head, 1).data)  
print "\tTest Case 4 - (ll, 6) - Expected : {}, Got : {}".format(None, question5(head, 6))  
