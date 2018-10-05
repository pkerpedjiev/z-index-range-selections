import math
import numba
import numpy as np
import functools as ft


@numba.jit(nopython=True)
def less_than(xa, ya, xb, yb):
    '''
    Assuming a and b are on adjacent vertices of
    a z-order curve, return -1 if a < b, 0 if
    a == b and 1 if a > b
    
    Parameters
    ----------
    a: (float, float)
        Point
    b: (float, float)
        Point
    '''
    #print("less_than")
        
    ret = 1
    
    if xa == xb and ya == yb:
        return 0
    
    if ya < yb:
        ret = -1
    else:
        if xa < xb:
            ret = -1
    if ya > yb:
        ret = 1
    
    #print(ret)
    return ret

@numba.jit(nopython=True)
def tile_children(a,b,c,d):
    '''
    Split a tile into its four children
    
    Parameters
    ----------
    tile: (float, float, float, float)
    '''
    tile_width = c - a
    tile_height = d - b
    
    return np.array([a, b, a + tile_width / 2, b + tile_height / 2,
            a, b + tile_height / 2, a + tile_width / 2, d,
           a + tile_width / 2, b, c, b + tile_height / 2,
           a + tile_width / 2, b + tile_height / 2, c, d])

@numba.jit(nopython=True)
def zindex_compare(a,b,tx0, ty0, tx1, ty1):
    '''
    Compare two points along z curve wihin the bounds of 
    the given tile.
    
    Parameters
    ----------
    a: (float, float)
        Point 1
    b: (float, float)
        Point 2
    tile: (float, float, float, float)
        The bounds of the region in which we want to compare them
    '''
    #print(a, tile, in_tile(a,tile))
    #print("a,b,tile", a,b, tile)
    # assert(in_tile(a,tile))
    # assert(in_tile(b, tile))
    
    #print("comparing", a, b)
    
    if a[0] == b[0] and a[1] == b[1]:
        return 0
    
    tile_width = tx1 - tx0
    tile_height = ty1 - ty0
    
    #children = tile_children(*tile)
    
    quadrant_a = (math.floor( 2 * (a[0] - tx0) / tile_width), 
                        math.floor( 2 * (a[1] - ty0) / tile_height))
    quadrant_b = (math.floor( 2 * (b[0] - tx0) / tile_width), 
                        math.floor( 2 * (b[1] - ty0) / tile_height))
    
    if quadrant_a[0] == quadrant_b[0] and quadrant_a[1] == quadrant_b[1]:
        #print("child_a", child_a, "child_b", child_b)
        x0 = tx0 + quadrant_a[0] * tile_width / 2
        y0 = ty0 + quadrant_a[1] * tile_height / 2
        
        x1 = x0 + tile_width / 2
        y1 = y0 + tile_width / 2
        
        return zindex_compare(a, b, x0,y0,x1,y1)
    else:
        xa = tx0 + quadrant_a[0] * tile_width / 2
        ya = ty0 + quadrant_a[1] * tile_height / 2
        
        xb = tx0 + quadrant_b[0] * tile_width / 2
        yb = ty0 + quadrant_b[1] * tile_height / 2
        
        return less_than(xa,ya,xb,yb)
        #return less_than(child_a[:2], child_b[:2])
 
@numba.jit(nopython=True)
def lower_bound(sequence, value, tx0, ty0, tx1, ty1):
    """Find the index of the first element in sequence >= value"""
    elements = len(sequence)
    offset = 0
    middle = 0
    found = len(sequence)
 
    while elements > 0:
        middle = elements // 2
        #print("middle:", middle)
        ix = offset + middle
        if zindex_compare(value, sequence[ix], tx0, ty0, tx1, ty1) > 0:
            offset = offset + middle + 1
            elements = elements - (middle + 1)
        else:
            found = offset + middle
            elements = middle
    return found

def upper_bound(sequence, value, tx0, ty0, tx1, ty1):
    """Find the index of the first element in sequence > value"""
    elements = len(sequence)
    offset = 0
    middle = 0
    found = 0
 
    while elements > 0:
        middle = elements // 2
        if zindex_compare(value, sequence[offset + middle], tx0, ty0, tx1, ty1) < 0:
            elements = middle
        else:
            offset = offset + middle + 1
            found = offset
            elements = elements - (middle + 1)
    return found

def all_point_boundaries(points_list, tile, bounding_tile):   
    # print("points_list", points_list)
    left_index = lower_bound(points_list, 
        np.array([tile[0], tile[1]]), *bounding_tile)
    right_index = lower_bound(points_list, 
        np.array([tile[2] - 0.000001, tile[3] - 0.000001]), *bounding_tile)
    #print("right_index", right_index)
    
    return left_index, right_index

def all_points(points_list, tile, bounding_tile):
    '''
    Return all points that are in the tile given tile
    '''
    left_index, right_index = all_point_boundaries(points_list, tile, bounding_tile)
    
    return points_list[left_index:right_index]


@numba.jit(nopython=True)
def all_in(tile, rect):
    '''
    Is this tile completely enclosed in the rectangle:
    
    Parameters
    ----------
    tile: [float, float, float, float]
        The xmin,ymin,xmax,ymax of the tile
    rect: [float, float, float, float]
        The xmin, ymin, xmas, ymax of the rectangle
    '''

    # epsilon to prevent rejecting rectangles and 
    # leading to an infine loop when floating point
    # precision prevents further subdivision
    epsilon = 0.00000001
    #ret = np.all(np.abs(tile - rect) < 0.0000001)
    #return ret
    #print("ret1:", ret)

    ret = (tile[0] >= (rect[0] - epsilon) and
            tile[1] >= (rect[1] - epsilon) and 
            tile[2] <= (rect[2] + epsilon) and
            tile[3] <= (rect[3] + epsilon))
    #print("ret2:", ret)
    #print("all_in:", ret, tile, rect)
    return ret

@numba.jit(nopython=True)
def some_in(tile, rect):
    '''
    Is part of this tile within the rectangle?
    
    Parameters
    ----------
    tile: [float, float, float, float]
        The xmin,ymin,xmax,ymax of the tile
    rect: [float, float, float, float]
        The xmin, ymin, xmas, ymax of the tile
    '''
    ret = (tile[0] < rect[2] and
            tile[1] < rect[3] and 
            tile[2] > rect[0] and
            tile[3] >= rect[1])
    
    #print("some_in", ret)
    return ret


import base64

def get_points(points_list, rect, tile, bounding_tile):
    '''
    Get all the points in the tile that intersect the rectangle
    
    Parameters
    ----------
    rect: [float, float, float, float]
        minx, miny, maxx, maxy
    tile: [float, float, float, float]
        minx, miny, maxx, maxy
    '''
    points = []
    #print("tile:", tile)
    tiles_to_check = [tile]
    tiles_checked = 0
    
    while len(tiles_to_check) > 0:
        tile = tiles_to_check.pop()
        # print("tile:", tile)
        
        tiles_checked += 1

    
        for child in tile_children(*tile).reshape((4,-1)):
            if not some_in(child, rect):
                # no intersection
                continue

            left_index, right_index = all_point_boundaries(
                points_list, child, bounding_tile)

            if left_index == right_index:
                continue

            if all_in(child, rect):
                #print("all points", tile)
                points += list(points_list[left_index:right_index])
                continue

            tiles_to_check += [child]
            #points += get_points(points_list, rect, child, bounding_tile) 

        '''
        if len(tiles_to_check) > 100:
            print("len", len(tiles_to_check), base64.b64encode(tile), base64.b64encode(rect), tile[0].hex()) #, rect)
        '''
    
    # print("tiles_checked:", tiles_checked)
    return points


@numba.jit(nopython=True)
def ix(i,j):
    return i


@numba.jit( nopython=True)
def quicksort_zindex(a, s, e, x0, y0, x1, y1):
    '''
    # From here: https://github.com/lprakash/Sorting-Algorithms/blob/master/sorts.ipynb
    '''
    #print(a, s, e)
    #print(a.__repr__)
    # stack = []    

    if (e-s)==0:
        return 

    
    pivot_a = a[2*(e-1)]
    pivot_b = a[2*(e-1) + 1]

    p1 = s
    p2 = e - 1

    while (p1 != p2):
        #p1 += 2
                
        #print("2*p1", 2*p1+1)
        comp = zindex_compare([a[2*p1], a[2*p1+1] ], [pivot_a, pivot_b], x0, y0, x1, y1)
        #print("y", 2*(p2-1)+1)
        if comp > 0:
            #x = ix(p2, 0)
            #a[ix(p2, 0)] = a[ix(p1,0)]
            
            a[2*p2 + 0] = a[2*p1+0]
            a[2*p2 + 1] = a[2*p1+1]

            a[2*p1+0] = a[2*(p2-1)+0]
            a[2*p1+1] = a[2*(p2-1)+1]

            a[2*(p2-1)+0] = pivot_a
            a[2*(p2-1)+1] = pivot_b
            
            p2 = p2 -1
        else: 
            p1+=1
        

    quicksort_zindex(a, s, p2, x0, y0, x1, y1)
    quicksort_zindex(a, p2+1, e, x0, y0, x1, y1)

    return a

@numba.jit(nopython=True)
def quicksort_zindex1(a, s, e, x0, y0, x1, y1):
    '''
    # From here: https://github.com/lprakash/Sorting-Algorithms/blob/master/sorts.ipynb
    '''
    #print(a, s, e)
    #print(a.__repr__)
    # stack = []

    if (e-s)==0:
        return 

    pivot_a = a[e-1,0]
    pivot_b = a[e-1,1]

    p1 = s
    p2 = e - 1
    while (p1 != p2):
        comp = zindex_compare(a[p1], [pivot_a, pivot_b], x0, y0, x1, y1)

        if comp > 0:
            a[p2,0] = a[p1,0]
            a[p2,1] = a[p1,1]
            a[p1,0] = a[p2-1,0]
            a[p1,1] = a[p2-1,1]
            a[p2-1,0] = pivot_a
            a[p2-1,1] = pivot_b
            p2 = p2 -1
        else: 
            p1+=1

    quicksort_zindex(a, s, p2, x0, y0, x1, y1)
    quicksort_zindex(a, p2+1, e, x0, y0, x1, y1)

    return a

def quicksort(a, s, e):
    #print(a, s, e)
    #print(a.__repr__)
    if (e-s)==0:
        return 
    pivot = a[e-1]
    print("pivot:", pivot)
    p1 = s
    p2 = e - 1
    while (p1 != p2):
        if (a[p1] > pivot):
            print('setting {} to {} and {} to {} and {} to pivot ({})'.format(p2, p1, p1, p2-1, p2-1, e-1, pivot) )
            print("pre:", a)
            a[p2] = a[p1]
            print("pivot:", pivot)
            a[p1] = a[p2-1]
            print("pivot:", pivot)
            a[p2-1] = pivot
            print("post:", a)
            print("--------------")
            p2 = p2 -1
        else: 
            p1+=1
    quicksort(a, s, p2)
    quicksort(a, p2+1, e)

def zindex_sort(points, bounds):
    '''
    Sort a 2d array according to a z-index based ordering

    Parameters
    ----------
    array: np.array (shape: (-1,2))
        The array of points to sort
    bounds: [float, float, float, float]
        An array of the bounds of the area to sort
    '''
    return quicksort_zindex(points.reshape((-1,)), 
                                   0, len(points), 
                                   *bounds).reshape((-1,2))




