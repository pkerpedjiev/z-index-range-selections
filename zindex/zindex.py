import math
import numba as nb
import numpy as np
import functools as ft

"""
@nb.njit()
def __part1by1(n):
    n &= 0x0000ffff                  # base10: 65535,      binary: 1111111111111111,                 len: 16
    n = (n | (n << 8))  & 0x00FF00FF # base10: 16711935,   binary: 111111110000000011111111,         len: 24
    n = (n | (n << 4))  & 0x0F0F0F0F # base10: 252645135,  binary: 1111000011110000111100001111,     len: 28
    n = (n | (n << 2))  & 0x33333333 # base10: 858993459,  binary: 110011001100110011001100110011,   len: 30
    n = (n | (n << 1))  & 0x55555555 # base10: 1431655765, binary: 1010101010101010101010101010101,  len: 31

    return n
"""
@nb.njit(fastmath=True)
def __part1by1(n):
    n = n & 0x00000000ffffffff
    n = (n | (n << 16)) & 0x0000FFFF0000FFFF 
    n = (n | (n << 8))  & 0x00FF00FF00FF00FF # base10: 16711935,   binary: 111111110000000011111111,         len: 24
    n = (n | (n << 4))  & 0x0F0F0F0F0F0F0F0F # base10: 252645135,  binary: 1111000011110000111100001111,     len: 28
    n = (n | (n << 2))  & 0x3333333333333333 # base10: 858993459,  binary: 110011001100110011001100110011,   len: 30
    n = (n | (n << 1))  & 0x5555555555555555 # base10: 1431655765, binary: 1010101010101010101010101010101,  len: 31

    return n

@nb.guvectorize([(nb.uint64[:], nb.uint64[:], nb.uint64[:])], '(n),(n)->(n)')
def interleave(x,y,out):
    for i in range(x.shape[0]):
        out[i] = __part1by1(x[i]) | (__part1by1(y[i]) << 1)

@nb.njit()
def interleave1(x):
    return nb.uint64(__part1by1(np.uint64(x[0])) | (__part1by1(nb.uint64(x[1])) << 1))

@nb.njit()
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


@nb.njit(fastmath=True)
def num2(atx, aty, i):    
    axbit = atx >> i;
    aybit = (aty >> i) << 1;
    anum = axbit | aybit;
    return anum
    
    bxbit = btx >> i;
    bybit = (bty >> i) << 1;
    bnum = bxbit | bybit;
    
    return anum - bnum

@nb.njit(fastmath=True)
def zindex_compare(a0, a1, b0, b1, tx0, ty0, tx1, ty1):
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
    
    if a0 == b0 and a1 == b1:
        return 0
    
    whole_width = tx1 - tx0
    whole_height = ty1 - ty0
    #print("point1", a0, a1, "point2", b0, b1)

    zoom = 32
    xz = 2 ** zoom

    tile_width = whole_width / xz
    tile_height = whole_height / xz  

    atx = int((a0 - tx0) // tile_width)
    btx = int((b0 - tx0) // tile_width)

    aty = int((a1 - ty0) // tile_height)
    bty = int((b1 - ty0) // tile_height)

    return atx - aty + bty

    """
    while zoom > 0:
        #print("a0:", a0, "a0 - tx0", a0 - tx0, )

        i = zoom - 1
        #for i in range(zoom-1, -1, -1):
        #print("zoom", zoom, "i:", i,  "atx,y", atx, aty, "btx,y", btx, bty);
        #print("[", atx, ",", aty, ",", btx, ",", bty, "],")
                
        #axbit = atx >> i;
        #aybit = (aty >> i-1)

        #bxbit = btx >> i;
        #ybit = (bty >> i-1)

        return atx - aty + btx

        d1 = num2(atx, aty, i)
        d2 = num2(btx, bty, i)

        #if d != 0:
        #    return d
        return d1 - d2
        zoom -= 1
    """

    #print("0")
    return 0
    #children = tile_children(*tile)
    tile_width = tx1 - tx0
    tile_height = ty1 - ty0
    
    quadrant_a0 = math.floor( 2 * (a0 - tx0) / tile_width) 
    quadrant_a1 = math.floor( 2 * (a1 - ty0) / tile_height)
    quadrant_b0 = math.floor( 2 * (b0 - tx0) / tile_width) 
    quadrant_b1 = math.floor( 2 * (b1 - ty0) / tile_height)
    
    x0 = tx0 + quadrant_a0 * tile_width / 2
    y0 = ty0 + quadrant_a1 * tile_height / 2

    if quadrant_a0 == quadrant_b0 and quadrant_a1 == quadrant_b1:
        #print("child_a", child_a, "child_b", child_b)
        x1 = x0 + tile_width / 2
        y1 = y0 + tile_width / 2
        
        return zindex_compare(a0,a1, b0,b1, x0,y0,x1,y1)
    else:
        xb = tx0 + quadrant_b0 * tile_width / 2
        yb = ty0 + quadrant_b1 * tile_height / 2
        
        return less_than(x0,y0,xb,yb)
 
@nb.jit(nopython=True)
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
        if zindex_compare(value[0], value[1], 
                sequence[ix][0], sequence[ix][1], 
                tx0, ty0, tx1, ty1) > 0:
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
        if zindex_compare(value[0], value[1], 
                sequence[offset + middle][0], 
                sequence[offset + middle][1],
                tx0, ty0, tx1, ty1) < 0:
            elements = middle
        else:
            offset = offset + middle + 1
            found = offset
            elements = elements - (middle + 1)
    return found

'''
def all_point_boundaries(points_list, tile, bounding_tile):   
    # print("points_list", points_list)
    left_index = lower_bound(points_list, 
        np.array([tile[0], tile[1]]), *bounding_tile)
    right_index = lower_bound(points_list, 
        np.array([tile[2] - 0.000001, tile[3] - 0.000001]), *bounding_tile)
    #print("right_index", right_index)
    
    return left_index, right_index
'''




def all_points(points_list, tile, bounding_tile):
    '''
    Return all points that are in the tile given tile
    '''
    left_index, right_index = all_point_boundaries(points_list, tile, bounding_tile)
    
    return points_list[left_index:right_index]


@nb.njit()
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

    ret = (tile[0] >= (rect[0]) and
            tile[1] >= (rect[1]) and 
            tile[2]-1 <= (rect[2]) and
            tile[3]-1 <= (rect[3]))
    #print("ret2:", ret)
    #print("all_in:", ret, tile, rect)
    return ret

@nb.njit()
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
    ret = (tile[0] <= rect[2] and
            tile[1] <= rect[3] and 
            tile[2] >= rect[0] and
            tile[3] >= rect[1])
    #print("ret:", ret, tile, rect, tile[0], rect[2], tile[0] < rect[2])    
    return ret


import base64

@nb.njit()
def tile_children(a,b,c,d):
    '''
    Split a tile into its four children
    
    Parameters
    ----------
    tile: (float, float, float, float)
    '''
    tile_width = c - a
    tile_height = d - b
    
    return np.array([a, b, a + tile_width // 2, b + tile_height // 2,
            a, b + tile_height // 2, a + tile_width // 2, d,
           a + tile_width // 2, b, c, b + tile_height // 2,
           a + tile_width // 2, b + tile_height // 2, c, d]).astype(np.uint64)


@nb.njit
def all_point_boundaries(points_list, int_bounds):
    # print("searchsorted")
    #print("int_bounds:", int_bounds)
    #print("int_bounds:", int_bounds[:,0])
    #interleaved_bounds = interleave(int_bounds[:,0], int_bounds[:,1])
    interleaved_x = interleave1(int_bounds[0])
    interleaved_y = interleave1(int_bounds[1])
    #print("int bounds", int_bounds)
    #print("interleaved_bounds:", interleaved_bounds)
    #print("points list", points_list[-10:])

    left_index = np.searchsorted(points_list, interleaved_x)
    right_index = np.searchsorted(points_list, interleaved_y)

    if right_index < len(points_list) and points_list[right_index] == interleaved_y:
        right_index += 1
    #print("left_index:", left_index, "right_index", right_index, "zoom:", zoom)

    return left_index, right_index
    
@nb.njit
def get_points(points_list, rect, bounds, zoom=32):
    '''
    Get all the points in the tile that intersect the rectangle
    
    Parameters
    ----------
    rect: [float, float, float, float]
        minx, miny, maxx, maxy
    tile: [float, float, float, float]
        minx, miny, maxx, maxy
    '''
    #print("tile:", tile)
    tiles_checked = 0

    tile_width = (bounds[2] - bounds[0]) / 2 ** zoom
    tile_height = (bounds[3] - bounds[1]) / 2 ** zoom

    #print("tile_width", tile_width)

    tile_bounds = bounds.reshape((-1,2))
    tile_int_bounds = ((tile_bounds - bounds[0]) // tile_width).astype(np.uint64).reshape((-1,))

    rect_bounds = rect.reshape((-1,2))
    rect_int_bounds = ((rect_bounds - bounds[0]) // tile_width).astype(np.uint64).reshape((-1,))

    #print("tile_int_bounds:", tile_int_bounds)
    indeces = [0]
    tiles_to_check = [0,0,0,0]
    tiles_to_check += [tile_int_bounds[0], 
            tile_int_bounds[1], 
            tile_int_bounds[2], 
            tile_int_bounds[3]]

    #print("tiles_to_check:", tiles_to_check, "rect_bounds:", rect_int_bounds)

    m1 = np.array([0,0,-1,-1])
    not_visited = 0

    while len(tiles_to_check) > 1:
        y1 = tiles_to_check.pop()
        x1 = tiles_to_check.pop()
        y0 = tiles_to_check.pop()
        x0 = tiles_to_check.pop()
        
#        print("len(tiles_to_check)", len(tiles_to_check))
        
        children = tile_children(x0, y0, x1, y1).reshape((4, -1))
        #print("children:", children)
        for i in range(len(children)):
            child = children[i]
        
            tiles_checked += 1
            if not some_in(child, rect_int_bounds):
                # print("continuing", child)
                # no intersection
                continue

            #print("child:", child, (child+m1).astype(np.uint64).reshape((-1,2)))
            left_index, right_index = all_point_boundaries(
                points_list, (child+m1).astype(np.uint64).reshape((-1,2)))

            #print("left_index:", left_index, "right_index:", right_index)

            if left_index == right_index:
                not_visited += 1
                #print("same index")
                continue

            #print("li:", left_index, "ri", right_index)
            #print("child", child, [c * tile_width for c in child], left_index, right_index, len(tiles_to_check))
            #print("child", child)

            if all_in(child, rect_int_bounds):
                #print("all points", child, "indeces", left_index, right_index)
                indeces += [left_index, right_index]
                #points += list(points_list[left_index:right_index])
                continue

            #print("adding:", child, rect)
            if child[3] - child[1] > 1 and child[2] - child[0] > 1:
                tiles_to_check += [child[0]]
                tiles_to_check += [child[1]]
                tiles_to_check += [child[2]]
                tiles_to_check += [child[3]]
            #points += get_points(points_list, rect, child, bounding_tile) 

        '''
        if len(tiles_to_check) > 100:
            print("len", len(tiles_to_check), base64.b64encode(tile), base64.b64encode(rect), tile[0].hex()) #, rect)
        '''
    
    #print("tiles_checked:", tiles_checked, not_visited)
    return indeces


@nb.jit(nopython=True)
def ix(i,j):
    return i

@nb.njit(fastmath=True)
def quicksort_zindex(a, s, e, x0, y0, x1, y1):
    '''
    # From here: https://github.com/lprakash/Sorting-Algorithms/blob/master/sorts.ipynb
    '''
    #print(a, s, e)
    #print(a.__repr__)
    # stack = []    
    pivot_a = a[2*(e-1)]
    pivot_b = a[2*(e-1) + 1]

    p1 = s
    p2 = e - 1

    while (p1 != p2):
        tp2 = 2*p2
        tp1 = 2*p1

        comp = zindex_compare(a[tp1], a[tp1+1] , pivot_a, pivot_b, x0, y0, x1, y1)
        comp = a[tp1] < a[tp2]
        # print("comp", comp)
        #print("y", 2*(p2-1)+1)
        if comp > 0:
            #x = ix(p2, 0)
            #a[ix(p2, 0)] = a[ix(p1,0)]
            
            a[tp2] = a[tp1]
            a[tp2 + 1] = a[tp1+1]

            a[tp1+0] = a[tp2-2]
            a[tp1+1] = a[tp2-1]

            a[tp2-2] = pivot_a
            a[tp2-1] = pivot_b
            
            p2 = p2 -1
        else: 
            p1+=1
        

    if p2-s > 0:
        quicksort_zindex(a, s, p2, x0, y0, x1, y1)
    if e-(p2+1) > 0:
        quicksort_zindex(a, p2+1, e, x0, y0, x1, y1)

    return a

@nb.jit(nopython=True)
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




