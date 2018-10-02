
import numpy as np

from numpy.linalg import norm

eps = 1e-10


def rot_matrix(theta):
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ct, -st],[st, ct]])

def polygon_to_rbox(xy):
    # center point plus width, height and orientation angle
    tl, tr, br, bl = xy
    # length of top and bottom edge
    dt, db = tr-tl, bl-br
    # center is mean of all 4 vetrices
    cx, cy = c = np.sum(xy, axis=0) / len(xy)
    # width is mean of top and bottom edge length
    w = (norm(dt) + norm(db)) / 2.
    # height is distance from center to top edge plus distance form center to bottom edge
    h = norm(np.cross(dt, tl-c))/(norm(dt)+eps) + norm(np.cross(db, br-c))/(norm(db)+eps)
    #h = point_line_distance(c, tl, tr) +  point_line_distance(c, br, bl)
    #h = (norm(tl-bl) + norm(tr-br)) / 2.
    # angle is mean of top and bottom edge angle
    theta = (np.arctan2(dt[0], dt[1]) + np.arctan2(db[0], db[1])) / 2.
    return np.array([cx, cy, w, h, theta])

def rbox_to_polygon(rbox):
    cx, cy, w, h, theta = rbox
    box = np.array([[-w,h],[w,h],[w,-h],[-w,-h]]) / 2.
    box = np.dot(box, rot_matrix(theta))
    box += rbox[:2]
    return box

def polygon_to_rbox2(xy):
    # two points at the top left and top right corner plus height
    tl, tr, br, bl = xy
    # length of top and bottom edge
    dt, db = tr-tl, bl-br
    # height is mean between distance from top to bottom right and distance from top edge to bottom left
    h = (norm(np.cross(dt, tl-br)) + norm(np.cross(dt, tr-bl))) / (2*(norm(dt)+eps))
    return np.hstack((tl,tr,h))

def rbox2_to_polygon(rbox):
    x1, y1, x2, y2, h = rbox
    alpha = np.arctan2(x1-x2, y2-y1)
    dx = -h*np.cos(alpha)
    dy = -h*np.sin(alpha)
    xy = np.reshape([x1,y1,x2,y2,x2+dx,y2+dy,x1+dx,y1+dy], (-1,2))
    return xy

def polygon_to_rbox3(xy):
    # two points at the center of the left and right edge plus heigth
    tl, tr, br, bl = xy
    # length of top and bottom edge
    dt, db = tr-tl, bl-br
    # height is mean between distance from top to bottom right and distance from top edge to bottom left
    h = (norm(np.cross(dt, tl-br)) + norm(np.cross(dt, tr-bl))) / (2*(norm(dt)+eps))
    p1 = (tl + bl) / 2.
    p2 = (tr + br) / 2. 
    return np.hstack((p1,p2,h))

def rbox3_to_polygon(rbox):
    x1, y1, x2, y2, h = rbox
    alpha = np.arctan2(x1-x2, y2-y1)
    dx = -h*np.cos(alpha) / 2.
    dy = -h*np.sin(alpha) / 2.
    xy = np.reshape([x1-dx,y1-dy,x2-dx,y2-dy,x2+dx,y2+dy,x1+dx,y1+dy], (-1,2))
    return xy
