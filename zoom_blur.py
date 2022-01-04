import math
import numpy as np

def rand(co, seed):
    a = 12.9898
    b = 78.233
    c = 43758.5453
    dt = np.dot(np.array(co) + seed, np.array([a, b]))
    sn = dt % 3.14159
    return (math.sin(sn) * c + seed) % 1
 

def zoom_blur(strength, center, inner_radius, radius, filter_area, v_texture_coord, max_kernel_size):
    min_gradient = inner_radius * 0.3
    inner_radius = (inner_radius + min_gradient * 0.5) / filter_area[0]

    gradient = radius * 0.3
    radius = (radius - gradient * 0.5) / filter_area[0]

    count_limit = max_kernel_size
    dir = [(center[i] / filter_area[i]) - v_texture_coord[i]
           for i in range(2)]
    dist = math.hypot([dir[0], dir[1] * filter_area[1]/filter_area[0]])

    delta = 0.0
    gap = None
    if dist < inner_radius:
        delta = inner_radius - dist
        gap = min_gradient
    elif radius >= 0.0 and dist > radius:
        delta = dist - radius
        gap = gradient
    
    if delta > 0.0:
        normal_count = gap / filter_area[0]
        delta = (normal_count - delta) / normal_count
        count_limit *= delta
        strength *= delta
        if count_limit < 1.0:
            gl_frag_color = sampler[v_texture_coord[0]][v_texture_coord[1]]
            return

    offset = rand(v_texture_coord, 0.0)

    total = 0.0
    color = [0.0] * 4
    
    dir *= strength

    for t in range(max_kernel_size):
        percent = (t + offset) / max_kernel_size
        weight = 4.0 * (percent - percent * percent)
        p = np.array(v_texture_coord) + dir * percent
        sample = sampler[p[0]][p[1]]

        color += sample * weight
        total += weight

        if t > count_limit:
            break
    
    color /= total

    gl_frag_color = color




sampler = img


filter_area = [1, 1, 1, 1]  # TODO
v_texture_coord = [1, 1]  # TODO
MAX_KERNEL_SIZE = 32


strength = 0.1
center = [0, 0]
inner_radius = 0
radius = -1

u_center = center
u_inner_radius = inner_radius
u_radius = radius


# uniform sampler2D uSampler

zoom_blur(strength, center, inner_radius, radius,
          filter_area, v_texture_coord, MAX_KERNEL_SIZE)
