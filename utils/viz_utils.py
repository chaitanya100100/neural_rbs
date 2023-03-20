import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
import torch
from utils.renderer import MeshViewer, colors
import trimesh
import warnings


def dcn(a):
    """Convert a torch tensor to numpy array."""
    if isinstance(a, np.ndarray):
        return a
    return a.detach().cpu().numpy()


def hstack_images(imgs, channel_first=True):
    if imgs.ndim == 5:
        return np.stack([hstack_images(im) for im in imgs], 0)
    if not channel_first:
        # imgs B x H x W x C
        b, h, w, c = imgs.shape
        imgs = np.transpose(imgs, [1, 0, 2, 3]) # H x B x W x C
        imgs = np.reshape(imgs, [h, -1, c])
    else:
        # imgs B x C x H x W
        b, c, h, w = imgs.shape
        imgs = np.transpose(imgs, [1, 2, 0, 3]) # C x H x B x W
        imgs = np.reshape(imgs, [c, h, -1])
    return imgs


def gridimg(array, nrows=3):
    """Get a single grid image from a batch of images."""
    # array: B x H x W x C
    nindex, height, width, intensity = array.shape
    ncols = nindex//nrows
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result

def imshowreal(img):
    """plt.imshow with roughly real image size."""
    dpi = matplotlib.rcParams['figure.dpi']

    height, width, depth = img.shape
    figsize = 1.5*width / float(dpi), 1.5*height / float(dpi)

    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def plot_sequence_images(image_array, fps=30):
    ''' Display images sequence as an animation in jupyter notebook
    
    Args:
        image_array(numpy.ndarray): image_array.shape equal to (num_images, height, width, num_channels)
    '''
    from matplotlib import animation
    from IPython.display import display, HTML
    dpi = 72.0
    xpixels, ypixels = image_array[0].shape[:2]
    fig = plt.figure(figsize=(ypixels/dpi, xpixels/dpi), dpi=dpi)
    im = plt.figimage(image_array[0])

    def animate(i):
        im.set_array(image_array[i])
        return (im,)

    anim = animation.FuncAnimation(fig, animate, frames=len(image_array), interval=1000/fps, repeat_delay=1, repeat=True)
    display(HTML(anim.to_html5_video()))


def save_video(imgs, vid_path, fps):
    """Save a sequence of images as a video."""
    height, width = imgs.shape[1:3]
    # fourcc = cv2.VideoWriter_fourcc(*'avc1')
    # fourcc = cv2.VideoWriter_fourcc(*'H264')
    # fourcc = -1
    if vid_path.endswith('.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    elif vid_path.endswith('.webm'):
        fourcc = cv2.VideoWriter_fourcc(*'VP90')
    video = cv2.VideoWriter(vid_path, fourcc, fps, (width, height))
    for img in imgs:
        video.write(img[:,:,[2, 1, 0]])
    cv2.destroyAllWindows()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="OpenCV: FFMPEG: tag 0x30395056/'VP90' is not supported with codec id 167 and format 'webm / WebM'")
        video.release()


def put_text_on_img(img, text):
    font=cv2.FONT_HERSHEY_COMPLEX_SMALL
    pos=(4, 4)
    font_scale=1
    font_thickness=1
    text_color=(255, 255, 255)
    text_color_bg=(0, 0, 0)
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    img = cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    img = cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    return img
    

def put_text_on_vid(vid, text):
    for i in range(len(vid)):
        vid[i] = put_text_on_img(vid[i], text)
    return vid


def viz_points(pts, instance_idx=None, lines=None, cam_pos=[0.,1.25,3.], cam_lookat=[0.,0.,0.], cam_up=[0.,1.,0.], add_floor=False, add_axis=False, img_size=512):
    """A helper function to visualize points.
    Args:
        pts: (V, 3) or (N, V, 3) or a list of (V, 3)
        instance_idx: (O+1) or a list of (O+1). If provided, the points will be colored according to the instance index.
        lines: (R, 2) or a list of (R, 2). If provided, the lines will be drawn.
        Other args self-explanatory.
    Returns:
        img: (H, W, 3) or (N, H, W, 3)
    """
    if 'PYOPENGL_PLATFORM' not in os.environ:
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
    if 'DISPLAY' not in os.environ:
        os.environ['DISPLAY'] = ':10'

    if isinstance(pts, list) and isinstance(instance_idx, list) and pts[0].ndim == 2:
        if lines is None: lines = [None] * len(pts)
        if instance_idx is None: instance_idx = [None] * len(pts)
        imgs = [viz_points(pt, istidx, ln, cam_pos, cam_lookat, cam_up, add_floor, add_axis, img_size) for pt, istidx, ln in zip(pts, instance_idx, lines)]
        return np.stack(imgs)

    # pts: V x 3 or N x V x 3
    # instance_idx: O+1
    # lines: R x 2
    if pts.ndim == 2:
        pts = pts[None]
    if lines is not None and not isinstance(lines, list) and lines.ndim == 2:
        lines = lines[None]
    if instance_idx is None:
        instance_idx = [0, pts.shape[1]]
    colnames = ['yellow', 'red'] + [c for c in colors.keys() if c not in ['yellow', 'red']]

    mv = MeshViewer(use_offscreen=True, width=img_size, height=img_size, cam_pos=cam_pos, cam_lookat=cam_lookat, cam_up=cam_up, add_axis=add_axis, add_floor=add_floor)
    for oi, (st, en) in enumerate(zip(instance_idx[:-1], instance_idx[1:])):
        mv.add_point_seq(pts[:, st:en, :3], color=colors[colnames[oi % len(colnames)]])
        if lines is not None:
            ln_seq = [np.concatenate([pt[ln[:, 0], :3], pt[ln[:, 1], :3]], -1).reshape([-1, 3]) for ln, pt in zip(lines, pts)]  # N x [R x 6]
            mv.add_line_seq(ln_seq)

    if pts.shape[0] == 1:
        return mv.render()
    else:
        return mv.animate()


def print_dict(d, prefix=""):
    for k, v in d.items():
        kk = prefix + k
        if isinstance(v, dict):
            print_dict(v, kk+"/")
        elif isinstance(v, int) or isinstance(v, float) or isinstance(v, str):
            print(kk, v)
        elif isinstance(v, list):
            print(kk, len(v))
        else:
            print(kk, v.shape)


def append_to_file(fpath, txt):
    with open(fpath, "a") as myfile:
        myfile.write(txt)


def trimesh_spheres_from_points(pt, color):
    color = np.array(color)
    if len(color.shape) == 1:
        color = np.tile(color[None, :], [pt.shape[0], 1])

    sm = trimesh.creation.uv_sphere(radius=0.05, count=[3, 3])
    ret = []
    for pid, p in enumerate(pt):     
        x = sm.copy()
        x.visual.vertex_colors = color[pid]
        x.vertices += p[None, :]
        ret.append(x)
    ret = trimesh.util.concatenate(ret)
    return ret


def load_binvox(filename):
    with open(filename, 'rb') as fpt:
        vox = trimesh.exchange.binvox.load_binvox(fpt)
    return vox


def trimesh_show_points(pts, col):
    s = trimesh.Scene()
    s.add_geometry(trimesh.creation.axis())
    s.add_geometry(trimesh_spheres_from_points(pts, col))
    s.show()
    return s