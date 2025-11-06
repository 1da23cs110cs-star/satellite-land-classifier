# app.py
import os
import base64
import io
import numpy as np
import cv2
from flask import Flask, request, render_template, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64 MB limit

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return "Upload too large. Please use a smaller crop or lower resolution.", 413

def decode_uploaded_image(filebytes):
    arr = np.frombuffer(filebytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def make_data_url_from_bgr(bgr):
    _, buf = cv2.imencode('.png', bgr)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode('ascii')

# ESA WorldCover representative palette (RGB). We'll convert to BGR for OpenCV.
ESA_PALETTE = {
    10: ("tree_cover",   (0, 100,   0)),   # #006400
    20: ("shrubs",       (255,187, 34)),   # #FFBB22
    30: ("grassland",    (255,255, 76)),   # #FFFF4C
    40: ("cropland",     (240,150,255)),   # #F096FF
    50: ("built_up",     (255,   0,   0)), # #FF0000 (urban)
    60: ("bare_soil",    (180,180,180)),   # #B4B4B4
    70: ("snow_ice",     (220,220,220)),   # #DCDCDC
    80: ("water",        (0,   0, 255)),   # #0000FF
    90: ("wetland",      (0, 150, 120)),   # #009678
    95: ("mangroves",    (0,  64,   0)),   # #004000
    100:("moss_lichen",  (255,230,164)),   # #FFE6A4
}

CLASS_KEYS = list(ESA_PALETTE.keys())
CLASS_NAMES = {k: ESA_PALETTE[k][0] for k in CLASS_KEYS}
CLASS_RGB = {k: ESA_PALETTE[k][1] for k in CLASS_KEYS}
CLASS_BGR = {k: np.array([rgb[2], rgb[1], rgb[0]], dtype=np.int32) for k, rgb in CLASS_RGB.items()}

def classify_all_esa_colors(bgr_img):
    """Assign each pixel to nearest ESA palette color (no threshold)."""
    h, w = bgr_img.shape[:2]
    flat = bgr_img.reshape((-1, 3)).astype(np.int32)  # Nx3 BGR
    keys = CLASS_KEYS
    centers = np.stack([CLASS_BGR[k] for k in keys], axis=0)  # Kx3

    # compute squared distances NxK
    dists = np.sum((flat[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    nearest_idx = np.argmin(dists, axis=1)  # Nx

    label_idx_img = nearest_idx.reshape((h, w))

    masks = {}
    for i, k in enumerate(keys):
        cname = CLASS_NAMES[k]
        masks[cname] = (label_idx_img == i)

    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

    order = []
    for k in keys:
        name = CLASS_NAMES[k]
        rgb = CLASS_RGB[k]
        hexc = rgb_to_hex(rgb)
        order.append((k, name, hexc))
    return masks, order

def masks_to_counts_percentages(masks):
    h, w = next(iter(masks.values())).shape
    total = h * w
    counts = {k: int(np.count_nonzero(v)) for k, v in masks.items()}
    percents = {k: round(100.0 * counts[k] / total, 2) for k in counts}
    return counts, percents

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/upload_blob', methods=['POST'])
def upload_blob():
    f = request.files.get('imagefile')
    if not f:
        return "No file uploaded (expected field 'imagefile').", 400
    filebytes = f.read()
    bgr = decode_uploaded_image(filebytes)
    if bgr is None:
        return "Unable to decode uploaded image.", 400

    # optional bounding to speed/limit
    max_dim = 2048
    h, w = bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    masks, order = classify_all_esa_colors(bgr)
    counts, percents = masks_to_counts_percentages(masks)
    orig_url = make_data_url_from_bgr(bgr)

    results = []
    for class_id, class_name, hexc in order:
        results.append({
            'id': class_id,
            'label': class_name,
            'hex': hexc,
            'count': counts[class_name],
            'percent': percents[class_name]
        })

    return render_template('result.html', original_image=orig_url, results=results)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
