# app.py
import os
import base64
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
    """Decode bytes to an OpenCV BGR image (uint8)."""
    arr = np.frombuffer(filebytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    return img

def make_data_url_from_bgr(bgr):
    """Return a data URL PNG from a BGR image (cv2)."""
    _, buf = cv2.imencode('.png', bgr)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode('ascii')

# ESA WorldCover representative palette (RGB)
ESA_PALETTE = {
    10: ("tree_cover",   (0, 100,   0)),   # #006400
    20: ("shrubs",       (255,187, 34)),   # #FFBB22
    30: ("grassland",    (255,255, 76)),   # #FFFF4C
    40: ("cropland",     (240,150,255)),   # #F096FF
    50: ("built_up",     (255,   0,   0)), # #FF0000
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
# palette as numpy array in RGB order (K,3), dtype=int32 for safe distance compute
PALETTE_ARRAY = np.array([CLASS_RGB[k] for k in CLASS_KEYS], dtype=np.int32)

def classify_all_esa_colors(bgr_img):
    """
    Robust classification:
      1) Convert BGR->RGB
      2) Fast exact-match pass using integer packing (R<<16|G<<8|B).
      3) For pixels not exactly matched, compute nearest palette color (vectorized).
    Returns:
      - masks: dict mapping class_name -> boolean 2D mask
      - order: list of (class_id, class_name, hex_color) in palette order
    """
    # Convert to RGB
    rgb = bgr_img[:, :, ::-1]  # BGR -> RGB
    h, w = rgb.shape[:2]
    flat = rgb.reshape(-1, 3).astype(np.int32)  # Nx3

    # 1) exact-match fast path using packed ints
    # pack color into uint32: R<<16 | G<<8 | B
    packed = (flat[:,0].astype(np.uint32) << 16) | (flat[:,1].astype(np.uint32) << 8) | flat[:,2].astype(np.uint32)

    # prepare mapping from palette packed value -> palette index
    palette_packed = ((PALETTE_ARRAY[:,0].astype(np.uint32) << 16) |
                      (PALETTE_ARRAY[:,1].astype(np.uint32) << 8) |
                       PALETTE_ARRAY[:,2].astype(np.uint32))
    packed_to_index = {int(palette_packed[i]): i for i in range(len(palette_packed))}

    # try exact lookup
    exact_idx = np.full(packed.shape[0], -1, dtype=np.int32)
    # vectorized: create boolean mask of packed values that are in palette
    # fallback to python loop for mapping (fast enough for palette size <= 12)
    for val, idx in packed_to_index.items():
        matches = (packed == val)
        if np.any(matches):
            exact_idx[matches] = idx

    # find which positions still unmatched
    unmatched_mask = (exact_idx == -1)
    unmatched_indices = np.nonzero(unmatched_mask)[0]

    # 2) nearest-color for unmatched pixels (if any)
    if unmatched_indices.size > 0:
        # take only unmatched pixel colors
        unmatched_colors = flat[unmatched_indices].astype(np.int32)  # Mx3
        # compute squared distances from each unmatched pixel to each palette color (MxK)
        # Use broadcasting: (M,1,3) - (1,K,3) -> (M,K,3) then sum squares over axis=2
        diffs = unmatched_colors[:, None, :] - PALETTE_ARRAY[None, :, :]  # MxKx3
        d2 = np.sum(diffs * diffs, axis=2)  # MxK
        nearest = np.argmin(d2, axis=1)  # M
        exact_idx[unmatched_indices] = nearest

    # now exact_idx is Nx palette_index
    label_idx_img = exact_idx.reshape((h, w))  # HxW

    # build boolean masks
    masks = {}
    for i, k in enumerate(CLASS_KEYS):
        cname = CLASS_NAMES[k]
        masks[cname] = (label_idx_img == i)

    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

    order = []
    for k in CLASS_KEYS:
        name = CLASS_NAMES[k]
        rgbc = CLASS_RGB[k]
        hexc = rgb_to_hex(rgbc)
        order.append((k, name, hexc))
    return masks, order

def masks_to_counts_percentages(masks):
    """Return counts and percentages (percent rounded to 2 decimals)."""
    h, w = next(iter(masks.values())).shape
    total = int(h) * int(w)
    counts = {k: int(np.count_nonzero(v)) for k, v in masks.items()}
    # Protect against division by zero (shouldn't happen)
    if total == 0:
        percents = {k: 0.0 for k in counts}
    else:
        percents = {k: round(100.0 * counts[k] / total, 2) for k in counts}
    return counts, percents

@app.route('/')
def index():
    # serve your existing index.html in static folder
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
    max_dim = 4096  # raise a bit if you want larger images; keep safe limits in deployment
    h, w = bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    masks, order = classify_all_esa_colors(bgr)
    counts, percents = masks_to_counts_percentages(masks)
    orig_url = make_data_url_from_bgr(bgr)

    # Prepare results in the order of palette
    results = []
    for class_id, class_name, hexc in order:
        results.append({
            'id': class_id,
            'label': class_name,
            'hex': hexc,
            'count': counts[class_name],
            'percent': percents[class_name]
        })

    # Optionally sort results by percent descending before showing (commented out — leave natural order)
    # results = sorted(results, key=lambda r: r['percent'], reverse=True)

    return render_template('result.html', original_image=orig_url, results=results)

if __name__ == '__main__':
    # use debug=False when deploying
    app.run(host='127.0.0.1', port=5000, debug=True)
