import os, time, random, shutil
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

IMG_W, IMG_H = 640, 640
GRID_N = 3
FACES_PER_IMAGE = GRID_N ** 2
NUM_IMAGES = 20

def resolve_project_root():
    cur = os.getcwd()
    for _ in range(8):
        if os.path.isdir(os.path.join(cur, 'data', 'class_images')):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise FileNotFoundError(f"Could not find data/class_images from cwd={os.getcwd()}")

ROOT = resolve_project_root()
FACE_POOL_DIR = os.path.join(ROOT, 'data', 'class_images')
OUTPUT_DIR = os.path.join(ROOT, 'data', 'synthetic_grid')
print(f"Project root : {ROOT}")
print(f"Face pool    : {FACE_POOL_DIR}")
print(f"Output dir   : {OUTPUT_DIR}")

identity_pools = {}
for name in sorted(os.listdir(FACE_POOL_DIR)):
    class_dir = os.path.join(FACE_POOL_DIR, name)
    if not os.path.isdir(class_dir):
        continue
    paths = sorted(
        os.path.join(class_dir, f)
        for f in os.listdir(class_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    if paths:
        identity_pools[name] = paths

CLASS_NAMES = sorted(identity_pools.keys(), key=lambda x: int(x) if x.isdigit() else x)
name_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)
total_faces = sum(len(v) for v in identity_pools.values())
print(f"Loaded face pool ({total_faces} faces, {NUM_CLASSES} identities):")
for name in CLASS_NAMES:
    print(f"  class {name_to_idx[name]:2d}  ID {name:<6s}  {len(identity_pools[name]):>3d} faces")
assert NUM_CLASSES >= FACES_PER_IMAGE, f"Need >= {FACES_PER_IMAGE} identities, found {NUM_CLASSES}"

def linear_gradient(w, h, c1, c2):
    t = np.linspace(0.0, 1.0, h)[:, None]
    t = np.broadcast_to(t, (h, w))
    c1 = np.asarray(c1, float).reshape(1, 1, 3)
    c2 = np.asarray(c2, float).reshape(1, 1, 3)
    arr = c1 * (1.0 - t[..., None]) + c2 * t[..., None]
    return Image.fromarray(arr.astype(np.uint8), 'RGB')

def bg_gradient(w, h):
    c1 = [random.randint(25, 130) for _ in range(3)]
    c2 = [random.randint(25, 130) for _ in range(3)]
    return linear_gradient(w, h, c1, c2)

def bg_noise(w, h):
    arr = np.random.randint(50, 160, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, 'RGB').filter(ImageFilter.GaussianBlur(radius=random.randint(20, 60)))

def generate_background(w, h):
    r = random.random()
    if r < 0.50:
        return bg_gradient(w, h)
    return bg_noise(w, h)

def prepare_face(img_path, cell_w, cell_h, used_paths):
    if img_path in used_paths:
        return None
    used_paths.add(img_path)
    face = Image.open(img_path).convert('RGB')
    if random.random() < 0.5:
        face = face.transpose(Image.FLIP_LEFT_RIGHT)
    face = ImageEnhance.Brightness(face).enhance(random.uniform(0.85, 1.15))
    scale = random.uniform(0.80, 0.95)
    target_w = int(cell_w * scale)
    target_h = int(cell_h * scale)
    src_ar = face.width / face.height
    tgt_ar = target_w / target_h
    if src_ar > tgt_ar:
        new_w = int(face.height * tgt_ar)
        left = (face.width - new_w) // 2
        face = face.crop((left, 0, left + new_w, face.height))
    else:
        new_h = int(face.width / tgt_ar)
        top = (face.height - new_h) // 2
        face = face.crop((0, top, face.width, top + new_h))
    face = face.resize((target_w, target_h), Image.BICUBIC)
    return face

def compose_grid(used_paths):
    all_ids = list(identity_pools.keys())
    chosen_ids = random.sample(all_ids, FACES_PER_IMAGE)
    canvas = generate_background(IMG_W, IMG_H)
    cell_w = IMG_W // GRID_N
    cell_h = IMG_H // GRID_N
    yolo_boxes = []
    for cell_idx, ident in enumerate(chosen_ids):
        row, col = divmod(cell_idx, GRID_N)
        pool = identity_pools[ident]
        available = [p for p in pool if p not in used_paths]
        if not available:
            available = pool
        img_path = random.choice(available)
        face = prepare_face(img_path, cell_w, cell_h, used_paths)
        if face is None:
            continue
        jitter_x = random.randint(-cell_w // 20, cell_w // 20)
        jitter_y = random.randint(-cell_h // 20, cell_h // 20)
        paste_x = col * cell_w + (cell_w - face.width) // 2 + jitter_x
        paste_y = row * cell_h + (cell_h - face.height) // 2 + jitter_y
        paste_x = max(col * cell_w, min(paste_x, (col + 1) * cell_w - face.width))
        paste_y = max(row * cell_h, min(paste_y, (row + 1) * cell_h - face.height))
        canvas.paste(face, (paste_x, paste_y))
        cx = (paste_x + face.width / 2.0) / IMG_W
        cy = (paste_y + face.height / 2.0) / IMG_H
        bw = face.width / IMG_W
        bh = face.height / IMG_H
        cls_idx = name_to_idx[ident]
        yolo_boxes.append((cls_idx, cx, cy, bw, bh))
    return canvas, yolo_boxes, chosen_ids

# Task 4: Generate
print("\n--- Starting generation ---")
if os.path.isdir(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Cleared {OUTPUT_DIR}")

staging = os.path.join(OUTPUT_DIR, 'staging')
os.makedirs(os.path.join(staging, 'images'), exist_ok=True)
os.makedirs(os.path.join(staging, 'labels'), exist_ok=True)

used_paths = set()
manifest = []

t0 = time.time()
for img_idx in range(NUM_IMAGES):
    canvas, yolo_boxes, chosen_ids = compose_grid(used_paths)
    stem = f"grid_{img_idx:04d}"
    img_path = os.path.join(staging, 'images', stem + '.jpg')
    lbl_path = os.path.join(staging, 'labels', stem + '.txt')
    canvas.save(img_path, quality=90)
    with open(lbl_path, 'w') as f:
        for (ci, cx, cy, bw, bh) in yolo_boxes:
            f.write(f"{ci} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    manifest.append({
        'index': img_idx,
        'image': img_path,
        'label': lbl_path,
        'n_faces': len(yolo_boxes),
        'classes': [CLASS_NAMES[b[0]] for b in yolo_boxes],
    })
    if (img_idx + 1) % 5 == 0:
        print(f"  generated {img_idx + 1} / {NUM_IMAGES}")

elapsed = time.time() - t0
print(f"Generated {len(manifest)} grid images -> {staging}")
print(f"Time: {elapsed:.1f}s")
print("\nSUCCESS - no errors")

