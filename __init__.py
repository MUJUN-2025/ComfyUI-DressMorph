import os
import math
import cv2
import numpy as np
import torch
import folder_paths

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None

try:
    import onnxruntime as ort
except Exception as exc:
    ort = None
    _ORT_IMPORT_ERROR = exc
else:
    _ORT_IMPORT_ERROR = None

_SESSION = None
_MODEL_PATH = None


def _resolve_input_path(name):
    name = os.path.expanduser(str(name).strip())
    if os.path.isabs(name) and os.path.isfile(name):
        return name
    candidate = os.path.join(folder_paths.get_input_directory(), name)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f"找不到视频：{name}。相对路径会从 ComfyUI/input 查找。")


def _to_uint8(images):
    return np.clip(images.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)


def _to_tensor(images):
    return torch.from_numpy(images.astype(np.float32) / 255.0)


def _ease_out_cubic(t):
    t = max(0.0, min(1.0, float(t)))
    return 1.0 - (1.0 - t) ** 3


def _smoothstep(t):
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def _motion_kernel(length, dx, dy):
    length = max(1, int(length))
    if length % 2 == 0:
        length += 1
    kernel = np.zeros((length, length), np.float32)
    center = (length - 1) / 2.0
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        kernel[length // 2, :] = 1.0
    else:
        ux, uy = dx / norm, dy / norm
        x1, y1 = int(round(center - ux * center)), int(round(center - uy * center))
        x2, y2 = int(round(center + ux * center)), int(round(center + uy * center))
        cv2.line(kernel, (x1, y1), (x2, y2), 1.0, 1)
    s = float(kernel.sum())
    return kernel / s if s > 0 else kernel


class DressMorphVideoLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("STRING", {"default": "dm_outfit_test/look_A.mp4"}),
            "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            "width": ("INT", {"default": 720, "min": 64, "max": 4096, "step": 2}),
            "height": ("INT", {"default": 1280, "min": 64, "max": 4096, "step": 2}),
        }}

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("frames", "fps")
    FUNCTION = "load"
    CATEGORY = "DressMorph/换装转场"

    def load(self, video, frame_load_cap, width, height):
        path = _resolve_input_path(video)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频：{path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or not math.isfinite(fps):
            fps = 30.0
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
            frames.append(frame)
            if frame_load_cap > 0 and len(frames) >= frame_load_cap:
                break
        cap.release()
        if not frames:
            raise RuntimeError(f"视频没有读取到任何帧：{path}")
        return (_to_tensor(np.stack(frames)), fps)


class DressMorphStickerFly:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video_a": ("IMAGE",),
            "video_b": ("IMAGE",),
            "sticker": ("IMAGE",),
            "sticker_mask": ("MASK",),
            "transition_frames": ("INT", {"default": 8, "min": 4, "max": 60, "step": 1}),
            "cut_frame": ("INT", {"default": 5, "min": 1, "max": 59, "step": 1}),
            "start_x": ("FLOAT", {"default": 0.82, "min": -0.5, "max": 1.5, "step": 0.01}),
            "start_y": ("FLOAT", {"default": 0.20, "min": -0.5, "max": 1.5, "step": 0.01}),
            "end_x": ("FLOAT", {"default": 0.50, "min": -0.5, "max": 1.5, "step": 0.01}),
            "end_y": ("FLOAT", {"default": 0.52, "min": -0.5, "max": 1.5, "step": 0.01}),
            "start_scale": ("FLOAT", {"default": 0.24, "min": 0.02, "max": 3.0, "step": 0.01}),
            "peak_scale": ("FLOAT", {"default": 1.08, "min": 0.02, "max": 3.0, "step": 0.01}),
            "end_scale": ("FLOAT", {"default": 1.00, "min": 0.02, "max": 3.0, "step": 0.01}),
            "start_rotation": ("FLOAT", {"default": -4.0, "min": -180.0, "max": 180.0, "step": 0.5}),
            "motion_blur": ("INT", {"default": 7, "min": 0, "max": 51, "step": 1}),
            "invert_loadimage_mask": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "compose"
    CATEGORY = "DressMorph/换装转场"

    def compose(self, video_a, video_b, sticker, sticker_mask, transition_frames,
                cut_frame, start_x, start_y, end_x, end_y, start_scale,
                peak_scale, end_scale, start_rotation, motion_blur,
                invert_loadimage_mask):
        a = _to_uint8(video_a)
        b = _to_uint8(video_b)
        h, w = a.shape[1:3]
        if b.shape[1] != h or b.shape[2] != w:
            b = np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_LANCZOS4) for x in b])

        rgb = _to_uint8(sticker[:1])[0]
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
        sm = sticker_mask.detach().cpu().numpy()
        if sm.ndim == 3:
            sm = sm[0]
        if sm.shape != (h, w):
            sm = cv2.resize(sm.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        alpha = (1.0 - sm) if invert_loadimage_mask else sm
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

        n = max(4, int(transition_frames))
        cut = min(max(1, int(cut_frame)), n - 1)
        cut = min(cut, len(a))
        b_used = n - cut
        if len(b) < b_used:
            pad = np.repeat(b[-1:], b_used - len(b), axis=0)
            b = np.concatenate([b, pad], axis=0)

        bases = list(a[-cut:]) + list(b[:b_used])
        premul = rgb.astype(np.float32) * alpha[..., None]
        src_center = (w / 2.0, h / 2.0)
        trans = []
        prev_cx, prev_cy = start_x * w, start_y * h

        for i in range(n):
            if i <= cut:
                t = _ease_out_cubic(i / max(1, cut))
                scale = start_scale + (peak_scale - start_scale) * t
                opacity = 1.0
            else:
                t2 = _smoothstep((i - cut) / max(1, n - 1 - cut))
                scale = peak_scale + (end_scale - peak_scale) * t2
                opacity = 1.0 - t2
            move_t = _ease_out_cubic(i / max(1, cut)) if i <= cut else 1.0
            cx = (start_x + (end_x - start_x) * move_t) * w
            cy = (start_y + (end_y - start_y) * move_t) * h
            angle = start_rotation * (1.0 - move_t)

            matrix = cv2.getRotationMatrix2D(src_center, angle, scale)
            matrix[0, 2] += cx - src_center[0]
            matrix[1, 2] += cy - src_center[1]
            warped_rgb = cv2.warpAffine(premul, matrix, (w, h), flags=cv2.INTER_LANCZOS4,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            warped_a = cv2.warpAffine(alpha, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            if motion_blur > 1 and 0 < i < cut + 1:
                strength = int(round(motion_blur * math.sin(math.pi * i / max(1, cut + 1))))
                if strength > 1:
                    kernel = _motion_kernel(strength, cx - prev_cx, cy - prev_cy)
                    warped_rgb = cv2.filter2D(warped_rgb, -1, kernel)
                    warped_a = cv2.filter2D(warped_a, -1, kernel)
            prev_cx, prev_cy = cx, cy

            aa = np.clip(warped_a * opacity, 0.0, 1.0)
            base = bases[i].astype(np.float32)
            # warped_rgb is premultiplied by its original alpha.
            fg = warped_rgb * opacity
            out = fg + base * (1.0 - aa[..., None])
            trans.append(np.clip(out, 0, 255).astype(np.uint8))

        pre = a[:-cut] if cut < len(a) else a[:0]
        post = b[b_used:] if b_used < len(b) else b[:0]
        result = np.concatenate([pre, np.stack(trans), post], axis=0)
        return (_to_tensor(result),)



def _composite_premultiplied(base, premul_rgb, alpha, opacity=1.0):
    aa = np.clip(alpha * float(opacity), 0.0, 1.0)
    fg = premul_rgb * float(opacity)
    return np.clip(fg + base.astype(np.float32) * (1.0 - aa[..., None]), 0, 255).astype(np.uint8)


def _render_object(obj_premul, obj_alpha, canvas_w, canvas_h, cx, cy, target_height, angle=0.0):
    oh, ow = obj_alpha.shape
    scale = max(1.0, float(target_height)) / max(1.0, float(oh))
    matrix = cv2.getRotationMatrix2D((ow / 2.0, oh / 2.0), float(angle), scale)
    matrix[0, 2] += float(cx) - ow / 2.0
    matrix[1, 2] += float(cy) - oh / 2.0
    rgb = cv2.warpAffine(obj_premul, matrix, (canvas_w, canvas_h), flags=cv2.INTER_LANCZOS4,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = cv2.warpAffine(obj_alpha, matrix, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rgb, np.clip(alpha, 0.0, 1.0)


def _render_object_bbox(obj_premul, obj_alpha, canvas_w, canvas_h,
                        target_x, target_y, target_w, target_h,
                        content_x=0.0, content_y=0.0, content_w=None, content_h=None):
    """Render sticker so its 'content' bbox lands on (target_x, target_y, target_w, target_h).

    The content bbox is the region inside the sticker (in sticker image coords) that
    must align with the target.  Sticker pixels outside the content bbox — typically
    the white outline — extend beyond the target so the outline ends up *outside*
    the slice rather than eating into it.

    Non-uniform scaling is used so the content bbox is pixel-aligned with the target.
    """
    oh, ow = obj_alpha.shape
    cw = float(content_w) if content_w is not None else float(ow)
    ch = float(content_h) if content_h is not None else float(oh)
    cx = float(content_x)
    cy = float(content_y)
    sx_scale = float(target_w) / float(max(1.0, cw))
    sy_scale = float(target_h) / float(max(1.0, ch))
    # Sticker pixel (px, py) -> canvas (px*sx + offset_x, py*sy + offset_y).
    # We want (cx, cy) -> (target_x, target_y).
    offset_x = float(target_x) - cx * sx_scale
    offset_y = float(target_y) - cy * sy_scale
    matrix = np.array([[sx_scale, 0.0, offset_x],
                       [0.0, sy_scale, offset_y]], dtype=np.float32)
    rgb = cv2.warpAffine(obj_premul, matrix, (canvas_w, canvas_h), flags=cv2.INTER_LANCZOS4,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = cv2.warpAffine(obj_alpha, matrix, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rgb, np.clip(alpha, 0.0, 1.0)


def _cursor_layer(width, height, tip_x, tip_y, size=1.0, opacity=1.0, click_phase=-1.0):
    layer = np.zeros((height, width, 4), np.uint8)
    s = max(0.25, float(size))
    base_pts = np.array([[0, 0], [2, 31], [9, 24], [18, 43], [27, 39],
                         [18, 20], [32, 20]], np.float32) * s
    pts = np.round(base_pts + np.array([tip_x, tip_y], np.float32)).astype(np.int32)
    shadow = pts + np.array([3, 4], np.int32)
    a = int(round(255 * max(0.0, min(1.0, opacity))))
    if a <= 0:
        return layer
    cv2.fillPoly(layer, [shadow], (0, 0, 0, int(a * 0.35)))
    cv2.fillPoly(layer, [pts], (250, 250, 250, a))
    cv2.polylines(layer, [pts], True, (20, 20, 20, a), max(1, int(round(2 * s))), cv2.LINE_AA)
    if click_phase >= 0.0:
        p = max(0.0, min(1.0, float(click_phase)))
        radius = int(round((8 + 28 * p) * s))
        ring_a = int(round(a * (1.0 - p)))
        if ring_a > 0:
            cv2.circle(layer, (int(round(tip_x)), int(round(tip_y))), radius,
                       (255, 255, 255, ring_a), max(1, int(round(3 * s))), cv2.LINE_AA)
    return layer


def _overlay_rgba(base, layer):
    alpha = layer[:, :, 3].astype(np.float32) / 255.0
    return np.clip(layer[:, :, :3].astype(np.float32) * alpha[..., None] +
                   base.astype(np.float32) * (1.0 - alpha[..., None]), 0, 255).astype(np.uint8)


class DressMorphClickSwap:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video_a": ("IMAGE",),
            "video_b": ("IMAGE",),
            "selected_sticker": ("IMAGE",),
            "sticker_mask": ("MASK",),
            "pre_click_frames": ("INT", {"default": 8, "min": 3, "max": 60, "step": 1}),
            "move_frames": ("INT", {"default": 8, "min": 4, "max": 60, "step": 1}),
            "cut_move_frame": ("INT", {"default": 5, "min": 1, "max": 59, "step": 1}),
            "start_x": ("FLOAT", {"default": 0.85, "min": -0.5, "max": 1.5, "step": 0.01}),
            "start_y": ("FLOAT", {"default": 0.18, "min": -0.5, "max": 1.5, "step": 0.01}),
            "end_x": ("FLOAT", {"default": 0.49, "min": -0.5, "max": 1.5, "step": 0.01}),
            "end_y": ("FLOAT", {"default": 0.575, "min": -0.5, "max": 1.5, "step": 0.005}),
            "start_height": ("FLOAT", {"default": 0.28, "min": 0.03, "max": 1.5, "step": 0.01}),
            "peak_height": ("FLOAT", {"default": 0.91, "min": 0.03, "max": 1.5, "step": 0.01}),
            "end_height": ("FLOAT", {"default": 0.87, "min": 0.03, "max": 1.5, "step": 0.01}),
            "start_rotation": ("FLOAT", {"default": -3.0, "min": -180.0, "max": 180.0, "step": 0.5}),
            "motion_blur": ("INT", {"default": 7, "min": 0, "max": 51, "step": 1}),
            "cursor_size": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05}),
            "erase_source_slot": ("BOOLEAN", {"default": True}),
            "invert_loadimage_mask": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "compose"
    CATEGORY = "DressMorph/换装转场"

    def compose(self, video_a, video_b, selected_sticker, sticker_mask,
                pre_click_frames, move_frames, cut_move_frame,
                start_x, start_y, end_x, end_y, start_height,
                peak_height, end_height, start_rotation, motion_blur,
                cursor_size, erase_source_slot, invert_loadimage_mask):
        a = _to_uint8(video_a)
        b = _to_uint8(video_b)
        h, w = a.shape[1:3]
        if b.shape[1] != h or b.shape[2] != w:
            b = np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_LANCZOS4) for x in b])

        rgb = _to_uint8(selected_sticker[:1])[0]
        sm = sticker_mask.detach().cpu().numpy()
        if sm.ndim == 3:
            sm = sm[0]
        if rgb.shape[:2] != sm.shape:
            sm = cv2.resize(sm.astype(np.float32), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        alpha = (1.0 - sm) if invert_loadimage_mask else sm
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

        ys, xs = np.where(alpha > 0.03)
        if len(xs) == 0:
            raise RuntimeError("贴图透明通道为空，请检查 LoadImage 的 MASK 或 invert_loadimage_mask。")
        pad = 3
        x1, x2 = max(0, int(xs.min()) - pad), min(rgb.shape[1], int(xs.max()) + pad + 1)
        y1, y2 = max(0, int(ys.min()) - pad), min(rgb.shape[0], int(ys.max()) + pad + 1)
        obj_rgb = rgb[y1:y2, x1:x2].astype(np.float32)
        obj_alpha = alpha[y1:y2, x1:x2]
        obj_premul = obj_rgb * obj_alpha[..., None]

        pre_n = max(3, int(pre_click_frames))
        move_n = max(4, int(move_frames))
        cut = min(max(1, int(cut_move_frame)), move_n - 1)
        a_needed = min(len(a), pre_n + cut)
        # If A is very short, reduce the cursor hold first, but retain movement frames.
        if a_needed < pre_n + cut:
            pre_n = max(1, a_needed - cut)
        b_needed = move_n - cut
        if len(b) < b_needed:
            b = np.concatenate([b, np.repeat(b[-1:], b_needed - len(b), axis=0)], axis=0)

        bases = list(a[-(pre_n + cut):]) + list(b[:b_needed])
        total_n = pre_n + move_n
        if len(bases) != total_n:
            raise RuntimeError("视频帧数量不足，无法组成点击换装转场。")

        start_cx, start_cy = start_x * w, start_y * h
        end_cx, end_cy = end_x * w, end_y * h
        start_rgb, start_a = _render_object(obj_premul, obj_alpha, w, h,
                                             start_cx, start_cy, start_height * h, 0.0)
        erase_mask = cv2.dilate((start_a > 0.08).astype(np.uint8) * 255,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
        out_frames = []
        prev_cx, prev_cy = start_cx, start_cy

        for i in range(total_n):
            base = bases[i].copy()
            if i < pre_n:
                # Cursor is itself a composited animation layer: appear -> move -> click.
                appear = min(1.0, (i + 1) / 2.0)
                move_t = _smoothstep(max(0.0, (i - 1) / max(1, pre_n - 3)))
                cursor_from_x = start_cx + 0.08 * w
                cursor_from_y = start_cy + 0.025 * h
                cursor_to_x = start_cx + 0.018 * w
                cursor_to_y = start_cy + 0.005 * h
                tip_x = cursor_from_x + (cursor_to_x - cursor_from_x) * move_t
                tip_y = cursor_from_y + (cursor_to_y - cursor_from_y) * move_t
                click_phase = -1.0
                csize = cursor_size
                if i >= pre_n - 2:
                    click_phase = (i - (pre_n - 2)) / max(1, 2)
                    csize *= 0.90 if i == pre_n - 1 else 1.0
                cursor = _cursor_layer(w, h, tip_x, tip_y, csize, appear, click_phase)
                base = _overlay_rgba(base, cursor)
                out_frames.append(base)
                continue

            j = i - pre_n
            if erase_source_slot and j > 0 and j < cut:
                # Remove the baked selected thumbnail only after it starts moving.
                base = cv2.inpaint(base, erase_mask, 4, cv2.INPAINT_TELEA)

            if j <= cut:
                t = _ease_out_cubic(j / max(1, cut))
                cx = start_cx + (end_cx - start_cx) * t
                cy = start_cy + (end_cy - start_cy) * t
                height_px = (start_height + (peak_height - start_height) * t) * h
                angle = start_rotation * (1.0 - t)
                opacity = 1.0
                if j == 0:
                    height_px *= 0.96  # click-down squash
            else:
                t2 = _smoothstep((j - cut) / max(1, move_n - 1 - cut))
                cx, cy = end_cx, end_cy
                height_px = (peak_height + (end_height - peak_height) * t2) * h
                angle = 0.0
                opacity = 1.0 - t2

            warped_rgb, warped_a = _render_object(obj_premul, obj_alpha, w, h,
                                                   cx, cy, height_px, angle)
            if motion_blur > 1 and 0 < j <= cut:
                strength = int(round(motion_blur * math.sin(math.pi * j / max(1, cut + 1))))
                if strength > 1:
                    kernel = _motion_kernel(strength, cx - prev_cx, cy - prev_cy)
                    warped_rgb = cv2.filter2D(warped_rgb, -1, kernel)
                    warped_a = cv2.filter2D(warped_a, -1, kernel)
            prev_cx, prev_cy = cx, cy
            base = _composite_premultiplied(base, warped_rgb, warped_a, opacity)

            # Cursor rebounds and fades during the first two movement frames.
            if j < 3:
                cursor_opacity = max(0.0, 1.0 - j / 2.0)
                cursor = _cursor_layer(w, h, start_cx + 0.018 * w, start_cy + 0.005 * h,
                                       cursor_size * (0.90 + 0.10 * min(1.0, j)), cursor_opacity, -1.0)
                base = _overlay_rgba(base, cursor)
            out_frames.append(base)

        pre_frames = a[:-(pre_n + cut)] if pre_n + cut < len(a) else a[:0]
        post_frames = b[b_needed:] if b_needed < len(b) else b[:0]
        result = np.concatenate([pre_frames, np.stack(out_frames), post_frames], axis=0)
        return (_to_tensor(result),)

def _prepare_sticker_object(sticker, sticker_mask, invert_loadimage_mask, outline_px=0):
    """Return (premul_rgb, alpha, content_bbox).

    `alpha` is the with-outline alpha (used for both rendering and visible bbox crop).
    `content_bbox` is (x, y, w, h) in cropped-sticker coords for the original person
    silhouette (without the white outline), so the caller can align the *person* to
    the slice and let the outline spill outside.

    outline_px must match the LC_Sticker node's outline_px so the erosion radius
    matches the dilation that produced the outline.
    """
    rgb = _to_uint8(sticker[:1])[0]
    sm = sticker_mask.detach().cpu().numpy()
    if sm.ndim == 3:
        sm = sm[0]
    if rgb.shape[:2] != sm.shape:
        sm = cv2.resize(sm.astype(np.float32), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    alpha = (1.0 - sm) if invert_loadimage_mask else sm
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

    # Recover the original person alpha (without outline) by eroding, so the
    # content bbox excludes the white outline.  Erosion inverts LC_Sticker's dilation.
    if outline_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (int(outline_px) * 2 + 1, int(outline_px) * 2 + 1))
        alpha_raw = cv2.erode(alpha, k)
    else:
        alpha_raw = alpha

    ys, xs = np.where(alpha > 0.03)
    if len(xs) == 0:
        raise RuntimeError("贴图透明通道为空，请检查 LoadImage 的 MASK 或 invert_loadimage_mask。")
    pad = 3
    x1, x2 = max(0, int(xs.min()) - pad), min(rgb.shape[1], int(xs.max()) + pad + 1)
    y1, y2 = max(0, int(ys.min()) - pad), min(rgb.shape[0], int(ys.max()) + pad + 1)
    obj_rgb = rgb[y1:y2, x1:x2].astype(np.float32)
    obj_alpha = alpha[y1:y2, x1:x2]
    obj_premul = obj_rgb * obj_alpha[..., None]

    # Content bbox in cropped-sticker coords.
    ys_r, xs_r = np.where(alpha_raw > 0.03)
    if len(xs_r) == 0:
        content_bbox = (0.0, 0.0, float(obj_alpha.shape[1]), float(obj_alpha.shape[0]))
    else:
        cx1 = max(0, int(xs_r.min()) - x1)
        cy1 = max(0, int(ys_r.min()) - y1)
        cx2 = min(obj_alpha.shape[1], int(xs_r.max()) + 1 - x1)
        cy2 = min(obj_alpha.shape[0], int(ys_r.max()) + 1 - y1)
        if cx2 <= cx1 or cy2 <= cy1:
            content_bbox = (0.0, 0.0, float(obj_alpha.shape[1]), float(obj_alpha.shape[0]))
        else:
            content_bbox = (float(cx1), float(cy1), float(cx2 - cx1), float(cy2 - cy1))

    return obj_premul, obj_alpha, content_bbox


def _resize_video_to(video, width, height):
    arr = _to_uint8(video)
    if arr.shape[1] == height and arr.shape[2] == width:
        return arr
    return np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
                     for frame in arr])


def _parse_slot_order(value):
    valid = ("LT", "LM", "LB", "RT", "RM", "RB")
    aliases = {
        "左上": "LT", "左中": "LM", "左下": "LB",
        "右上": "RT", "右中": "RM", "右下": "RB",
    }
    tokens = str(value).replace("，", ",").replace(" ", "").upper().split(",")
    tokens = [aliases.get(x, x) for x in tokens if x]
    if len(tokens) != 6 or set(tokens) != set(valid):
        raise RuntimeError("点击顺序必须包含且只包含 LT,LM,LB,RT,RM,RB 六个槽位。")
    return tokens


class DressMorphSequence:
    """Seven clean clips + six independent stickers -> one stateful six-click outfit sequence."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video_0": ("IMAGE",),
            "video_1": ("IMAGE",),
            "video_2": ("IMAGE",),
            "video_3": ("IMAGE",),
            "video_4": ("IMAGE",),
            "video_5": ("IMAGE",),
            "video_6": ("IMAGE",),
            "sticker_LT": ("IMAGE",), "mask_LT": ("MASK",),
            "sticker_LM": ("IMAGE",), "mask_LM": ("MASK",),
            "sticker_LB": ("IMAGE",), "mask_LB": ("MASK",),
            "sticker_RT": ("IMAGE",), "mask_RT": ("MASK",),
            "sticker_RM": ("IMAGE",), "mask_RM": ("MASK",),
            "sticker_RB": ("IMAGE",), "mask_RB": ("MASK",),
            "click_order": ("STRING", {"default": "LT,LM,LB,RT,RM,RB"}),
            "pre_click_frames": ("INT", {"default": 3, "min": 3, "max": 60, "step": 1}),
            "move_frames": ("INT", {"default": 4, "min": 4, "max": 60, "step": 1}),
            "cut_move_frame": ("INT", {"default": 3, "min": 1, "max": 59, "step": 1}),
            "menu_height": ("FLOAT", {"default": 0.28, "min": 0.05, "max": 0.60, "step": 0.01}),
            "outline_px": ("INT", {"default": 5, "min": 0, "max": 80, "step": 1}),
            "motion_blur": ("INT", {"default": 7, "min": 0, "max": 51, "step": 1}),
            "cursor_size": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05}),
            "invert_loadimage_mask": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "compose"
    CATEGORY = "DressMorph/换装转场"

    def compose(self, video_0, video_1, video_2, video_3, video_4, video_5, video_6,
                sticker_LT, mask_LT, sticker_LM, mask_LM, sticker_LB, mask_LB,
                sticker_RT, mask_RT, sticker_RM, mask_RM, sticker_RB, mask_RB,
                click_order, pre_click_frames, move_frames, cut_move_frame,
                menu_height, outline_px,
                motion_blur, cursor_size, invert_loadimage_mask):
        videos_in = [video_0, video_1, video_2, video_3, video_4, video_5, video_6]
        first = _to_uint8(video_0)
        if len(first) == 0:
            raise RuntimeError("video_0 没有视频帧。")
        h, w = first.shape[1:3]
        videos = [first] + [_resize_video_to(v, w, h) for v in videos_in[1:]]
        if any(len(v) == 0 for v in videos):
            raise RuntimeError("7 段视频都必须至少包含 1 帧。")

        stickers_in = {
            "LT": (sticker_LT, mask_LT), "LM": (sticker_LM, mask_LM),
            "LB": (sticker_LB, mask_LB), "RT": (sticker_RT, mask_RT),
            "RM": (sticker_RM, mask_RM), "RB": (sticker_RB, mask_RB),
        }
        objects = {slot: _prepare_sticker_object(img, mask, invert_loadimage_mask, int(outline_px))
                   for slot, (img, mask) in stickers_in.items()}
        # Coordinates measured from the supplied 720x1280 reference video.
        slots = {
            "LT": (0.12, 0.18), "LM": (0.12, 0.52), "LB": (0.12, 0.83),
            "RT": (0.85, 0.18), "RM": (0.85, 0.53), "RB": (0.85, 0.83),
        }
        order = _parse_slot_order(click_order)
        pre_n = max(3, int(pre_click_frames))
        move_n = max(4, int(move_frames))
        cut = min(max(1, int(cut_move_frame)), move_n - 1)
        b_used = move_n - cut
        menu_h_px = float(menu_height) * h
        active = set(slots.keys())
        output = []

        # Auto-detect each "next" video's central person bbox via U²-Net so the sticker
        # has no XY / end-height knobs to set.  Computed once per unique next-video
        # (videos[1]..videos[6]) and cached.
        slice_cache = {}
        slice_bboxes = []
        for i in range(6):
            nxt_idx = i + 1
            if nxt_idx not in slice_cache:
                bbox = _detect_person_bbox(videos[nxt_idx][0])
                if bbox is None:
                    raise RuntimeError(
                        f"video_{nxt_idx} 第一帧 U²-Net 未检测到清晰人物，"
                        "无法自动确定切片位置。请检查该段视频首帧是否包含完整人物轮廓。"
                    )
                slice_cache[nxt_idx] = bbox
            slice_bboxes.append(slice_cache[nxt_idx])

        def draw_stationary(base, visible_slots):
            out = base
            for slot in ("LT", "LM", "LB", "RT", "RM", "RB"):
                if slot not in visible_slots:
                    continue
                premul, alpha, _content_bbox = objects[slot]
                sx, sy = slots[slot]
                wrgb, wa = _render_object(premul, alpha, w, h, sx * w, sy * h, menu_h_px, 0.0)
                out = _composite_premultiplied(out, wrgb, wa, 1.0)
            return out

        start_offsets = [0] * 7
        for transition_index, selected in enumerate(order):
            current = videos[transition_index]
            nxt = videos[transition_index + 1]
            start = start_offsets[transition_index]
            available = len(current) - start
            needed_a = pre_n + cut
            if available < needed_a:
                raise RuntimeError(
                    f"第 {transition_index + 1} 段视频太短：扣除上一转场后剩 {available} 帧，"
                    f"本次至少需要 {needed_a} 帧。"
                )
            if len(nxt) < b_used:
                nxt = np.concatenate([nxt, np.repeat(nxt[-1:], b_used - len(nxt), axis=0)], axis=0)
                videos[transition_index + 1] = nxt

            regular_end = len(current) - needed_a
            for frame in current[start:regular_end]:
                output.append(draw_stationary(frame.copy(), active))

            bases = list(current[regular_end:]) + list(nxt[:b_used])
            moving_premul, moving_alpha, content_bbox = objects[selected]
            sx, sy = slots[selected]
            start_cx, start_cy = sx * w, sy * h
            sh, sw = moving_alpha.shape
            c_x, c_y, c_w, c_h = content_bbox
            # Menu-state uniform scale (matches draw_stationary's _render_object call).
            menu_scale = menu_h_px / float(max(1, sh))
            # Menu-state content (person silhouette) bbox on canvas.  This is the
            # motion start point so the person — not the outline — is what eases
            # toward the slice.
            start_content_x = start_cx + (c_x + c_w / 2.0 - sw / 2.0) * menu_scale - c_w * menu_scale / 2.0
            start_content_y = start_cy + (c_y + c_h / 2.0 - sh / 2.0) * menu_scale - c_h * menu_scale / 2.0
            start_content_w = c_w * menu_scale
            start_content_h = c_h * menu_scale
            # Account for the j==0 pop in phase 1.
            start_content_w0 = start_content_w * 0.96
            start_content_h0 = start_content_h * 0.96
            start_content_x0 = start_cx + (c_x + c_w / 2.0 - sw / 2.0) * menu_scale - start_content_w0 / 2.0
            start_content_y0 = start_cy + (c_y + c_h / 2.0 - sh / 2.0) * menu_scale - start_content_h0 / 2.0

            # Slice target = the detected person silhouette in the next video.
            slice_x, slice_y, slice_w, slice_h = slice_bboxes[transition_index]
            end_content_x = float(slice_x)
            end_content_y = float(slice_y)
            end_content_w = float(slice_w)
            end_content_h = float(slice_h)
            prev_cx, prev_cy = start_cx, start_cy

            for i, source_frame in enumerate(bases):
                base = source_frame.copy()
                if i < pre_n:
                    # All six/remaining stickers have been visible since the beginning.
                    base = draw_stationary(base, active)
                    appear = min(1.0, (i + 1) / 2.0)
                    move_t = _smoothstep(max(0.0, (i - 1) / max(1, pre_n - 3)))
                    cursor_from_x = min(w - 38.0 * cursor_size, start_cx + 0.10 * w)
                    cursor_from_y = start_cy + 0.025 * h
                    cursor_to_x = min(w - 34.0 * cursor_size, start_cx + 0.018 * w)
                    cursor_to_y = start_cy + 0.005 * h
                    tip_x = cursor_from_x + (cursor_to_x - cursor_from_x) * move_t
                    tip_y = cursor_from_y + (cursor_to_y - cursor_from_y) * move_t
                    click_phase = -1.0
                    csize = cursor_size
                    if i >= pre_n - 2:
                        click_phase = (i - (pre_n - 2)) / 2.0
                        if i == pre_n - 1:
                            csize *= 0.90
                    base = _overlay_rgba(base, _cursor_layer(w, h, tip_x, tip_y, csize, appear, click_phase))
                    output.append(base)
                    continue

                j = i - pre_n
                # Once clicked, selected thumbnail is no longer drawn in its slot.
                base = draw_stationary(base, active - {selected})
                if j <= cut:
                    # Phase 1: sticker's *content* (person silhouette) eases from its
                    # menu-state canvas bbox to the slice bbox.  The outline, being
                    # outside the content bbox, lands outside the slice — exactly
                    # what we want.
                    t = _ease_out_cubic(j / max(1, cut))
                    if j == 0:
                        s_x, s_y, s_w, s_h = start_content_x0, start_content_y0, start_content_w0, start_content_h0
                    else:
                        s_x, s_y, s_w, s_h = start_content_x, start_content_y, start_content_w, start_content_h
                    target_x = s_x + (end_content_x - s_x) * t
                    target_y = s_y + (end_content_y - s_y) * t
                    target_w = s_w + (end_content_w - s_w) * t
                    target_h = s_h + (end_content_h - s_h) * t
                    # Cursor anchor for motion-blur direction (use sticker center).
                    cx = target_x + target_w / 2.0
                    cy = target_y + target_h / 2.0
                    opacity = 1.0
                else:
                    # Phase 2: content locked onto the slice bbox, fades out as the
                    # next video reveals itself underneath.
                    target_x = end_content_x
                    target_y = end_content_y
                    target_w = end_content_w
                    target_h = end_content_h
                    cx = target_x + target_w / 2.0
                    cy = target_y + target_h / 2.0
                    t2 = _smoothstep((j - cut) / max(1, move_n - 1 - cut))
                    opacity = 1.0 - t2

                wrgb, wa = _render_object_bbox(moving_premul, moving_alpha, w, h,
                                               target_x, target_y, target_w, target_h,
                                               content_x=c_x, content_y=c_y,
                                               content_w=c_w, content_h=c_h)
                if motion_blur > 1 and 0 < j <= cut:
                    strength = int(round(motion_blur * math.sin(math.pi * j / max(1, cut + 1))))
                    if strength > 1:
                        kernel = _motion_kernel(strength, cx - prev_cx, cy - prev_cy)
                        wrgb = cv2.filter2D(wrgb, -1, kernel)
                        wa = cv2.filter2D(wa, -1, kernel)
                prev_cx, prev_cy = cx, cy
                base = _composite_premultiplied(base, wrgb, wa, opacity)
                if j < 3:
                    cursor_opacity = max(0.0, 1.0 - j / 2.0)
                    cursor = _cursor_layer(w, h,
                                           min(w - 34.0 * cursor_size, start_cx + 0.018 * w),
                                           start_cy + 0.005 * h,
                                           cursor_size * (0.90 + 0.10 * min(1.0, j)),
                                           cursor_opacity, -1.0)
                    base = _overlay_rgba(base, cursor)
                output.append(base)

            active.remove(selected)
            start_offsets[transition_index + 1] = b_used

        last = videos[6]
        for frame in last[start_offsets[6]:]:
            output.append(draw_stationary(frame.copy(), active))
        return (_to_tensor(np.stack(output)),)


class DressMorphMP4Saver:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 1.0}),
            "filename_prefix": ("STRING", {"default": "DM_outfit_transition"}),
            "crf": ("INT", {"default": 18, "min": 0, "max": 51, "step": 1}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "DressMorph/换装转场"

    def save(self, images, fps, filename_prefix, crf):
        if imageio_ffmpeg is None:
            raise RuntimeError("缺少 imageio-ffmpeg，请在 ComfyUI Python 环境安装 imageio-ffmpeg。")
        arr = _to_uint8(images)
        h, w = arr.shape[1:3]
        out_dir = folder_paths.get_output_directory()
        safe = str(filename_prefix).replace("..", "_").replace("/", "_").replace("\\", "_")
        counter = 1
        while True:
            path = os.path.join(out_dir, f"{safe}_{counter:05d}.mp4")
            if not os.path.exists(path):
                break
            counter += 1
        writer = imageio_ffmpeg.write_frames(
            path, (w, h), fps=float(fps), codec="libx264", pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p", output_params=["-crf", str(int(crf)), "-movflags", "+faststart"]
        )
        writer.send(None)
        try:
            for frame in arr:
                writer.send(np.ascontiguousarray(frame))
        finally:
            writer.close()
        return {"ui": {"text": [path]}, "result": (path,)}


def _find_model():
    candidates = [
        os.path.join(folder_paths.models_dir, "u2net", "u2netp.onnx"),
        os.path.expanduser("~/Documents/ComfyUI/models/u2net/u2netp.onnx"),
        os.path.join(folder_paths.models_dir, "rembg", "u2netp.onnx"),
        os.path.join(os.path.dirname(__file__), "models", "u2netp.onnx"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "没有找到 u2netp.onnx。请放到 ComfyUI/models/u2net/u2netp.onnx"
    )


def _get_session():
    global _SESSION, _MODEL_PATH
    if ort is None:
        raise RuntimeError(f"缺少 onnxruntime：{_ORT_IMPORT_ERROR}")
    path = _find_model()
    if _SESSION is None or _MODEL_PATH != path:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _SESSION = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
        _MODEL_PATH = path
    return _SESSION


def _u2net_mask(rgb_u8):
    height, width = rgb_u8.shape[:2]
    resized = cv2.resize(rgb_u8, (320, 320), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    # Normalization used by the official U-2-Net preprocessing pipeline.
    normalized = np.empty_like(resized, dtype=np.float32)
    normalized[:, :, 0] = (resized[:, :, 0] - 0.485) / 0.229
    normalized[:, :, 1] = (resized[:, :, 1] - 0.456) / 0.224
    normalized[:, :, 2] = (resized[:, :, 2] - 0.406) / 0.225
    tensor = np.transpose(normalized, (2, 0, 1))[None].astype(np.float32)
    session = _get_session()
    pred = session.run(None, {session.get_inputs()[0].name: tensor})[0][0, 0]
    pred_min, pred_max = float(pred.min()), float(pred.max())
    if pred_max - pred_min > 1e-6:
        pred = (pred - pred_min) / (pred_max - pred_min)
    else:
        pred = np.zeros_like(pred, dtype=np.float32)
    return cv2.resize(pred.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)


def _detect_person_bbox(rgb_u8, threshold=0.5, min_area_ratio=0.04):
    """Run U²-Net on a frame and return (x, y, w, h) of the foreground person.

    Returns None if the foreground area is too small or the mask is empty, so callers
    can raise a clear error rather than silently placing the sticker at (0, 0).
    """
    mask = _u2net_mask(rgb_u8)
    binary = (mask > float(threshold))
    if not binary.any():
        return None
    total = rgb_u8.shape[0] * rgb_u8.shape[1]
    if float(binary.sum()) / float(total) < float(min_area_ratio):
        return None
    ys, xs = np.where(binary)
    pad = 2
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(rgb_u8.shape[1], int(xs.max()) + pad + 1)
    y2 = min(rgb_u8.shape[0], int(ys.max()) + pad + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)



def _clean_mask(mask, threshold, feather_px, cleanup_px):
    # Keep a soft transition around the chosen threshold instead of producing a jagged hard edge.
    softness = max(0.015, min(0.20, 0.035 + feather_px * 0.004))
    alpha = np.clip((mask - (threshold - softness)) / (2.0 * softness), 0.0, 1.0)
    if cleanup_px > 0:
        k = max(1, int(cleanup_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1))
        binary = (alpha > 0.50).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        alpha *= cv2.GaussianBlur(binary.astype(np.float32), (0, 0), max(0.5, feather_px * 0.35))
    if feather_px > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), max(0.35, feather_px * 0.45))
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def _make_white_outline(rgb, alpha, outline_px):
    radius = max(0, int(outline_px))
    if radius == 0:
        return rgb.copy(), alpha.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    expanded = cv2.dilate(alpha, kernel)
    expanded = np.clip(expanded, 0.0, 1.0)
    outline = np.clip(expanded - alpha, 0.0, 1.0)
    premul = rgb.astype(np.float32) / 255.0 * alpha[:, :, None]
    premul += outline[:, :, None]  # white outline
    straight = np.zeros_like(premul)
    valid = expanded > 1e-5
    straight[valid] = premul[valid] / expanded[valid, None]
    return np.clip(straight, 0.0, 1.0), expanded


class LiteCutoutSticker:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "threshold": ("FLOAT", {"default": 0.45, "min": 0.05, "max": 0.95, "step": 0.01}),
            "outline_px": ("INT", {"default": 5, "min": 0, "max": 80, "step": 1}),
            "feather_px": ("INT", {"default": 2, "min": 0, "max": 20, "step": 1}),
            "cleanup_px": ("INT", {"default": 1, "min": 0, "max": 12, "step": 1}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("白边人物贴图", "贴图前景MASK", "原始人物MASK")
    FUNCTION = "process"
    CATEGORY = "LiteCutout/抠图描边"

    def process(self, image, threshold, outline_px, feather_px, cleanup_px):
        images = np.clip(image.detach().cpu().numpy(), 0.0, 1.0)
        sticker_images, sticker_masks, raw_masks = [], [], []
        for frame in images:
            rgb = frame[:, :, :3]
            rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
            raw = _u2net_mask(rgb_u8)
            alpha = _clean_mask(raw, float(threshold), int(feather_px), int(cleanup_px))
            sticker_rgb, sticker_alpha = _make_white_outline(rgb_u8, alpha, int(outline_px))
            sticker_images.append(sticker_rgb.astype(np.float32))
            sticker_masks.append(sticker_alpha.astype(np.float32))
            raw_masks.append(alpha.astype(np.float32))
        return (
            torch.from_numpy(np.stack(sticker_images)),
            torch.from_numpy(np.stack(sticker_masks)),
            torch.from_numpy(np.stack(raw_masks)),
        )


NODE_CLASS_MAPPINGS = {
    "DM_VideoLoader": DressMorphVideoLoader,
    "DM_StickerFly": DressMorphStickerFly,
    "DM_ClickSwap": DressMorphClickSwap,
    "DM_Sequence": DressMorphSequence,
    "DM_MP4Saver": DressMorphMP4Saver,
    "LC_Sticker": LiteCutoutSticker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DM_VideoLoader": "DM 视频帧加载",
    "DM_StickerFly": "DM 贴纸飞入换装",
    "DM_ClickSwap": "DM 点击拾取换装",
    "DM_Sequence": "DM 六贴图连续点击",
    "DM_MP4Saver": "DM 导出 MP4",
    "LC_Sticker": "LC 抠图描白边（U²-Net Lite）",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
