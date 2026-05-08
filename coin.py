"""
Egyptian Coin Counter — CV Pipeline Visualizer
================================================
CSE 473 / Computer Vision Project  –  Spring 2025

Detects and classifies Egyptian coins (1 EGP, 50 Pt, 25 Pt) in a photo.
The pipeline:
  1. Load & resize image
  2. HSV-based background suppression
  3. CLAHE contrast enhancement + Gaussian blur
  4. Hough Circle Transform (with contour fallback for cluttered scenes)
  5. IoU-based NMS + per-coin HSV classification

Each stage is visualised as a thumbnail so we can see exactly where the
algorithm succeeds or fails — helpful for debugging and the project demo.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageOps


# ── Display constants ─────────────────────────────────────────────────────────
# Thumbnail size for each pipeline stage panel (keeps the UI compact).
THUMB_W = 300
THUMB_H = 220

# The final result gets a bigger canvas so the coin labels are actually readable.
RESULT_W = 640
RESULT_H = 480

# Hough / detection thresholds — kept as named constants so they're easy to
# find if we ever need to re-tune for a different camera / lighting setup.
HOUGH_DP          = 1.5   # inverse ratio of accumulator resolution
HOUGH_MIN_DIST    = 80    # min distance between circle centres (px)
HOUGH_PARAM1      = 50    # upper Canny threshold
HOUGH_PARAM2      = 55    # accumulator threshold (lower → more false positives)
HOUGH_MIN_RADIUS  = 18    # smallest coin radius we expect (px, after resize)
HOUGH_MAX_RADIUS  = 70    # largest coin radius we expect

IOU_THRESHOLD     = 0.10  # overlap above this → keep only the larger circle

# We only trigger the contour fallback if Hough finds fewer than this many circles.
# Set to 1 so a single missed coin still gets a second chance.
MIN_HOUGH_CIRCLES = 1

# Fallback contour filter thresholds
FALLBACK_MIN_AREA        = 900   # ignore tiny noise blobs
FALLBACK_MIN_CIRCULARITY = 0.55  # coins are round; reject obviously non-circular shapes
FALLBACK_DIST_THRESHOLD  = 0.35  # distance-transform foreground threshold

# HSV background mask — keep pixels that are either saturated OR bright enough.
# This drops flat grey/white backgrounds while keeping metallic coin surfaces.
MASK_SAT_LOW = 30
MASK_VAL_LOW = 100

# Coin classifier thresholds (tuned empirically on our test set).
# 1 EGP has a bimetal design: a high-saturation silver centre inside a gold ring.
BIMETAL_CENTER_RING_DIFF = 30   # centre_sat − ring_sat must exceed this
BIMETAL_RING_SAT_MAX     = 115  # ring must be "not too saturated" (gold-ish)
HIGH_SAT_THRESHOLD       = 85   # 50 Pt is noticeably more saturated than 25 Pt
# ─────────────────────────────────────────────────────────────────────────────


def cv2_to_photoimage(cv2_img, target_w, target_h):
    """
    Convert an OpenCV image (BGR, grayscale, or RGB) into a Tkinter PhotoImage
    that fits within (target_w × target_h), letterboxed with black bars.

    Two things worth noting:
    - We must keep a Python reference to the returned PhotoImage or Tkinter will
      garbage-collect it and the label goes blank (Tkinter quirk).
    - OpenCV is BGR; PIL/Tkinter expects RGB, so we always convert.
    """
    # Convert to RGB so PIL accepts it regardless of input colour space.
    if len(cv2_img.shape) == 2:
        rgb = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    # Scale down (or up) to fit the target box while keeping aspect ratio.
    h, w = rgb.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Paste onto a black canvas of exactly the target size.
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    pil_img = Image.fromarray(canvas)
    return ImageTk.PhotoImage(pil_img)


class StagePanel(tk.Frame):
    """
    One labelled thumbnail in the pipeline strip.
    Wraps a Canvas + title + a one-line caption so each stage panel is
    self-contained and easy to update as processing proceeds.
    """
    def __init__(self, parent, title, width=THUMB_W, height=THUMB_H, **kwargs):
        super().__init__(parent, **kwargs)
        self.img_width  = width
        self.img_height = height

        tk.Label(self, text=title, font=("Helvetica", 9, "bold"),
                 anchor="w").pack(fill=tk.X, padx=4, pady=(4, 1))

        self.canvas = tk.Canvas(self, width=width, height=height,
                                bg="#1a1a1a", highlightthickness=1,
                                highlightbackground="#444444")
        self.canvas.pack(padx=4, pady=2)

        self.caption_var = tk.StringVar(value="—")
        tk.Label(self, textvariable=self.caption_var,
                 font=("Helvetica", 8), fg="#555555",
                 anchor="w", wraplength=width).pack(fill=tk.X, padx=4)

        # Storing _photo here prevents it being garbage-collected by Tkinter.
        self._photo = None

    def update_image(self, cv2_img, caption=""):
        """Swap in a new OpenCV image and update the caption."""
        self._photo = cv2_to_photoimage(cv2_img, self.img_width, self.img_height)
        self.canvas.delete("all")
        self.canvas.create_image(self.img_width  // 2,
                                 self.img_height // 2,
                                 anchor=tk.CENTER, image=self._photo)
        self.caption_var.set(caption)

    def clear(self):
        self.canvas.delete("all")
        self._photo = None
        self.caption_var.set("—")


class CoinCountingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Egyptian Coin Detection — CV Pipeline Visualizer")
        self.root.resizable(True, True)

        # ── Header bar ────────────────────────────────────────────────────────
        header_frame = tk.Frame(root, pady=8)
        header_frame.pack(fill=tk.X)

        tk.Label(header_frame, text="Egyptian Coin Counter",
                 font=("Helvetica", 16, "bold")).pack(side=tk.LEFT, padx=16)

        self.upload_btn = tk.Button(header_frame, text="Upload Image",
                                    command=self.upload_and_process,
                                    font=("Helvetica", 11))
        self.upload_btn.pack(side=tk.LEFT, padx=8)

        self.stats_label = tk.Label(
            header_frame,
            text="Total Value: 0.00 EGP   |   Coins Detected: 0",
            font=("Helvetica", 13), fg="#1a5fb4"
        )
        self.stats_label.pack(side=tk.LEFT, padx=16)

        # ── Pipeline strip (5 stage thumbnails) ──────────────────────────────
        stages_outer = tk.LabelFrame(root, text=" CV Pipeline Stages ",
                                     font=("Helvetica", 9, "bold"),
                                     padx=6, pady=6)
        stages_outer.pack(fill=tk.X, padx=10, pady=(0, 6))

        stages_inner = tk.Frame(stages_outer)
        stages_inner.pack()

        self.stage_original  = StagePanel(stages_inner, "① Original image")
        self.stage_mask      = StagePanel(stages_inner, "② HSV / BG mask")
        self.stage_preproc   = StagePanel(stages_inner, "③ CLAHE + blur")
        self.stage_detection = StagePanel(stages_inner, "④ Raw detection")
        self.stage_final     = StagePanel(stages_inner, "⑤ Classified output")

        for panel in (self.stage_original, self.stage_mask,
                      self.stage_preproc, self.stage_detection,
                      self.stage_final):
            panel.pack(side=tk.LEFT, padx=4, pady=2)

        # ── Large result panel ────────────────────────────────────────────────
        result_outer = tk.LabelFrame(root, text=" Final Result ",
                                     font=("Helvetica", 9, "bold"),
                                     padx=6, pady=6)
        result_outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.canvas = tk.Canvas(result_outer, width=RESULT_W, height=RESULT_H,
                                bg="gray")
        self.canvas.pack()

        self.tk_image = None  # kept as attribute to prevent GC

    # ── Coin classification ───────────────────────────────────────────────────

    def classify_coin(self, hsv_img, x, y, r):
        """
        Classify a detected circle as 1 EGP, 50 Pt, or 25 Pt using the
        saturation difference between the coin's centre and its outer ring.

        Why HSV saturation?
        - 1 EGP is bimetallic: a silver centre (low sat) inside a gold ring
          (higher sat) — so the centre-ring saturation gap is large.
        - 50 Pt is uniformly higher saturation overall (brassy colour).
        - 25 Pt is the dullest / least saturated of the three.

        Splitting the coin into a centre circle and an annular ring lets us
        capture that bimetal pattern without any edge detection.
        """
        # Build masks for the full disk, the inner 60 %, and the annular ring.
        mask_full   = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
        mask_center = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
        cv2.circle(mask_full,   (x, y), r,            255, -1)
        cv2.circle(mask_center, (x, y), int(r * 0.6), 255, -1)
        mask_ring = cv2.bitwise_xor(mask_full, mask_center)

        # Mean HSV for each zone — we only really use the S channel (index 1).
        mean_center = cv2.mean(hsv_img, mask=mask_center)[:3]
        mean_ring   = cv2.mean(hsv_img, mask=mask_ring)[:3]
        mean_full   = cv2.mean(hsv_img, mask=mask_full)[:3]

        center_sat = mean_center[1]
        ring_sat   = mean_ring[1]
        full_sat   = mean_full[1]

        print(f"Coin at X:{x}, Y:{y} -> Center Sat: {center_sat:.1f} | "
              f"Ring Sat: {ring_sat:.1f} | Full Sat: {full_sat:.1f}")

        # 1 EGP: big centre-ring saturation gap + ring isn't too saturated.
        if (center_sat - ring_sat) > BIMETAL_CENTER_RING_DIFF and ring_sat < BIMETAL_RING_SAT_MAX:
            return "1 EGP", 1.0, (255, 0, 0)
        # 50 Pt: noticeably saturated all over (brass/gold tint).
        elif full_sat > HIGH_SAT_THRESHOLD:
            return "50 Pt", 0.5, (0, 200, 255)
        # 25 Pt: everything else (dull silver/nickel).
        else:
            return "25 Pt", 0.25, (0, 0, 0)

    # ── Circle IoU (NMS helper) ───────────────────────────────────────────────

    @staticmethod
    def circle_iou(c1, c2):
        """
        Intersection-over-Union for two circles.
        Used in the NMS step to suppress duplicate detections of the same coin.

        Handles the three geometric cases:
          - No overlap at all → 0.0
          - One circle fully inside the other → area ratio of the smaller
          - Partial overlap → proper lens-area formula
        """
        x1, y1, r1 = float(c1[0]), float(c1[1]), float(c1[2])
        x2, y2, r2 = float(c2[0]), float(c2[1]), float(c2[2])
        dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        if dist >= r1 + r2:
            return 0.0  # circles don't touch

        if dist <= abs(r1 - r2):
            # One is completely inside the other.
            smaller_area = np.pi * min(r1, r2) ** 2
            larger_area  = np.pi * max(r1, r2) ** 2
            return smaller_area / larger_area

        # Partial overlap — standard circle-circle intersection area formula.
        a = r1 ** 2 * np.arccos(np.clip((dist**2 + r1**2 - r2**2) / (2 * dist * r1), -1, 1))
        b = r2 ** 2 * np.arccos(np.clip((dist**2 + r2**2 - r1**2) / (2 * dist * r2), -1, 1))
        c = 0.5 * np.sqrt(max((-dist + r1 + r2) * (dist + r1 - r2) *
                               (dist - r1 + r2) * (dist + r1 + r2), 0))
        intersection = a + b - c
        union = np.pi * r1**2 + np.pi * r2**2 - intersection
        return intersection / union if union > 0 else 0.0

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def upload_and_process(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return

        # Clear stage panels so stale results don't persist between runs.
        for panel in (self.stage_original, self.stage_mask,
                      self.stage_preproc, self.stage_detection,
                      self.stage_final):
            panel.clear()

        # ── Step 0: Load & resize ─────────────────────────────────────────────
        # exif_transpose fixes phones that store images rotated in metadata.
        try:
            pil_image = Image.open(file_path)
            pil_image = ImageOps.exif_transpose(pil_image)
            img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error reading image: {e}")
            return

        # Shrink to fit 640×480 — keeps processing fast and Hough params consistent.
        h, w = img.shape[:2]
        scaling_factor = min(640 / w, 480 / h)
        new_size = (int(w * scaling_factor), int(h * scaling_factor))
        img = cv2.resize(img, new_size)
        output_img = img.copy()  # we'll draw annotations on this later

        # Stage 1: show the original (post-resize) image.
        self.stage_original.update_image(img, caption=f"{new_size[0]}×{new_size[1]} px")
        self.root.update_idletasks()   # flush UI so thumbnails appear incrementally

        # ── Step 1: Colour spaces ─────────────────────────────────────────────
        # HSV is better than RGB for colour-based masks because lighting changes
        # mostly affect the V channel, leaving H and S relatively stable.
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── Step 2: Background suppression via HSV mask ───────────────────────
        # Keep pixels that are either clearly coloured (sat ≥ 30) or bright
        # enough to be a coin surface (val ≥ 100).  This kills flat, dark, or
        # near-greyscale backgrounds that would confuse the circle detector.
        sat = hsv_img[:, :, 1]
        val = hsv_img[:, :, 2]

        coin_mask = cv2.bitwise_or(
            cv2.inRange(sat, MASK_SAT_LOW, 255),
            cv2.inRange(val, MASK_VAL_LOW, 255)
        )

        # Morphological clean-up: opening removes small noise specks,
        # closing fills small holes inside coin regions.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        coin_mask = cv2.morphologyEx(coin_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        coin_mask = cv2.morphologyEx(coin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Stage 2: darken suppressed background so the thumbnail shows what survived.
        tinted = img.copy()
        tinted[coin_mask == 0] = (tinted[coin_mask == 0] * 0.2).astype(np.uint8)
        self.stage_mask.update_image(
            tinted,
            caption=f"sat≥{MASK_SAT_LOW} OR val≥{MASK_VAL_LOW} → morph open+close"
        )
        self.root.update_idletasks()

        # Replace suppressed pixels with mid-grey (128) rather than black so
        # CLAHE and Hough don't create strong edges at the mask boundary.
        gray = np.where(coin_mask > 0, gray, np.uint8(128))

        # ── Step 3: CLAHE + Gaussian blur ─────────────────────────────────────
        # CLAHE boosts local contrast without over-amplifying noise — helps
        # Hough pick up coins that are unevenly lit.
        # The blur smooths out texture so HoughCircles finds the coin boundary
        # rather than internal engraving details.
        clahe        = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq      = clahe.apply(gray)
        gray_blurred = cv2.GaussianBlur(gray_eq, (9, 9), 2)

        # Stage 3: show the contrast-enhanced, blurred greyscale.
        self.stage_preproc.update_image(
            gray_blurred,
            caption="CLAHE (clipLimit=2) + GaussianBlur (9×9, σ=2)"
        )
        self.root.update_idletasks()

        # ── Step 4: Hough Circle Transform ───────────────────────────────────
        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=HOUGH_DP,
            minDist=HOUGH_MIN_DIST,
            param1=HOUGH_PARAM1,
            param2=HOUGH_PARAM2,
            minRadius=HOUGH_MIN_RADIUS,
            maxRadius=HOUGH_MAX_RADIUS
        )

        # ── Step 4b: Contour fallback for overlapping / touching coins ────────
        # Hough sometimes misses coins when they overlap or the image is noisy.
        # The fallback uses distance transform + watershed-style thresholding to
        # recover those cases.
        def contour_fallback(gray_src):
            # Otsu threshold → distance transform → threshold distance map.
            # This is a simplified watershed that separates touching blobs.
            _, thresh = cv2.threshold(gray_src, 0, 255,
                                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
            cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

            _, sure_fg = cv2.threshold(dist, FALLBACK_DIST_THRESHOLD, 1.0, cv2.THRESH_BINARY)
            sure_fg = np.uint8(sure_fg * 255)

            k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            sure_fg = cv2.dilate(sure_fg, k2, iterations=2)

            contours, _ = cv2.findContours(sure_fg,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            result = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < FALLBACK_MIN_AREA:
                    continue  # too small to be a coin

                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue

                # Circularity filter: 1.0 = perfect circle.  Coins are close
                # to circular; anything below 0.55 is probably background noise.
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                if circularity < FALLBACK_MIN_CIRCULARITY:
                    continue

                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                radius = int(round(radius))
                if not (HOUGH_MIN_RADIUS <= radius <= HOUGH_MAX_RADIUS):
                    continue  # outside our expected coin size range

                result.append(np.array([int(cx), int(cy), radius], dtype=np.uint16))
            return result

        used_fallback = False
        if circles is None or len(circles[0]) < MIN_HOUGH_CIRCLES:
            fallback_circles = contour_fallback(gray_blurred)
            if fallback_circles:
                circles = np.array([fallback_circles])
                used_fallback = True

        # Stage 4: draw all raw candidates before NMS so we can see duplicates.
        detection_vis  = img.copy()
        detection_label = "No circles found"
        raw_count = 0

        if circles is not None:
            raw_circles_vis = np.uint16(np.around(circles))[0, :]
            raw_count = len(raw_circles_vis)
            for rc in raw_circles_vis:
                cv2.circle(detection_vis, (rc[0], rc[1]), rc[2],  (0, 255, 255), 2)
                cv2.circle(detection_vis, (rc[0], rc[1]), 3,      (0, 255, 255), -1)
            method = "fallback (contour)" if used_fallback else "Hough"
            detection_label = f"{raw_count} candidates via {method}"

        self.stage_detection.update_image(detection_vis, caption=detection_label)
        self.root.update_idletasks()

        # ── Step 5: IoU NMS + classification ──────────────────────────────────
        # Sort by radius descending so we always keep the largest circle when
        # two detections overlap — larger = more confident match.
        coin_count  = 0
        total_value = 0.0

        if circles is not None:
            raw_circles        = np.uint16(np.around(circles))[0, :]
            raw_circles_sorted = sorted(raw_circles, key=lambda c: c[2], reverse=True)

            filtered_circles = []
            suppressed       = [False] * len(raw_circles_sorted)

            for i, current_circle in enumerate(raw_circles_sorted):
                if suppressed[i]:
                    continue
                filtered_circles.append(current_circle)
                # Suppress any remaining circle that overlaps this one too much.
                for j in range(i + 1, len(raw_circles_sorted)):
                    if suppressed[j]:
                        continue
                    if self.circle_iou(current_circle, raw_circles_sorted[j]) > IOU_THRESHOLD:
                        suppressed[j] = True

            coin_count = len(filtered_circles)

            for i in filtered_circles:
                x, y, r = i[0], i[1], i[2]
                label, value, color = self.classify_coin(hsv_img, x, y, r)
                total_value += value

                cv2.circle(output_img, (x, y), r, color, 3)       # coloured ring
                cv2.circle(output_img, (x, y), 2, (0, 0, 255), 3) # centre dot
                cv2.putText(output_img, label, (x - 20, y - r - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Stage 5: final annotated image with coin labels.
        self.stage_final.update_image(
            output_img,
            caption=f"{coin_count} coins → {total_value:.2f} EGP"
        )
        self.root.update_idletasks()

        # ── Update stats bar ──────────────────────────────────────────────────
        self.stats_label.config(
            text=f"Total Value: {total_value:.2f} EGP   |   Coins Detected: {coin_count}"
        )

        # ── Render final result in the large panel ────────────────────────────
        output_rgb    = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        final_pil_img = Image.fromarray(output_rgb)
        self.tk_image = ImageTk.PhotoImage(final_pil_img)

        self.canvas.delete("all")
        x_offset = (RESULT_W - new_size[0]) // 2
        y_offset = (RESULT_H - new_size[1]) // 2
        self.canvas.create_image(x_offset, y_offset,
                                 anchor=tk.NW, image=self.tk_image)


if __name__ == "__main__":
    root = tk.Tk()
    app  = CoinCountingApp(root)
    root.mainloop()
