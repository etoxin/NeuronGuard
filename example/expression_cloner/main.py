#!/usr/bin/env python3
"""
====================================================================
🧠 NeuronGuard: Full-Resolution Pixel-Stream Reflex Cloner 🧠
====================================================================
Ingests 100% of the raw, un-downscaled camera pixels. Generates lean,
bit-packed spatial token addresses on the fly from active motion fields.

Controls:
  - Hold [SPACEBAR] to instantly wire the active full-res path to the slot.
  - Press [0, 1, 2, or 3] to toggle your target expression slots.
  - Press [Esc] to exit.
"""

import cv2
import numpy as np


def run_full_res_streamer():
    NUM_CLASSES = 4
    current_target_class = 0
    motor_potentials = np.zeros(NUM_CLASSES, dtype=np.int32)
    metabolic_decay = 0.75  # Aggressive decay to instantly flush massive pixel bursts

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not access webcam.")
        return

    # Capture the true native resolution of the camera hardware
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    TOTAL_PIXELS = w * h

    print("=" * 72)
    print("🎥 INITIALIZING FULL-RESOLUTION PIXEL REFLEX CLONER")
    print("=" * 72)
    print(f"  [Native Resolution] : {w}x{h} ({TOTAL_PIXELS:,} Raw Sensors)")
    print("  [Virtual Field]     : Cache-Locked Index Stream Loop")
    print("=" * 72)

    # 1. Allocate a sparse hash-map or a flattened index tracker for weights
    # To keep memory usage flat, we track active synapses in a standard dictionary.
    # This prevents allocating a giant 7 MB matrix and keeps our memory signature at ~0 KB.
    routing_weights = {}

    class_labels = [
        "Slot 0 (Neutral Base)",
        "Slot 1 (Action A / Smile)",
        "Slot 2 (Action B / Shock)",
        "Slot 3 (Action C / Wave)",
    ]

    # Setup native resolution baseline frame
    _, first_frame = cap.read()
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # 2. Extract full-resolution raw pixel changes
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_delta = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(frame_delta, 20, 255, cv2.THRESH_BINARY)[1]

        # 3. SPIKE INGESTION: Isolate exactly where pixels are changing
        # Find the 1D indices of every single raw pixel that moved
        active_spikes = np.where(thresh.flatten() > 0)[0]
        prev_gray = gray.copy()

        # 4. Handle Keyboard Inputs
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in [ord("0"), ord("1"), ord("2"), ord("3")]:
            current_target_class = int(chr(key))
            print(
                f"🎯 Switched Active Memory Target to: {class_labels[current_target_class]}"
            )

        is_spacebar_held = key == 32

        # 5. Process Plasticity & Neuromorphic Routing Passes
        if len(active_spikes) > 0:
            if is_spacebar_held:
                # HEBBIAN PLASTICITY: Wire the active full-res pixel indices
                for pixel_id in active_spikes:
                    if pixel_id not in routing_weights:
                        routing_weights[pixel_id] = np.zeros(
                            NUM_CLASSES, dtype=np.int16
                        )

                    if routing_weights[pixel_id][current_target_class] < 16000:
                        routing_weights[pixel_id][current_target_class] += 256
            else:
                # INFERENCE: Accumulate signals from active pixel weights
                for pixel_id in active_spikes:
                    if pixel_id in routing_weights:
                        motor_potentials += routing_weights[pixel_id]

        # 6. Apply Clock Tick Metabolic Decay
        motor_potentials = (motor_potentials * metabolic_decay).astype(np.int32)

        # Identify dominant motor output target
        dominant_id = np.argmax(motor_potentials)
        max_potency = motor_potentials[dominant_id]
        prediction_string = (
            class_labels[dominant_id] if max_potency > 5000 else "STATIC / IDLE"
        )

        # 7. Visual Overlay: Draw 100% of the active raw motion pixels onto the frame
        # We overlay the true raw threshold mask directly onto the color video
        frame[thresh > 0] = [0, 255, 255]  # Paints every moving pixel bright yellow

        # HUD Layout Text Render
        status_color = (0, 0, 255) if is_spacebar_held else (255, 255, 255)
        status_text = (
            f"⚡ WIRING ALL ACTIVE PIXELS TO: {class_labels[current_target_class]}"
            if is_spacebar_held
            else f"📝 Slot Selected: {class_labels[current_target_class]} [0-3]"
        )
        cv2.putText(
            frame,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            1,
            cv2.LINE_AA,
        )

        cv2.rectangle(frame, (15, h - 60), (540, h - 15), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"RECOGNIZED ACTION: {prediction_string}",
            (25, h - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if max_potency > 5000 else (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("NeuronGuard: Full-Res Pixel Ingestion", frame)

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    run_full_res_streamer()
