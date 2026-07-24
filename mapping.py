# mapping.py
# 28 MediaPipe Face Mesh indices explicitly ordered to match the hysts/anime-face-detector output

MEDIAPIPE_TO_ANIME_INDICES = [
    # JAWLINE (0-4)
    234,  # 0: Left cheek edge
    132,  # 1: Left lower jaw
    152,  # 2: Chin
    361,  # 3: Right lower jaw
    454,  # 4: Right cheek edge

    # RIGHT EYEBROW - User perspective (5-7)
    46,   # 5: Outer edge
    52,   # 6: Middle
    55,   # 7: Inner edge

    # LEFT EYEBROW - User perspective (8-10)
    285,  # 8: Inner edge
    282,  # 9: Middle
    276,  # 10: Outer edge

    # RIGHT EYE - User perspective (11-16)
    33,   # 11: Outer corner
    159,  # 12: Top middle
    133,  # 13: Inner corner
    145,  # 14: Bottom outer
    144,  # 15: Bottom middle
    153,  # 16: Bottom inner

    # LEFT EYE - User perspective (17-22)
    362,  # 17: Inner corner
    386,  # 18: Top middle
    263,  # 19: Outer corner
    380,  # 20: Bottom inner
    374,  # 21: Bottom middle
    373,  # 22: Bottom outer

    # NOSE (23)
    1,    # 23: Nose tip

    # MOUTH (24-27)
    61,   # 24: Left corner
    13,   # 25: Top lip center
    291,  # 26: Right corner
    14    # 27: Bottom lip center
]
