# constants.py

# Threshold EAR (Eye Aspect Ratio) below which the eye is considered closed
EYE_AR_THRESH = 0.21

# Number of consecutive frames the eye must be below threshold to register a blink (dot)
EYE_AR_CONSEC_FRAMES = 2

# Number of consecutive frames the eye must be closed to register a long blink (dash)
EYE_AR_CONSEC_FRAMES_CLOSED = 6

# Number of consecutive open-eye frames to detect the pause between characters
PAUSE_CONSEC_FRAMES = 10

# Number of consecutive open-eye frames to detect the pause between words
WORD_PAUSE_CONSEC_FRAMES = 20

# Number of consecutive closed-eye frames to exit the application
BREAK_LOOP_FRAMES = 40
