import RPi.GPIO as GPIO
import time

# Set GPIO pins for STEP and DIR
STEP_PIN = 17  # GPIO pin connected to STEP input
DIR_PIN = 27   # GPIO pin connected to DIR input
DELAY = 0.001  # Delay between steps (smaller delay = faster speed)

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(STEP_PIN, GPIO.OUT)
GPIO.setup(DIR_PIN, GPIO.OUT)

def stepper_rotate(steps, direction):
    """
    Function to rotate the stepper motor a specified number of steps in a given direction.

    Args:
    - steps (int): Number of steps to move.
    - direction (bool): True for forward, False for backward.
    """
    # Set the direction
    GPIO.output(DIR_PIN, GPIO.HIGH if direction else GPIO.LOW)
    
    # Pulse the STEP pin to move the motor
    for _ in range(steps):
        GPIO.output(STEP_PIN, GPIO.HIGH)
        time.sleep(DELAY)
        GPIO.output(STEP_PIN, GPIO.LOW)
        time.sleep(DELAY)

try:
    # Rotate forward 200 steps (1 full revolution for most 1.8-degree stepper motors)
    print("Rotating forward...")
    stepper_rotate(200, True)
    time.sleep(1)

    # Rotate backward 200 steps
    print("Rotating backward...")
    stepper_rotate(200, False)
    time.sleep(1)

except KeyboardInterrupt:
    print("Program stopped by user")

finally:
    # Cleanup GPIO settings
    GPIO.cleanup()
