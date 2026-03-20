# """ 
# My CAR CONFIG 

# This file is read by your car application's manage.py script to change the car
# performance

# If desired, all config overrides can be specified here. 
# The update operation will not touch this file.
# """

# import os
# 
# #PATHS
# CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
# DATA_PATH = os.path.join(CAR_PATH, 'data')
# MODELS_PATH = os.path.join(CAR_PATH, 'models')
# 
# #VEHICLE
DRIVE_LOOP_HZ = 60      # the vehicle loop will pause if faster than this speed.
# MAX_LOOPS = None        # the vehicle loop can abort after this many iterations, when given a positive integer.
# 
# #CAMERA
CAMERA_TYPE = "PICAM"   # (PICAM|WEBCAM|CVCAM|CSIC|V4L|D435|MOCK|IMAGE_LIST)
###
# "PICAMLIDAR" for CAM1 + LIDAR IMAGE fusion
# "LIDAR" is only LIDAR ver outputs duplicates lidar/image_array to cam/image_array
# and use CAMERA_NUM_FUSION = -1 

CAMERA_DUAL = True # dual camera for rpi(PICAM) and jetson(CSIC)

# CAMERA_QUAD configuration - Enable quad camera mode with USB cameras
## v4l2-ctl --list-devices　でカメラidを確認する
CAMERA_QUAD = False  # Enable quad camera mode (2 PICAM + 2 USB cameras)
CAMERA_QUAD_USB1_INDEX = 9  # USB camera 1 index (/dev/video*)
CAMERA_QUAD_USB2_INDEX = 11  # USB camera 2 index (/dev/video*)
CAMERA_QUAD_USB_TYPE = "USBCAM"  # Camera type for USB cameras
CAMERA_QUAD_USB_FRAMERATE = 120  # Max framerate for USB cameras (up to 120fps)
CAMERA_QUAD_USB_WIDTH = 224  # USB camera width
CAMERA_QUAD_USB_HEIGHT = 224  # USB camera height
CAMERA_QUAD_USB1_VFLIP = False  # Vertical flip for USB camera 1
CAMERA_QUAD_USB1_HFLIP = False  # Horizontal flip for USB camera 1
CAMERA_QUAD_USB2_VFLIP = False  # Vertical flip for USB camera 2
CAMERA_QUAD_USB2_HFLIP = False  # Horizontal flip for USB camera 2

IMAGE_FIELD_NAME = 'cam/image_array'  # 学習/推論 対象画像フィールド名, デフォルト：'cam/image_array'
IMAGE_FIELD_WEB = 'cam/image_array'     # web表示対象画像フィールド名, デフォルト：'cam/image_array'
IMAGE_FIELD_COLOR_FILTER = 'cam0/image_array'#'cam1/image_array'  # カラーフィルターに使う画像

CAMERA_NUM_FUSION = 1 #0 for original,-1for lidar cam only, 1for cam embed, 2for 2cam horizontal, 3for 2cam&lidar, 4for 4cam grid, 5for main+3embed horizontal
# CAMERA_NUM_FUSION = 5 埋め込み画像の順序設定（左から右への順番）
IMAGE_EMBED_ORDER = ['cam2/image_array', 'cam1/image_array', 'cam3/image_array']  # 左, 中央, 右の順番
#UI上の学習メモ"C:\Users\user\projects\donkeycar_440_old\donkeycar\donkeycar\pipeline\types.py"で指定を変更
IMAGE_MAIN_FIELD_NAME = 'lidar/image_array'  # 画像埋め込みモード時の全体画像フィールド名, cam + cam1 or lidar
IMAGE_EMBED_FIELD_NAME = 'cam0/image_array'  # 画像埋め込みモード時の埋込み画像フィールド名, cam + cam1 or lidar

# クリップ領域の直接指定 (0.0〜1.0の比率で指定)
IMAGE_CLIP_TOP = 0.5 #0.4 sig+ #0.5 sig #0.6 norm     # 上端からの比率 (0.4=上から40%の位置から開始)
IMAGE_CLIP_BOTTOM = 0.75  # 上端からの比率 (0.7=上から70%の位置で終了)
#0.3 IMAGE_CLIP_RATIO_H:0.4 中央中心設定時
#0.7 IMAGE_CLIP_RATIO_H:0.4 中央中心設定時
IMAGE_CLIP_LEFT = 0.0    # 左端からの比率 (0.0=最左部から開始)
IMAGE_CLIP_RIGHT = 1.0   # 左端からの比率 (1.0=最右部で終了)
#LIDAR_CLIP_RATIO=0.45 # lidar画像をクリップして残すサイズ 後方互換性のため
# LIDAR_CLIP_RATIO_W = IMAGE_CLIP_RATIO_W  
# LIDAR_CLIP_RATIO_H = IMAGE_CLIP_RATIO_H
# LIDAR_CLIP_TOP = IMAGE_CLIP_TOP
# LIDAR_CLIP_BOTTOM = IMAGE_CLIP_BOTTOM
# LIDAR_CLIP_LEFT = IMAGE_CLIP_LEFT
# LIDAR_CLIP_RIGHT = IMAGE_CLIP_RIGHT

IMAGE_W = 224 #160
IMAGE_H = 224 #120
IMAGE_DEPTH = 3         # default RGB=3, make 1 for mono
CAMERA_FRAMERATE = 60  # カメラのフレームレート (FPS)
CAMERA_VFLIP = False
CAMERA_HFLIP = False
# カメラ別のフリップ設定 (複数カメラ使用時)
CAMERA_VFLIP_0 = False   # カメラ0の垂直フリップ
CAMERA_HFLIP_0 = False   # カメラ0の水平フリップ
CAMERA_VFLIP_1 = False   # カメラ1の垂直フリップ
CAMERA_HFLIP_1 = False  # カメラ1の水平フリップ
# モーションブラー対策設定
CAMERA_MOTION_BLUR_CONTROL = False  # モーションブラー対策のON/OFF
CAMERA_EXPOSURE_TIME = 5000  # 露出時間(マイクロ秒) - 短いほどブラーが減る (1000-20000)
CAMERA_ANALOGUE_GAIN = 14.0   # アナログゲイン - 暗い場所での感度 (1.0-16.0)

### add for PiCAM 2, but dual camera automatically sets the index
CAMERA_INDEX = 0 #1  # used for 'WEBCAM' and 'CVCAM' when there is more than one camera connected
# # For CSIC camera - If the camera is mounted in a rotated position, changing the below parameter will correct the output frame orientation
# CSIC_CAM_GSTREAMER_FLIP_PARM = 0 # (0 => none , 4 => Flip horizontally, 6 => Flip vertically)
# BGR2RGB = False  # true to convert from BRG format to RGB format; requires opencv
# SHOW_PILOT_IMAGE = False  # show the image used to do the inference when in autopilot mode
# 
# # For IMAGE_LIST camera
# # PATH_MASK = "~/mycar/data/tub_1_20-03-12/*.jpg"
# 
# #9865, over rides only if needed, ie. TX2..
# PCA9685_I2C_ADDR = 0x40     #I2C address, use i2cdetect to validate this number
# PCA9685_I2C_BUSNUM = None   #None will auto detect, which is fine on the pi. But other platforms should specify the bus num.
# 
# #SSD1306_128_32
# USE_SSD1306_128_32 = False    # Enable the SSD_1306 OLED Display
# SSD1306_128_32_I2C_ROTATION = 0 # 0 = text is right-side up, 1 = rotated 90 degrees clockwise, 2 = 180 degrees (flipped), 3 = 270 degrees
# SSD1306_RESOLUTION = 1 # 1 = 128x32; 2 = 128x64
# 
# #
# # DRIVE_TRAIN_TYPE
# # These options specify which chasis and motor setup you are using.
# # See Actuators documentation https://docs.donkeycar.com/parts/actuators/
# # for a detailed explanation of each drive train type and it's configuration.
# # Choose one of the following and then update the related configuration section:
# #
# # "PWM_STEERING_THROTTLE" uses two PWM output pins to control a steering servo and an ESC, as in a standard RC car.
# # "MM1" Robo HAT MM1 board
# # "SERVO_HBRIDGE_2PIN" Servo for steering and HBridge motor driver in 2pin mode for motor
# # "SERVO_HBRIDGE_3PIN" Servo for steering and HBridge motor driver in 3pin mode for motor
# # "DC_STEER_THROTTLE" uses HBridge pwm to control one steering dc motor, and one drive wheel motor
# # "DC_TWO_WHEEL" uses HBridge in 2-pin mode to control two drive motors, one on the left, and one on the right.
# # "DC_TWO_WHEEL_L298N" using HBridge in 3-pin mode to control two drive motors, one of the left and one on the right.
# # "MOCK" no drive train.  This can be used to test other features in a test rig.
# # "VESC" VESC Motor controller to set servo angle and duty cycle
# # (deprecated) "SERVO_HBRIDGE_PWM" use ServoBlaster to output pwm control from the PiZero directly to control steering,
# #                                  and HBridge for a drive motor.
# # (deprecated) "PIGPIO_PWM" uses Raspberrys internal PWM
# # (deprecated) "I2C_SERVO" uses PCA9685 servo controller to control a steering servo and an ESC, as in a standard RC car
# #
DRIVE_TRAIN_TYPE = "PWM_STEERING_THROTTLE"
# 
# #
# # PWM_STEERING_THROTTLE
# #
# # Drive train for RC car with a steering servo and ESC.
# # Uses a PwmPin for steering (servo) and a second PwmPin for throttle (ESC)
# # Base PWM Frequence is presumed to be 60hz; use PWM_xxxx_SCALE to adjust pulse with for non-standard PWM frequencies
# #
PWM_STEERING_THROTTLE = {
     "PWM_STEERING_PIN": "PCA9685.1:40.0",   # PWM output pin for steering servo
     "PWM_STEERING_SCALE": 1.0,              # used to compensate for PWM frequency differents from 60hz; NOT for adjusting steering range
     "PWM_STEERING_INVERTED": False,         # True if hardware requires an inverted PWM pulse
     "PWM_THROTTLE_PIN": "PCA9685.1:40.1",   # PWM output pin for ESC
     "PWM_THROTTLE_SCALE": 1.0,              # used to compensate for PWM frequence differences from 60hz; NOT for increasing/limiting speed
     "PWM_THROTTLE_INVERTED": False,         # True if hardware requires an inverted PWM pulse
     "STEERING_LEFT_PWM": 350, #280,#286,#,               #pwm value for full left steering
     "STEERING_RIGHT_PWM": 520, #470,#468,#,              #pwm value for full right steering
     "THROTTLE_FORWARD_PWM": 250, #270,#           #pwm value for max forward throttle
     "THROTTLE_STOPPED_PWM": 390,            #pwm value for no movement
     "THROTTLE_REVERSE_PWM": 450 #450             #pwm value for max reverse throttle
}

# 
# #
# # I2C_SERVO (deprecated in favor of PWM_STEERING_THROTTLE)
# #
# STEERING_CHANNEL = 1            #(deprecated) channel on the 9685 pwm board 0-15
# STEERING_LEFT_PWM = 460         #pwm value for full left steering
# STEERING_RIGHT_PWM = 290        #pwm value for full right steering
# THROTTLE_CHANNEL = 0            #(deprecated) channel on the 9685 pwm board 0-15
# THROTTLE_FORWARD_PWM = 500      #pwm value for max forward throttle
# THROTTLE_STOPPED_PWM = 370      #pwm value for no movement
# THROTTLE_REVERSE_PWM = 220      #pwm value for max reverse throttle
# 
# #
# # PIGPIO_PWM (deprecated in favor of PWM_STEERING_THROTTLE)
# #
# STEERING_PWM_PIN = 13           #(deprecated) Pin numbering according to Broadcom numbers
# STEERING_PWM_FREQ = 50          #Frequency for PWM
# STEERING_PWM_INVERTED = False   #If PWM needs to be inverted
# THROTTLE_PWM_PIN = 18           #(deprecated) Pin numbering according to Broadcom numbers
# THROTTLE_PWM_FREQ = 50          #Frequency for PWM
# THROTTLE_PWM_INVERTED = False   #If PWM needs to be inverted
# 
# #
# # SERVO_HBRIDGE_2PIN
# # - configures a steering servo and an HBridge in 2pin mode (2 pwm pins)
# # - Servo takes a standard servo PWM pulse between 1 millisecond (fully reverse)
# #   and 2 milliseconds (full forward) with 1.5ms being neutral.
# # - the motor is controlled by two pwm pins, 
# #   one for forward and one for backward (reverse). 
# # - the pwm pin produces a duty cycle from 0 (completely LOW)
# #   to 1 (100% completely high), which is proportional to the
# #   amount of power delivered to the motor.
# # - in forward mode, the reverse pwm is 0 duty_cycle,
# #   in backward mode, the forward pwm is 0 duty cycle.
# # - both pwms are 0 duty cycle (LOW) to 'detach' motor and 
# #   and glide to a stop.
# # - both pwms are full duty cycle (100% HIGH) to brake
# #
# # Pin specifier string format:
# # - use RPI_GPIO for RPi/Nano header pin output
# #   - use BOARD for board pin numbering
# #   - use BCM for Broadcom GPIO numbering
# #   - for example "RPI_GPIO.BOARD.18"
# # - use PIPGIO for RPi header pin output using pigpio server
# #   - must use BCM (broadcom) pin numbering scheme
# #   - for example, "PIGPIO.BCM.13"
# # - use PCA9685 for PCA9685 pin output
# #   - include colon separated I2C channel and address 
# #   - for example "PCA9685.1:40.13"
# # - RPI_GPIO, PIGPIO and PCA9685 can be mixed arbitrarily,
# #   although it is discouraged to mix RPI_GPIO and PIGPIO.
# #
# SERVO_HBRIDGE_2PIN = {
#     "FWD_DUTY_PIN": "RPI_GPIO.BOARD.18",  # provides forward duty cycle to motor
#     "BWD_DUTY_PIN": "RPI_GPIO.BOARD.16",  # provides reverse duty cycle to motor
#     "PWM_STEERING_PIN": "RPI_GPIO.BOARD.33",       # provides servo pulse to steering servo
#     "PWM_STEERING_SCALE": 1.0,        # used to compensate for PWM frequency differents from 60hz; NOT for adjusting steering range
#     "PWM_STEERING_INVERTED": False,   # True if hardware requires an inverted PWM pulse
#     "STEERING_LEFT_PWM": 460,         # pwm value for full left steering (use `donkey calibrate` to measure value for your car)
#     "STEERING_RIGHT_PWM": 290,        # pwm value for full right steering (use `donkey calibrate` to measure value for your car)
# }
# 
# #
# # SERVO_HBRIDGE_3PIN
# # - configures a steering servo and an HBridge in 3pin mode (2 ttl pins, 1 pwm pin)
# # - Servo takes a standard servo PWM pulse between 1 millisecond (fully reverse)
# #   and 2 milliseconds (full forward) with 1.5ms being neutral.
# # - the motor is controlled by three pins, 
# #   one ttl output for forward, one ttl output 
# #   for backward (reverse) enable and one pwm pin
# #   for motor power.
# # - the pwm pin produces a duty cycle from 0 (completely LOW)
# #   to 1 (100% completely high), which is proportional to the
# #   amount of power delivered to the motor.
# # - in forward mode, the forward pin  is HIGH and the
# #   backward pin is LOW,
# # - in backward mode, the forward pin is LOW and the 
# #   backward pin is HIGH.
# # - both forward and backward pins are LOW to 'detach' motor 
# #   and glide to a stop.
# # - both forward and backward pins are HIGH to brake
# #
# # Pin specifier string format:
# # - use RPI_GPIO for RPi/Nano header pin output
# #   - use BOARD for board pin numbering
# #   - use BCM for Broadcom GPIO numbering
# #   - for example "RPI_GPIO.BOARD.18"
# # - use PIPGIO for RPi header pin output using pigpio server
# #   - must use BCM (broadcom) pin numbering scheme
# #   - for example, "PIGPIO.BCM.13"
# # - use PCA9685 for PCA9685 pin output
# #   - include colon separated I2C channel and address 
# #   - for example "PCA9685.1:40.13"
# # - RPI_GPIO, PIGPIO and PCA9685 can be mixed arbitrarily,
# #   although it is discouraged to mix RPI_GPIO and PIGPIO.
# #
# SERVO_HBRIDGE_3PIN = {
#     "FWD_PIN": "RPI_GPIO.BOARD.18",   # ttl pin, high enables motor forward
#     "BWD_PIN": "RPI_GPIO.BOARD.16",   # ttl pin, high enables motor reverse
#     "DUTY_PIN": "RPI_GPIO.BOARD.35",  # provides duty cycle to motor
#     "PWM_STEERING_PIN": "RPI_GPIO.BOARD.33",   # provides servo pulse to steering servo
#     "PWM_STEERING_SCALE": 1.0,        # used to compensate for PWM frequency differents from 60hz; NOT for adjusting steering range
#     "PWM_STEERING_INVERTED": False,   # True if hardware requires an inverted PWM pulse
#     "STEERING_LEFT_PWM": 460,         # pwm value for full left steering (use `donkey calibrate` to measure value for your car)
#     "STEERING_RIGHT_PWM": 290,        # pwm value for full right steering (use `donkey calibrate` to measure value for your car)
# }
# 
# #
# # DRIVETRAIN_TYPE == "SERVO_HBRIDGE_PWM" (deprecated in favor of SERVO_HBRIDGE_2PIN)
# # - configures a steering servo and an HBridge in 2pin mode (2 pwm pins)
# # - Uses ServoBlaster library, which is NOT installed by default, so
# #   you will need to install it to make this work.
# # - Servo takes a standard servo PWM pulse between 1 millisecond (fully reverse)
# #   and 2 milliseconds (full forward) with 1.5ms being neutral.
# # - the motor is controlled by two pwm pins,
# #   one for forward and one for backward (reverse).
# # - the pwm pins produce a duty cycle from 0 (completely LOW)
# #   to 1 (100% completely high), which is proportional to the
# #   amount of power delivered to the motor.
# # - in forward mode, the reverse pwm is 0 duty_cycle,
# #   in backward mode, the forward pwm is 0 duty cycle.
# # - both pwms are 0 duty cycle (LOW) to 'detach' motor and
# #   and glide to a stop.
# # - both pwms are full duty cycle (100% HIGH) to brake
# #
# HBRIDGE_PIN_FWD = 18       # provides forward duty cycle to motor
# HBRIDGE_PIN_BWD = 16       # provides reverse duty cycle to motor
# STEERING_CHANNEL = 0       # PCA 9685 channel for steering control
# STEERING_LEFT_PWM = 460    # pwm value for full left steering (use `donkey calibrate` to measure value for your car)
# STEERING_RIGHT_PWM = 290   # pwm value for full right steering (use `donkey calibrate` to measure value for your car)
# 
# #VESC controller, primarily need to change VESC_SERIAL_PORT  and VESC_MAX_SPEED_PERCENT
# VESC_MAX_SPEED_PERCENT =.2  # Max speed as a percent of the actual speed
# VESC_SERIAL_PORT= "/dev/ttyACM0" # Serial device to use for communication. Can check with ls /dev/tty*
# VESC_HAS_SENSOR= True # Whether or not the bldc motor is using a hall effect sensor
# VESC_START_HEARTBEAT= True # Whether or not to automatically start the heartbeat thread that will keep commands alive.
# VESC_BAUDRATE= 115200 # baudrate for the serial communication. Shouldn't need to change this.
# VESC_TIMEOUT= 0.05 # timeout for the serial communication
# VESC_STEERING_SCALE= 0.5 # VESC accepts steering inputs from 0 to 1. Joystick is usually -1 to 1. This changes it to -0.5 to 0.5
# VESC_STEERING_OFFSET = 0.5 # VESC accepts steering inputs from 0 to 1. Coupled with above change we move Joystick to 0 to 1
# 
# #
# # DC_STEER_THROTTLE with one motor as steering, one as drive
# # - uses L298N type motor controller in two pin wiring
# #   scheme utilizing two pwm pins per motor; one for 
# #   forward(or right) and one for reverse (or left)
# # 
# # GPIO pin configuration for the DRIVE_TRAIN_TYPE=DC_STEER_THROTTLE
# # - use RPI_GPIO for RPi/Nano header pin output
# #   - use BOARD for board pin numbering
# #   - use BCM for Broadcom GPIO numbering
# #   - for example "RPI_GPIO.BOARD.18"
# # - use PIPGIO for RPi header pin output using pigpio server
# #   - must use BCM (broadcom) pin numbering scheme
# #   - for example, "PIGPIO.BCM.13"
# # - use PCA9685 for PCA9685 pin output
# #   - include colon separated I2C channel and address 
# #   - for example "PCA9685.1:40.13"
# # - RPI_GPIO, PIGPIO and PCA9685 can be mixed arbitrarily,
# #   although it is discouraged to mix RPI_GPIO and PIGPIO.
# #
# DC_STEER_THROTTLE = {
#     "LEFT_DUTY_PIN": "RPI_GPIO.BOARD.18",   # pwm pin produces duty cycle for steering left
#     "RIGHT_DUTY_PIN": "RPI_GPIO.BOARD.16",  # pwm pin produces duty cycle for steering right
#     "FWD_DUTY_PIN": "RPI_GPIO.BOARD.15",    # pwm pin produces duty cycle for forward drive
#     "BWD_DUTY_PIN": "RPI_GPIO.BOARD.13",    # pwm pin produces duty cycle for reverse drive
# }
# 
# #
# # DC_TWO_WHEEL pin configuration
# # - configures L298N_HBridge_2pin driver
# # - two wheels as differential drive, left and right.
# # - each wheel is controlled by two pwm pins, 
# #   one for forward and one for backward (reverse). 
# # - each pwm pin produces a duty cycle from 0 (completely LOW)
# #   to 1 (100% completely high), which is proportional to the
# #   amount of power delivered to the motor.
# # - in forward mode, the reverse pwm is 0 duty_cycle,
# #   in backward mode, the forward pwm is 0 duty cycle.
# # - both pwms are 0 duty cycle (LOW) to 'detach' motor and 
# #   and glide to a stop.
# # - both pwms are full duty cycle (100% HIGH) to brake
# #
# # Pin specifier string format:
# # - use RPI_GPIO for RPi/Nano header pin output
# #   - use BOARD for board pin numbering
# #   - use BCM for Broadcom GPIO numbering
# #   - for example "RPI_GPIO.BOARD.18"
# # - use PIPGIO for RPi header pin output using pigpio server
# #   - must use BCM (broadcom) pin numbering scheme
# #   - for example, "PIGPIO.BCM.13"
# # - use PCA9685 for PCA9685 pin output
# #   - include colon separated I2C channel and address 
# #   - for example "PCA9685.1:40.13"
# # - RPI_GPIO, PIGPIO and PCA9685 can be mixed arbitrarily,
# #   although it is discouraged to mix RPI_GPIO and PIGPIO.
# #
# DC_TWO_WHEEL = {
#     "LEFT_FWD_DUTY_PIN": "RPI_GPIO.BOARD.18",  # pwm pin produces duty cycle for left wheel forward
#     "LEFT_BWD_DUTY_PIN": "RPI_GPIO.BOARD.16",  # pwm pin produces duty cycle for left wheel reverse
#     "RIGHT_FWD_DUTY_PIN": "RPI_GPIO.BOARD.15", # pwm pin produces duty cycle for right wheel forward
#     "RIGHT_BWD_DUTY_PIN": "RPI_GPIO.BOARD.13", # pwm pin produces duty cycle for right wheel reverse
# }
# 
# #
# # DC_TWO_WHEEL_L298N pin configuration
# # - configures L298N_HBridge_3pin driver
# # - two wheels as differential drive, left and right.
# # - each wheel is controlled by three pins, 
# #   one ttl output for forward, one ttl output 
# #   for backward (reverse) enable and one pwm pin
# #   for motor power.
# # - the pwm pin produces a duty cycle from 0 (completely LOW)
# #   to 1 (100% completely high), which is proportional to the
# #   amount of power delivered to the motor.
# # - in forward mode, the forward pin  is HIGH and the
# #   backward pin is LOW,
# # - in backward mode, the forward pin is LOW and the 
# #   backward pin is HIGH.
# # - both forward and backward pins are LOW to 'detach' motor 
# #   and glide to a stop.
# # - both forward and backward pins are HIGH to brake
# #
# # GPIO pin configuration for the DRIVE_TRAIN_TYPE=DC_TWO_WHEEL_L298N
# # - use RPI_GPIO for RPi/Nano header pin output
# #   - use BOARD for board pin numbering
# #   - use BCM for Broadcom GPIO numbering
# #   - for example "RPI_GPIO.BOARD.18"
# # - use PIPGIO for RPi header pin output using pigpio server
# #   - must use BCM (broadcom) pin numbering scheme
# #   - for example, "PIGPIO.BCM.13"
# # - use PCA9685 for PCA9685 pin output
# #   - include colon separated I2C channel and address 
# #   - for example "PCA9685.1:40.13"
# # - RPI_GPIO, PIGPIO and PCA9685 can be mixed arbitrarily,
# #   although it is discouraged to mix RPI_GPIO and PIGPIO.
# #
# DC_TWO_WHEEL_L298N = {
#     "LEFT_FWD_PIN": "RPI_GPIO.BOARD.16",        # TTL output pin enables left wheel forward
#     "LEFT_BWD_PIN": "RPI_GPIO.BOARD.18",        # TTL output pin enables left wheel reverse
#     "LEFT_EN_DUTY_PIN": "RPI_GPIO.BOARD.22",    # PWM pin generates duty cycle for left motor speed
# 
#     "RIGHT_FWD_PIN": "RPI_GPIO.BOARD.15",       # TTL output pin enables right wheel forward
#     "RIGHT_BWD_PIN": "RPI_GPIO.BOARD.13",       # TTL output pin enables right wheel reverse
#     "RIGHT_EN_DUTY_PIN": "RPI_GPIO.BOARD.11",   # PWM pin generates duty cycle for right wheel speed
# }
# 
# #ODOMETRY
# HAVE_ODOM = False                   # Do you have an odometer/encoder 
# ENCODER_TYPE = 'GPIO'            # What kind of encoder? GPIO|Arduino|Astar 
# MM_PER_TICK = 12.7625               # How much travel with a single tick, in mm. Roll you car a meter and divide total ticks measured by 1,000
# ODOM_PIN = 13                        # if using GPIO, which GPIO board mode pin to use as input
# ODOM_DEBUG = False                  # Write out values on vel and distance as it runs
# 
# # #LIDAR
# USE_LIDAR = False
# LIDAR_TYPE = 'RP' #(RP|YD)
# LIDAR_LOWER_LIMIT = 90 # angles that will be recorded. Use this to block out obstructed areas on your car, or looking backwards. Note that for the RP A1M8 Lidar, "0" is in the direction of the motor
# LIDAR_UPPER_LIMIT = 270
# 
# # TFMINI
# HAVE_TMINI = False
# TFMINI_SERIAL_PORT = "/dev/serial0" # tfmini serial port, can be wired up or use usb/serial adapter
# 
# #TRAINING
# # The default AI framework to use. Choose from (tensorflow|pytorch)
#DEFAULT_AI_FRAMEWORK = 'tensorflow' 

### ENABLE_MODEL_SWITCHERでモデルを読み込む場合は自動判定
DEFAULT_AI_FRAMEWORK = 'pytorch' 
#DEFAULT_AI_FRAMEWORK = 'onnx' # x3 speed 
#DEFAULT_AI_FRAMEWORK = 'openvino' # x6 speed 
# # The DEFAULT_MODEL_TYPE will choose which model will be created at training
# # time. This chooses between different neural network designs. You can
# # override this setting by passing the command line parameter --type to the
# # python manage.py train and drive commands.
# # tensorflow models: (linear|categorical|tflite_linear|tensorrt_linear)
# # pytorch models: (resnet18)
#DEFAULT_MODEL_TYPE = 'linear'
#DEFAULT_MODEL_TYPE = 'resnet18'
#DEFAULT_MODEL_TYPE = 'edgenext_xx_small'
DEFAULT_MODEL_TYPE = 'donkey' # Use standard linear model type for .h5/.pth files
#DEFAULT_MODEL_TYPE = 'custom_cnn'
#DEFAULT_MODEL_TYPE = '3d' #'resnet18' #'linear'
# BATCH_SIZE = 128                #how many records to use when doing one pass of gradient decent. Use a smaller number if your gpu is running out of memory.
# TRAIN_TEST_SPLIT = 0.8          #what percent of records to use for training. the remaining used for validation.
MAX_EPOCHS = 30                #how many times to visit all records of your data
# SHOW_PLOT = True                #would you like to see a pop up display of final loss?
# VERBOSE_TRAIN = True            #would you like to see a progress bar with text during training?
USE_EARLY_STOP = False           #would you like to stop the training if we see it's not improving fit?
# EARLY_STOP_PATIENCE = 5         #how many epochs to wait before no improvement
# MIN_DELTA = .0005               #early stop will want this much loss change before calling it improved.
# PRINT_MODEL_SUMMARY = True      #print layers and weights to stdout
# OPTIMIZER = None                #adam, sgd, rmsprop, etc.. None accepts default
# LEARNING_RATE = 0.001           #only used when OPTIMIZER specified
# LEARNING_RATE_DECAY = 0.0       #only used when OPTIMIZER specified
# SEND_BEST_MODEL_TO_PI = False   #change to true to automatically send best model during training
# CREATE_TF_LITE = True           # automatically create tflite model in training
# CREATE_TENSOR_RT = False        # automatically create tensorrt model in training
# SAVE_MODEL_AS_H5 = False        # if old keras format should be used instead of savedmodel
# CACHE_IMAGES = True             # if images are cached in training for speed up
# 
# PRUNE_CNN = False               #This will remove weights from your model. The primary goal is to increase performance.
# PRUNE_PERCENT_TARGET = 75       # The desired percentage of pruning.
# PRUNE_PERCENT_PER_ITERATION = 20 # Percenge of pruning that is perform per iteration.
# PRUNE_VAL_LOSS_DEGRADATION_LIMIT = 0.2 # The max amout of validation loss that is permitted during pruning.
# PRUNE_EVAL_PERCENT_OF_DATASET = .05  # percent of dataset used to perform evaluation of model.
# 
# #
# # Augmentations and Transformations
# #
# # - Augmentations are changes to the image that are only applied during
# #   training and are applied randomly to create more variety in the data.
# #   Available augmentations are:
# #   - BRIGHTNESS  - modify the image brightness. See [albumentations](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomBrightnessContrast)
# #   - BLUR        - blur the image. See [albumentations](https://albumentations.ai/docs/api_reference/augmentations/blur/transforms/#albumentations.augmentations.blur.transforms.Blur)
# #
# # - Transformations are changes to the image that apply both in
# #   training and at inference.  They are always applied and in
# #   the configured order.  Available image transformations are:
# #   - Apply a mask to the image:
# #     - 'CROP'      - apply rectangular mask to borders of image
# #     - 'TRAPEZE'   - apply a trapezoidal mask to image
# #   - Apply an enhancement to the image
# #     - 'CANNY'     - apply canny edge detection
# #     - 'BLUR'      - blur the image
# #   - resize the image
# #     - 'RESIZE'    - resize to given pixel width and height
# #     - 'SCALE'     - resize by given scale factor
# #   - change the color space of the image
# #     - 'RGB2BGR'   - change color model from RGB to BGR
# #     - 'BGR2RGB'   - change color model from BGR to RGB
# #     - 'RGB2HSV'   - change color model from RGB to HSV
# #     - 'HSV2RGB'   - change color model from HSV to RGB
# #     - 'BGR2HSV'   - change color model from BGR to HSV
# #     - 'HSV2BGR'   - change color model from HSV to BGR
# #     - 'RGB2GRAY'  - change color model from RGB to greyscale
# #     - 'BGR2GRAY'  - change color model from BGR to greyscale
# #     - 'HSV2GRAY'  - change color model from HSV to greyscale
# #     - 'GRAY2RGB'  - change color model from greyscale to RGB
# #     - 'GRAY2BGR'  - change color model from greyscale to BGR
# #
# # You can create custom tranformations and insert them into the pipeline.
# # - Use a tranformer label that beings with `CUSTOM`, like `CUSTOM_CROP`
# #   and add that to the TRANSFORMATIONS or POST_TRANFORMATIONS list.
# #   So for the custom crop example, that might look like this;
# #   `POST_TRANSFORMATIONS = ['CUSTOM_CROP']`
# # - Set configuration properties for the module and class that
# #   implement your custom transformation.
# #   - The module config will begin with the transformer label
# #     and end with `_MODULE`, like `CUSTOM_CROP_MODULE`.  It's value is
# #     the absolute file path to the python file that has the transformer
# #     class.  For instance, if you called the file
# #     `my_custom_transformer.py` and put in in the root of
# #     your `mycar` folder, next to `myconfig.py`, then you would add 
# #     the following to your myconfig.py file (keeping with the crop example);
# #     `CUSTOM_CROP_MODULE = "/home/pi/mycar/my_custom_transformer.py"`
# #     The actual path will depend on what OS you are using and what
# #     your user name is.
# #   - The class config will begin with the transformer label and end with `_CLASS`,
# #     like `CUSTOM_CROP_CLASS`.  So if your class is called `CustomCropTransformer`
# #     the you would add the following property to your `myconfig.py` file:
# #     `CUSTOM_CROP_CLASS = "CustomCropTransformer"`
# # - Your custom class' constructor will take in the Config object to
# #   it it's constructor.  So you can add whatever configuration properties
# #   you need to your myconfig.py, then read them in the constructor.
# #   You can name the properties anything you want, but it is good practice
# #   to prefix them with the custom tranformer label so they don't conflict
# #   with any other config and so it is way to see what they go with.
# #   For instance, in the custom crop example, we would want the border
# #   values, so that could look like;
# #   ```
# #   CUSTOM_CROP_TOP = 45    # rows to ignore on the top of the image
# #   CUSTOM_CROP_BOTTOM = 5  # rows ignore on the bottom of the image
# #   CUSTOM_CROP_RIGHT = 10  # pixels to ignore on the right of the image
# #   CUSTOM_CROP_LEFT = 10   # pixels to ignore on the left of the image
# #   ```
# # - Your custom class must have a `run` method that takes an image and
# #   returns an image.  It is in this method where you will implement your
# #   transformation logic.
# # - For example, a custom crop that did a blur after the crop might look like;
# #   ```
# #   from donkeycar.parts.cv import ImgCropMask, ImgSimpleBlur
# #
# #   class CustomCropTransformer:
# #       def __init__(self, config) -> None:
# #           self.top = config.CUSTOM_CROP_TOP
# #           self.bottom = config.CUSTOM_CROP_BOTTOM
# #           self.left = config.CUSTOM_CROP_LEFT
# #           self.right = config.CUSTOM_CROP_RIGHT
# #           self.crop = ImgCropMask(self.left, self.top, self.right, self.bottom)
# #           self.blur = ImgSimpleBlur()
# #
# #       def run(self, image):
# #           image = self.crop.run(image)
# #           return self.blur.run(image)
# #   ```
# #
# AUGMENTATIONS = ['MULTIPLY']         # changes to image only applied in training to create
#                            # more variety in the data.
#TRANSFORMATIONS = ['CROP']       # changes applied _before_ training augmentations,
#                            # such that augmentations are applied to the transformed image,
#POST_TRANSFORMATIONS = TRANSFORMATIONS #['CROP']  # transformations applied _after_ training augmentations,
#                            # such that changes are applied to the augmented image
# 
# # Settings for brightness and blur, use 'MULTIPLY' and/or 'BLUR' in
# # AUGMENTATIONS
AUG_BRIGHTNESS_RANGE = 0.2  # this is interpreted as [-0.2, 0.2]
AUG_BLUR_RANGE = (0, 3)
# 
# # "CROP" Transformation
# # Apply mask to borders of the image
# # defined by a rectangle.
# # If these crops values are too large, they will cause the stride values to
# # become negative and the model with not be valid.
# # # # # # # # # # # # # #
# # xxxxxxxxxxxxxxxxxxxxx #
# # xxxxxxxxxxxxxxxxxxxxx #
# # xx                 xx # top
# # xx                 xx #
# # xx                 xx #
# # xxxxxxxxxxxxxxxxxxxxx # bottom
# # xxxxxxxxxxxxxxxxxxxxx #
# # # # # # # # # # # # # #
ROI_CROP_TOP = 0 #45               # the number of rows of pixels to ignore on the top of the image
ROI_CROP_BOTTOM =  int(IMAGE_H/3)           # the number of rows of pixels to ignore on the bottom of the image
ROI_CROP_RIGHT = 0              # the number of rows of pixels to ignore on the right of the image
ROI_CROP_LEFT = 0               # the number of rows of pixels to ignore on the left of the image
# 
# # "TRAPEZE" tranformation
# # Apply mask to borders of image
# # defined by a trapezoid.
# # # # # # # # # # # # # # #
# # xxxxxxxxxxxxxxxxxxxxxxx #
# # xxxx ul     ur xxxxxxxx # min_y
# # xxx             xxxxxxx #
# # xx               xxxxxx #
# # x                 xxxxx #
# # ll                lr xx # max_y
# # # # # # # # # # # # # # #
ROI_TRAPEZE_LL = 0
ROI_TRAPEZE_LR = IMAGE_W
ROI_TRAPEZE_UL = 0 #20
ROI_TRAPEZE_UR = IMAGE_W #- ROI_TRAPEZE_UL
ROI_TRAPEZE_MIN_Y = int(IMAGE_H/6)
ROI_TRAPEZE_MAX_Y = int(IMAGE_H/2)
# 
# # "CANNY" Canny Edge Detection tranformation
# CANNY_LOW_THRESHOLD = 60    # Canny edge detection low threshold value of intensity gradient
# CANNY_HIGH_THRESHOLD = 110  # Canny edge detection high threshold value of intensity gradient
# CANNY_APERTURE = 3          # Canny edge detect aperture in pixels, must be odd; choices=[3, 5, 7]
# 
# # "BLUR" transformation (not this is SEPARATE from the blur augmentation)
# BLUR_KERNEL = 5        # blur kernel horizontal size in pixels
# BLUR_KERNEL_Y = None   # blur kernel vertical size in pixels or None for square kernel
# BLUR_GAUSSIAN = True   # blur is gaussian if True, simple if False
# 
# # "RESIZE" transformation
# RESIZE_WIDTH = 160     # horizontal size in pixels
# RESIZE_HEIGHT = 120    # vertical size in pixels
# 
# # "SCALE" transformation
# SCALE_WIDTH = 1.0      # horizontal scale factor
# SCALE_HEIGHT = None    # vertical scale factor or None to maintain aspect ratio
# 
# #Model transfer options
# #When copying weights during a model transfer operation, should we freeze a certain number of layers
# #to the incoming weights and not allow them to change during training?
# FREEZE_LAYERS = False               #default False will allow all layers to be modified by training
# NUM_LAST_LAYERS_TO_TRAIN = 7        #when freezing layers, how many layers from the last should be allowed to train?
# 
# #WEB CONTROL
# WEB_CONTROL_PORT = int(os.getenv("WEB_CONTROL_PORT", 8887))  # which port to listen on when making a web controller
# WEB_INIT_MODE = "user"              # which control mode to start in. one of user|local_angle|local. Setting local will start in ai mode.
# 
# #JOYSTICK
USE_JOYSTICK_AS_DEFAULT = True      #when starting the manage.py, when True, will not require a --js option to use the joystick
JOYSTICK_MAX_THROTTLE = 1.0         #this scalar is multiplied with the -1 to 1 throttle value to limit the maximum throttle. This can help if you drop the controller or just don't need the full speed available.
JOYSTICK_STEERING_SCALE = 1.0       #some people want a steering that is less sensitve. This scalar is multiplied with the steering -1 to 1. It can be negative to reverse dir.
AUTO_RECORD_ON_THROTTLE = False      #if true, we will record whenever throttle is not zero. if false, you must manually toggle recording with some other trigger. Usually circle button on joystick.
CONTROLLER_TYPE = 'F710'            #(ps3|ps4|xbox|pigpio_rc|nimbus|wiiu|F710|rc3|MM1|custom) custom will run the my_joystick.py controller written by the `donkey createjs` command
# USE_NETWORKED_JS = False            #should we listen for remote joystick control over the network?
# NETWORK_JS_SERVER_IP = None         #when listening for network joystick control, which ip is serving this information
JOYSTICK_DEADZONE = 0.005            # when non zero, this is the smallest throttle before recording triggered.
JOYSTICK_THROTTLE_DIR = -1.0         # use -1.0 to flip forward/backward, use 1.0 to use joystick's natural forward/backward
USE_FPV = False                     # send camera data to FPV webserver
# JOYSTICK_DEVICE_FILE = "/dev/input/js0" # this is the unix file use to access the joystick.
# 
# #For the categorical model, this limits the upper bound of the learned throttle
# #it's very IMPORTANT that this value is matched from the training PC config.py and the robot.py
# #and ideally wouldn't change once set.
# MODEL_CATEGORICAL_MAX_THROTTLE_RANGE = 0.8
# 
###20250522HK
# RC制御設定
#CONTROLLER_TYPE = "xiao_i2c"
XIAO_I2C_ADDRESS = 0x08  # オプション（デフォルト: 0x08）
XIAO_I2C_BUS = 1         # オプション（デフォルト: 1）
RC_STEERING_LEFT = 1332         # Adjust this value if your car cannot run in a straight line
RC_STEERING_MID = 1480         # Adjust this value if your car cannot run in a straight line
RC_STEERING_RIGHT = 1974         # Adjust this value if your car cannot run in a straight line
RC_MAX_FORWARD = 990          # Max throttle to go fowrward. The bigger the faster
RC_STOPPED_PWM = 1480
RC_MAX_REVERSE = 1974          # Max throttle to go reverse. The smaller the faster
RC_SHOW_STEERING_VALUE = True
RC_INVERT = True
RC_JITTER = 0.1   # threshold below which no signal is reported
MODE_LIST = ["user", "local_angle", "local_recovery", "local"]
###
# #RNN or 3D
SEQUENCE_LENGTH = 3 #3             #some models use a number of images over time. This controls how many.
# 
# #IMU
# HAVE_IMU = False                #when true, this add a Mpu6050 part and records the data. Can be used with a
# IMU_SENSOR = 'mpu6050'          # (mpu6050|mpu9250)
# IMU_ADDRESS = 0x68              # if AD0 pin is pulled high them address is 0x69, otherwise it is 0x68
# IMU_DLP_CONFIG = 0              # Digital Lowpass Filter setting (0:250Hz, 1:184Hz, 2:92Hz, 3:41Hz, 4:20Hz, 5:10Hz, 6:5Hz)
# 

# #SOMBRERO
# HAVE_SOMBRERO = False           #set to true when using the sombrero hat from the Donkeycar store. This will enable pwm on the hat.
# 
# #PIGPIO RC control
STEERING_RC_GPIO = 26
THROTTLE_RC_GPIO = 21
DATA_WIPER_RC_GPIO = 19
PIGPIO_STEERING_LEFT = 1332         # Adjust this value if your car cannot run in a straight line
PIGPIO_STEERING_MID = 1480         # Adjust this value if your car cannot run in a straight line
PIGPIO_STEERING_RIGHT = 1974         # Adjust this value if your car cannot run in a straight line
PIGPIO_MAX_FORWARD = 990          # Max throttle to go fowrward. The bigger the faster
PIGPIO_STOPPED_PWM = 1480
PIGPIO_MAX_REVERSE = 1974          # Max throttle to go reverse. The smaller the faster
PIGPIO_SHOW_STEERING_VALUE = True
PIGPIO_INVERT = False
PIGPIO_JITTER = 0.1   # threshold below which no signal is reported
# 
# 
# 
# #ROBOHAT MM1
# MM1_STEERING_MID = 1500         # Adjust this value if your car cannot run in a straight line
# MM1_MAX_FORWARD = 2000          # Max throttle to go fowrward. The bigger the faster
# MM1_STOPPED_PWM = 1500
# MM1_MAX_REVERSE = 1000          # Max throttle to go reverse. The smaller the faster
# MM1_SHOW_STEERING_VALUE = False
# # Serial port 
# # -- Default Pi: '/dev/ttyS0'
# # -- Jetson Nano: '/dev/ttyTHS1'
# # -- Google coral: '/dev/ttymxc0'
# # -- Windows: 'COM3', Arduino: '/dev/ttyACM0'
# # -- MacOS/Linux:please use 'ls /dev/tty.*' to find the correct serial port for mm1 
# #  eg.'/dev/tty.usbmodemXXXXXX' and replace the port accordingly
# MM1_SERIAL_PORT = '/dev/ttyS0'  # Serial Port for reading and sending MM1 data.
# 
# #LOGGING
# HAVE_CONSOLE_LOGGING = True
# LOGGING_LEVEL = 'INFO'          # (Python logging level) 'NOTSET' / 'DEBUG' / 'INFO' / 'WARNING' / 'ERROR' / 'FATAL' / 'CRITICAL'
# LOGGING_FORMAT = '%(message)s'  # (Python logging format - https://docs.python.org/3/library/logging.html#formatter-objects
# 
# #TELEMETRY
# HAVE_MQTT_TELEMETRY = False
# TELEMETRY_DONKEY_NAME = 'my_robot1234'
# TELEMETRY_MQTT_TOPIC_TEMPLATE = 'donkey/%s/telemetry'
# TELEMETRY_MQTT_JSON_ENABLE = False
# TELEMETRY_MQTT_BROKER_HOST = 'broker.hivemq.com'
# TELEMETRY_MQTT_BROKER_PORT = 1883
# TELEMETRY_PUBLISH_PERIOD = 1
# TELEMETRY_LOGGING_ENABLE = True
# TELEMETRY_LOGGING_LEVEL = 'INFO' # (Python logging level) 'NOTSET' / 'DEBUG' / 'INFO' / 'WARNING' / 'ERROR' / 'FATAL' / 'CRITICAL'
# TELEMETRY_LOGGING_FORMAT = '%(message)s'  # (Python logging format - https://docs.python.org/3/library/logging.html#formatter-objects
# TELEMETRY_DEFAULT_INPUTS = 'pilot/angle,pilot/throttle,recording'
# TELEMETRY_DEFAULT_TYPES = 'float,float'
# 
# # PERF MONITOR
# HAVE_PERFMON = False
# 
# #RECORD OPTIONS
RECORD_DURING_AI = True        #normally we do not record during ai mode. Set this to true to get image and steering records for your Ai. Be careful not to use them to train.
# AUTO_CREATE_NEW_TUB = False     #create a new tub (tub_YY_MM_DD) directory when recording or append records to data directory directly
# 
# #LED
# HAVE_RGB_LED = False            #do you have an RGB LED like https://www.amazon.com/dp/B07BNRZWNF
# LED_INVERT = False              #COMMON ANODE? Some RGB LED use common anode. like https://www.amazon.com/Xia-Fly-Tri-Color-Emitting-Diffused/dp/B07MYJQP8B
# 
# #LED board pin number for pwm outputs
# #These are physical pinouts. See: https://www.raspberrypi-spy.co.uk/2012/06/simple-guide-to-the-rpi-gpio-header-and-pins/
# LED_PIN_R = 12
# LED_PIN_G = 10
# LED_PIN_B = 16
# 
# #LED status color, 0-100
# LED_R = 0
# LED_G = 0
# LED_B = 1
# 
# #LED Color for record count indicator
# REC_COUNT_ALERT = 1000          #how many records before blinking alert
# REC_COUNT_ALERT_CYC = 15        #how many cycles of 1/20 of a second to blink per REC_COUNT_ALERT records
# REC_COUNT_ALERT_BLINK_RATE = 0.4 #how fast to blink the led in seconds on/off
# 
# #first number is record count, second tuple is color ( r, g, b) (0-100)
# #when record count exceeds that number, the color will be used
# RECORD_ALERT_COLOR_ARR = [ (0, (1, 1, 1)),
#             (3000, (5, 5, 5)),
#             (5000, (5, 2, 0)),
#             (10000, (0, 5, 0)),
#             (15000, (0, 5, 5)),
#             (20000, (0, 0, 5)), ]
# 
# 
# #LED status color, 0-100, for model reloaded alert
# MODEL_RELOADED_LED_R = 100
# MODEL_RELOADED_LED_G = 0
# MODEL_RELOADED_LED_B = 0
# 
# 
# #BEHAVIORS
# #When training the Behavioral Neural Network model, make a list of the behaviors,
# #Set the TRAIN_BEHAVIORS = True, and use the BEHAVIOR_LED_COLORS to give each behavior a color
#TRAIN_BEHAVIORS = True
#BEHAVIOR_LIST = ['Left_Lane', "Right_Lane"]
#BEHAVIOR_LED_COLORS = [(0, 10, 0), (10, 0, 0)]  #RGB tuples 0-100 per chanel
# 
# #Localizer
# #The localizer is a neural network that can learn to predict its location on the track.
# #This is an experimental feature that needs more developement. But it can currently be used
# #to predict the segement of the course, where the course is divided into NUM_LOCATIONS segments.
#TRAIN_LOCALIZER = True
#NUM_LOCATIONS = 2
#BUTTON_PRESS_NEW_TUB = True #when enabled, makes it easier to divide our data into one tub per track length if we make a new tub on each X button press.
# 
# #DonkeyGym
# #Only on Ubuntu linux, you can use the simulator as a virtual donkey and
# #issue the same python manage.py drive command as usual, but have them control a virtual car.
# #This enables that, and sets the path to the simualator and the environment.
# #You will want to download the simulator binary from: https://github.com/tawnkramer/donkey_gym/releases/download/v18.9/DonkeySimLinux.zip
# #then extract that and modify DONKEY_SIM_PATH.
# DONKEY_GYM = False
# DONKEY_SIM_PATH = "path to sim" #"/home/tkramer/projects/sdsandbox/sdsim/build/DonkeySimLinux/donkey_sim.x86_64" when racing on virtual-race-league use "remote", or user "remote" when you want to start the sim manually first.
# DONKEY_GYM_ENV_NAME = "donkey-generated-track-v0" # ("donkey-generated-track-v0"|"donkey-generated-roads-v0"|"donkey-warehouse-v0"|"donkey-avc-sparkfun-v0")
# GYM_CONF = { "body_style" : "donkey", "body_rgb" : (128, 128, 128), "car_name" : "car", "font_size" : 100} # body style(donkey|bare|car01) body rgb 0-255
# GYM_CONF["racer_name"] = "Your Name"
# GYM_CONF["country"] = "Place"
# GYM_CONF["bio"] = "I race robots."
# 
# SIM_HOST = "127.0.0.1"              # when racing on virtual-race-league use host "trainmydonkey.com"
# SIM_ARTIFICIAL_LATENCY = 0          # this is the millisecond latency in controls. Can use useful in emulating the delay when useing a remote server. values of 100 to 400 probably reasonable.
# 
# # Save info from Simulator (pln)
# SIM_RECORD_LOCATION = False
# SIM_RECORD_GYROACCEL= False
# SIM_RECORD_VELOCITY = False
# SIM_RECORD_LIDAR = False
# 
# #publish camera over network
# #This is used to create a tcp service to publish the camera feed
# PUB_CAMERA_IMAGES = False
# 
# #When racing, to give the ai a boost, configure these values.
# AI_LAUNCH_DURATION = 0.0            # the ai will output throttle for this many seconds
# AI_LAUNCH_THROTTLE = 0.0            # the ai will output this throttle value
# AI_LAUNCH_ENABLE_BUTTON = 'R2'      # this keypress will enable this boost. It must be enabled before each use to prevent accidental trigger.
# AI_LAUNCH_KEEP_ENABLED = False      # when False ( default) you will need to hit the AI_LAUNCH_ENABLE_BUTTON for each use. This is safest. When this True, is active on each trip into "local" ai mode.
# 
# #Scale the output of the throttle of the ai pilot for all model types.
AI_THROTTLE_MULT = 1 #1.05 #0.85#donkeycar_20251111_122036_best_openvino.xml    # this multiplier will scale every throttle value for all output from NN models
# 
# #Path following
# PATH_FILENAME = "donkey_path.pkl"   # the path will be saved to this filename
# PATH_SCALE = 5.0                    # the path display will be scaled by this factor in the web page
# PATH_OFFSET = (0, 0)                # 255, 255 is the center of the map. This offset controls where the origin is displayed.
# PATH_MIN_DIST = 0.3                 # after travelling this distance (m), save a path point
# PID_P = -10.0                       # proportional mult for PID path follower
# PID_I = 0.000                       # integral mult for PID path follower
# PID_D = -0.2                        # differential mult for PID path follower
# PID_THROTTLE = 0.2                  # constant throttle value during path following
# USE_CONSTANT_THROTTLE = False       # whether or not to use the constant throttle or variable throttle captured during path recording
# SAVE_PATH_BTN = "cross"             # joystick button to save path
# RESET_ORIGIN_BTN = "triangle"       # joystick button to press to move car back to origin
# 
# # Intel Realsense D435 and D435i depth sensing camera
# REALSENSE_D435_RGB = True       # True to capture RGB image
# REALSENSE_D435_DEPTH = True     # True to capture depth as image array
# REALSENSE_D435_IMU = False      # True to capture IMU data (D435i only)
# REALSENSE_D435_ID = None        # serial number of camera or None if you only have one camera (it will autodetect)
# 
# # Stop Sign Detector
# STOP_SIGN_DETECTOR = False
# STOP_SIGN_MIN_SCORE = 0.2
# STOP_SIGN_SHOW_BOUNDING_BOX = True
# STOP_SIGN_MAX_REVERSE_COUNT = 10    # How many times should the car reverse when detected a stop sign, set to 0 to disable reversing
# STOP_SIGN_REVERSE_THROTTLE = -0.5     # Throttle during reversing when detected a stop sign
# 
# # FPS counter
SHOW_FPS = True
FPS_DEBUG_INTERVAL = 10    # the interval in seconds for printing the frequency info into the shell
# 
# # PI connection
# PI_USERNAME = "pi"
# PI_HOSTNAME = "donkeypi.local"


# ============================================================================
# LiDAR Type Selection (以下のうち1つだけをTrueにする)
# ============================================================================
HAVE_LIDAR = True
LIDAR_TYPE = "TMINI" #"TMINI", "UST20"

#USE_LIDAR = True # モデルにlidar点群（nparrayを含める）
USE_LIDAR_SLAM = False # UST-20ベースのSLAM実行

# ============================================================================
# SLAM モード設定
# ============================================================================
# SLAM動作モード: 'mapping' (地図生成), 'localization' (自己位置推定)
# localizationモードでは、既存のPNG画像を読み込んで格子点地図として使用します
SLAM_MODE = 'mapping'

# 既存地図ファイル (localizationモード時に使用)
# BreezySLAMで保存した生の地図データ画像（_raw.png）のパスを指定
# 注意: グラフ装飾のない_raw.pngファイルを使用してください
SLAM_MAP_FILE = 'ust20_slam_final_20250831_210928_raw.png'  # 例: 'ust20_slam_map_20250831_120000_raw.png'

# SLAM設定
SLAM_MAP_SIZE_PIXELS = 800      # マップサイズ（ピクセル単位）
SLAM_MAP_SIZE_METERS = 10       # マップサイズ（メートル単位）
SLAM_MAP_QUALITY = 3            # マップの品質（1-255、大きいほど高品質）

# ============================================================================
# 共通設定
# ============================================================================
SAVE_LIDAR_IMAGES = True    # LiDAR画像保存機能
SAVE_LIDAR_DATA = False      # LiDARの点群データをnumpy binary形式で保存
LIDAR_BINARY_IMAGE = False # LiDAR画像を白黒2値で表示するか（Trueの場合、点群は全て白、背景は黒）
WEB_SERVER_PORT = 8080      # Lidar単体確認用ウェブサーバーポート

# LiDAR画像縮尺設定
LIDAR_IMAGE_SCALE_FACTOR = 0.8  # 画像サイズに対するスケール係数 (0.0-1.0)
LIDAR_IMAGE_METERS_PER_PIXEL = 0.018  # 1ピクセルあたりの実際の距離（メートル）
# 注：0.045 = 約10m四方を224ピクセルで表示（10m/224px = 0.0446m/px）
#     0.018 = 約4m四方を224ピクセルで表示
#     0.089 = 約20m四方を224ピクセルで表示

# 車両サイズ設定（mm単位）
VEHICLE_WIDTH = 200   # 車両の幅（mm）
VEHICLE_LENGTH = 450  # 車両の長さ（mm）
VEHICLE_DISPLAY_COLOR = (255, 255, 255)  # 車両表示色（RGB)
VEHICLE_DISPLAY_THICKNESS = 2  # 車両枠線の太さ（ピクセル）

# LiDAR搭載位置オフセット（mm単位）
# 車両中心を原点として、前方が正のY、右が正のX
LIDAR_OFFSET_X = 0    # 左右方向のオフセット（右が正）
LIDAR_OFFSET_Y = 330-450/2    # 前後方向のオフセット（前が正）
# 例：LIDAR_OFFSET_Y = 100 とすると、LiDARは車両中心より100mm前方に搭載

# 共通ゾーン名の定義（4ゾーン）
# RrLH: Rear Left Half (左後方)
# FrLH: Front Left Half (左前方)
# FrFR: Front  (前方)
# FrRH: Front Right Half (右前方)
# RrRH: Rear Right Half (右後方)
#ZONE_NAMES = ["RrLH", "FrLH", "FrRH", "RrRH"]
ZONE_NAMES = ["RrLH", "FrLH", "FrFR", "FrRH", "RrRH"]
RECOVERY_DETECTION_DIV = len(ZONE_NAMES)     # 検出ゾーン分割数

# 検出設定
LIDAR_DETECT_POINTS_THRESHOLD = 20    # 検出点数閾値（lidar毎にデータ点数異なるため注意）
# ゾーン別検出閾値（5つのゾーンそれぞれに設定可能）
LIDAR_DETECT_POINTS_THRESHOLD_ZONE = [
    LIDAR_DETECT_POINTS_THRESHOLD*2,  # Zone 0: 左後方 (RrLH)
    LIDAR_DETECT_POINTS_THRESHOLD,  # Zone 1: 左前方 (FrLH)
    LIDAR_DETECT_POINTS_THRESHOLD,  # Zone 2: 前方 (FrFR)
    LIDAR_DETECT_POINTS_THRESHOLD,  # Zone 3: 右前方 (FrRH)
    LIDAR_DETECT_POINTS_THRESHOLD*2   # Zone 4: 右後方 (RrRH)
]

LIDAR_DETECT_DISTANCE_THRESHOLD = 300    # 検出距離閾値　unit:mm
# ゾーン別検出閾値（5つのゾーンそれぞれに設定可能）
LIDAR_DETECT_DISTANCE_THRESHOLD_ZONE = [
    LIDAR_DETECT_DISTANCE_THRESHOLD +0,  # Zone 0: 左後方 (RrLH)
    LIDAR_DETECT_DISTANCE_THRESHOLD,  # Zone 1: 左前方 (FrLH)
    LIDAR_DETECT_DISTANCE_THRESHOLD + 100, #togikai -100,  # Zone 2: 前方 (FrFR)
    LIDAR_DETECT_DISTANCE_THRESHOLD,  # Zone 3: 右前方 (FrRH)
    LIDAR_DETECT_DISTANCE_THRESHOLD +0   # Zone 4: 右後方 (RrRH)
]
LIDAR_WALL_DISTANCE = LIDAR_DETECT_DISTANCE_THRESHOLD # 壁検出（描画）ロジック用

# ==============================================================================
# 壁検出設定
# ==============================================================================
# 壁検出を有効にするか(推論実行と同時処理で負荷が高くなり処理が重くなる)
LIDAR_DETECT_WALLS =False

# 使用する検出手法（速度順）
# 選択肢: 'disTance_based', 'split_merge', 'sliding_window', 'ransac', 'hybrid'
LIDAR_DETECTION_METHOD = 'split_merge'


# 壁として認識する点間の最大距離 (mm)
LIDAR_WALL_MAX_GAP = 300

# 壁セグメントとして必要な最小点数
LIDAR_WALL_MIN_POINTS = 25

# 最大許容直線偏差（低いほど厳密な直線を要求）
LIDAR_WALL_MAX_LINEARITY = 0.08

# ------------------------------------------------------------------------------
# Split-and-Merge法用パラメータ
# ------------------------------------------------------------------------------

# 分割閾値 (mm) - 点から直線への最大許容距離
LIDAR_SPLIT_EPSILON = 90
# 最小セグメント長 (mm) - この長さ以下のセグメントは除外
LIDAR_MIN_SEGMENT_LENGTH = 900
# 適応的閾値を使用するか
LIDAR_USE_ADAPTIVE = True
# 2D最適化を使用するか
LIDAR_USE_2D_OPTIMIZATION = True

# ------------------------------------------------------------------------------
# RANSAC法用パラメータ
# ------------------------------------------------------------------------------
# RANSAC残差閾値 (mm) - インライアと判定する最大距離
LIDAR_RANSAC_THRESHOLD = 60
# 最小インライア率 - 直線として採用する最小の点の割合
LIDAR_MIN_INLIER_RATIO = 0.6
# RANSAC最大試行回数
LIDAR_RANSAC_MAX_TRIALS = 150
# RANSAC早期終了閾値
LIDAR_EARLY_STOP_RATIO = 0.9
# ------------------------------------------------------------------------------
# スライディングウィンドウ法用パラメータ
# ------------------------------------------------------------------------------
# ウィンドウサイズ（点数）
LIDAR_WINDOW_SIZE = 20
# ウィンドウの移動幅（点数）
LIDAR_WINDOW_STRIDE = 5
# 重複閾値 (mm) - セグメント重複判定用
LIDAR_OVERLAP_THRESHOLD = 700

# ------------------------------------------------------------------------------
# ハイブリッド検出器用パラメータ
# ------------------------------------------------------------------------------
# RANSAC検証の信頼度閾値
LIDAR_CONFIDENCE_THRESHOLD = 0.8

# ------------------------------------------------------------------------------
# セグメント統合用パラメータ
# ------------------------------------------------------------------------------
# 統合時の角度閾値 (度) - この角度差以内のセグメントを統合
LIDAR_MERGE_ANGLE_THRESHOLD = 10
# 統合時の距離閾値 (mm) - この距離以内のセグメントを統合
LIDAR_MERGE_DISTANCE_THRESHOLD = 100


# ============================================================================
# LiDAR種類別設定
# ============================================================================

if LIDAR_TYPE == "TMINI":
    # YDLIDAR TMINI 設定
    LIDAR_DATA_POINTS = 400
    LIDAR_ANGLE_RANGE = 360     # 度
    LIDAR_ANGLE_START = 0    # 度
    LIDAR_ANGLE_END = 360       # 度
    LIDAR_ANGLE_OFFSET = 90      # 度 - LiDARの向きを調整するオフセット値（正の値で右回転）
    LIDAR_CLOCKWISE = True      # スキャン方向（True:時計回り、False:反時計回り）
    
    # 通信設定
    LIDAR_COMM_TYPE = "serial"
    LIDAR_SERIAL_PORT = "/dev/ttyAMA0"
    LIDAR_SERIAL_BAUDRATE = 230400
    
    # 単位系設定
    LIDAR_UNIT_TYPE = "m"       # TMINIのネイティブ単位系（"m" or "mm"）
    LIDAR_TARGET_UNIT = "mm"    # システム内部で使用する単位系（"m" or "mm"）
    
    # 測定範囲
    LIDAR_MIN_DISTANCE = 20        # unit:mm
    LIDAR_MAX_DISTANCE = 6000      # unit:mm (TMINIの実用範囲)
            
    # 5つのゾーンに分割したLiDAR検出範囲を定義（UST20と同様の構成）
    # TMINIの400点を5つのゾーンに分割
    # 360度を400点で分割: 1点あたり0.9度
    # 角度オフセット90度により、インデックス0は車両の右方向（90度）を指す
    # RrLH, FrLH, FrFR, FrRH, RrRH の順（UST20と同じ順序）
    ZONE_INDEX = [
        [x for x in range(317, 350)],   # RrLH: 左後方 (-75°~-45° = 285°~315° = インデックス317~350)
        [x for x in range(350, 383)],   # FrLH: 左前方 (-45°~-15° = 315°~345° = インデックス350~383)
        [x for x in range(383, 400)]+[x for x in range(0, 17)],   # FrFR: 前方 (-15°~15° = 345°~15° = インデックス383~400,0~17)
        [x for x in range(17, 50)],     # FrRH: 右前方 (15°~45° = インデックス17~50)
        [x for x in range(50, 83)]      # RrRH: 右後方 (45°~75° = インデックス50~83)
    ]

# 北陽 UST-20 設定
elif LIDAR_TYPE == "UST20":
    LIDAR_DATA_POINTS = 1081
    LIDAR_CLOCKWISE = False     # スキャン方向（True:時計回り、False:反時計回り）- UST-20は反時計回り
    LIDAR_ANGLE_RANGE = 270     # 度
    LIDAR_ANGLE_START = -135   # 度（インデックス0の角度、LidarImageConverterの表示調整）
    LIDAR_ANGLE_END = 135       # 度（最後のインデックスの角度）
    LIDAR_ANGLE_STEP = 4        # 
    LIDAR_ANGLE_OFFSET =90   # 度（車両の向きとLidarImageConverterの表示調整、0が右向き）

    # 通信設定
    LIDAR_COMM_TYPE = "ethernet"
    LIDAR_IP_ADDRESS = "192.168.0.10"
    LIDAR_PORT = 10940
    
    # 単位系設定
    LIDAR_UNIT_TYPE = "mm"      # UST-20のネイティブ単位系（"m" or "mm"）
    LIDAR_TARGET_UNIT = "mm"    # システム内部で使用する単位系（"m" or "mm"）
    
    # 測定範囲
    LIDAR_MIN_DISTANCE = 100       # unit:mm (UST-20:20mm~)
    LIDAR_MAX_DISTANCE = 20000     # unit:mm (の最大範囲)
    
    # ゾーンインデックスの定義（ZONE_NAMESに対応）
    # RrLH, FrLH, FrFR, FrRH, RrRH の順
    ZONE_INDEX = [
        [x for x in range(180 *LIDAR_ANGLE_STEP,     240 *LIDAR_ANGLE_STEP)],  # RrLH: 左後方
        [x for x in range(150 *LIDAR_ANGLE_STEP,     180 *LIDAR_ANGLE_STEP)],     # FrLH: 左前方
        [x for x in range(120 *LIDAR_ANGLE_STEP,     150 *LIDAR_ANGLE_STEP)],     # FrFR: 前方
        [x for x in range(90 *LIDAR_ANGLE_STEP,      120 *LIDAR_ANGLE_STEP)],      # FrRH: 右前方
        [x for x in range(30 *LIDAR_ANGLE_STEP,      90 *LIDAR_ANGLE_STEP)]     # RrRH: 右後方
    ]

    # 30°づつ設定
    # ZONE_INDEX = [
    #     [x for x in range(180*LIDAR_ANGLE_STEP, (-15+225)*LIDAR_ANGLE_STEP)],  # RrLH: 左後方
    #     [x for x in range(150*LIDAR_ANGLE_STEP, 180*LIDAR_ANGLE_STEP)],     # FrLH: 左前方
    #     [x for x in range(120*LIDAR_ANGLE_STEP, 150*LIDAR_ANGLE_STEP)],     # FrFR: 前方
    #     [x for x in range(90*LIDAR_ANGLE_STEP, 120*LIDAR_ANGLE_STEP)],      # FrRH: 右前方
    #     [x for x in range((15+45)*LIDAR_ANGLE_STEP, 90*LIDAR_ANGLE_STEP)]     # RrRH: 右後方
    # ]


else:
    # LiDARが設定されていない場合のデフォルト値
    LIDAR_TYPE = "NONE"
    LIDAR_DATA_POINTS = 0
    LIDAR_ANGLE_RANGE = 0
    LIDAR_ANGLE_START = 0
    LIDAR_ANGLE_END = 0
    LIDAR_ANGLE_OFFSET = 0
    LIDAR_CLOCKWISE = True
    LIDAR_MIN_DISTANCE = 20        # unit:mm
    LIDAR_MAX_DISTANCE = 4000      # unit:mm

#  # #RECOVERY SETTING
RECOVERY_DISTANCE = 100 # 110 #togi #150 #nagoya ### mm
RECOVERY_DETECTION_TIMES = 2 #3 #N of detection times before recovering
RECOVERY_DURATION =  0.08 #togi #0.15 #nagoya  ### seconds
RECOVERY_DURATION_BACK = 0.3 #0.4 #togi 0.3 #0.6 nagoya # ### seconds
RECOVERY_THROTTLE = -0.7 ###-0.4 #-0.7 #togi #-0.4 nagoya 
RECOVERY_ANGLE = 0.6 #togi 0.8 #nagoya  

# LiDARベースのスロットル制御設定
# ============================================================================
# 特定のゾーンに障害物がない場合、スロットルを自動的に設定値に変更する機能
LIDAR_THROTTLE_ENABLED =  False # True: 有効, False: 無効

# 監視するゾーンのインデックス (0-based)
# Zone 0: 左後方 (RrLH)
# Zone 1: 左前方 (FrLH)  
# Zone 2: 前方 (FrFR)
# Zone 3: 右前方 (FrRH)
# Zone 4: 右後方 (RrRH)
LIDAR_THROTTLE_ZONE = 2  # 前方ゾーンを監視

# 障害物検出の距離閾値 (mm単位)
# この距離より近くに障害物がない場合、スロットルを設定値に変更
LIDAR_THROTTLE_DISTANCE = 4000  

# 障害物がない時のスロットル値
LIDAR_THROTTLE_VALUE = 1  # 最大スロットル
# ============================================================================

###

### add color filter logic
# logic id, roi, target
## 0:none, yellow straight signal
## 1:up, 'right': Magenta
## 2:up, 'left': Cyan  
## 3:lo,  'pylon':Yellow 
HAVE_COLOR_FILTER = False ###
HAVE_COLOR_FILTER_CONTOROLLER = False
NUM_FILTER = 3
NUM_LOGIC_PIXEL = IMAGE_W *0.8
COLOR_FILTER_TYPE = 'HSV' #HSV or RGB(not dep)
FILTER_NAMES = ['Magenta', 'Cyan', 'Pink', 'White']  # フィルター名（動的に変更可能）
#FILTER_NAMES = ['Magenta', 'Cyan', 'Yellow', 'Pink']  # フィルター名（動的に変更可能）

# カラーフィルターコントローラー設定
COLOR_FILTER_DETECTION_THRESHOLD = 3  # 連続検出回数の閾値（この回数検出されるとアクション開始）
COLOR_FILTER_BRAKE_DURATION = 0.5     # ブレーキ動作継続時間（秒）
COLOR_FILTER_BRAKE_COOLDOWN = 5.0    # ブレーキ後のクールダウン（無反応）時間（秒）

# 各フィルターのアクション設定（オプション：カスタマイズ時に使用）
# カラーフィルターアクション設定
# 'type': アクションタイプ ('turn', 'brake', 'speed', 'none')
# 'none'を指定すると該当フィルターは無効化される

# フィルター0（Magenta）: 左旋回
COLOR_FILTER_ACTION_0 = {
    'type': 'none',
    'angle_modifier': -0.5,
    'throttle_modifier': 0.7,
    'duration': 0.5,
    'cooldown': 2.0
}

# フィルター1（Cyan）: 右旋回
COLOR_FILTER_ACTION_1 = {
    'type': 'none',
    'angle_modifier': 0.5,
    'throttle_modifier': 0.7,
    'duration': 0.5,
    'cooldown': 2.0
}

# フィルター2（Pink）: ブレーキ
COLOR_FILTER_ACTION_2 = {
    'type': 'brake',
    'angle': 0.0,
    'throttle': -1.0,
    'duration': COLOR_FILTER_BRAKE_DURATION,
    'cooldown': COLOR_FILTER_BRAKE_COOLDOWN
}

# フィルター3（White）: 無効化（例）
COLOR_FILTER_ACTION_3 = {
    'type': 'none',  # 'none'を指定すると無効化
    'duration': 0.0,
    'cooldown': 0.0
}

# Hはopencv仕様で0~179
LOWER_FILTER = [
    [130, 40, 200],  # 0: Magenta (マゼンタ, 右) - H:160-179, S:100-255, V:100-255  
    [85, 90, 200],   # 1: Cyan (シアン, 左) - H:85-105, S:100-255, V:100-255
    [165, 0, 150],   # 2: Pink - H:165-179, S:0-100, V:150-250
    [80, 0, 210]      # 3: White (ホワイト) - H:0-179, S:0-30, V:200-255
]
UPPER_FILTER = [
    [160, 255, 255],  # 0: Magenta (マゼンタ) 
    [110, 255, 255],  # 1: Cyan (シアン)
    [179, 100, 250],  # 2: Pink
    [100, 30, 240]    # 3: White (ホワイト)
]
COLOR_FILTER_ROI_CROP_TOP = [0,0,112,112] #45               # the number of rows of pixels to ignore on the top of the image
COLOR_FILTER_ROI_CROP_BOTTOM = [112,112,224,168] #int(IMAGE_H/2)           # the number of rows of pixels to ignore on the bottom of the image
COLOR_FILTER_ROI_CROP_LEFT = [0, 0, 0, 56] #45               # the number of rows of pixels to ignore on the top of the image
COLOR_FILTER_ROI_CROP_RIGHT = [224, 224, 224, 224-56] #int(IMAGE_H/2)           # the number of rows of pixels to ignore on the bottom of the image
# ROI_CROP_TOP = 0 #45               # the number of rows of pixels to ignore on the top of the image
# ROI_CROP_BOTTOM =  90 #int(IMAGE_H/2)           # the number of rows of pixels to ignore on the bottom of the image

### add original IMU
HAVE_IMU_BNO055 = True                #when true, this add a Mpu6050 part and records the data. Can be used with a
DYNAMIC_CONTROL = False
G_MODE = "GLateral" # "GCounter" or "GVectoring" or "GLateral"
LAP_COUNT = True # if true, keras model read lap count to inference
LAP_N = 4 

### add opticalflow senser
HAVE_OPTICALFLOW_PMW3901 =True

### add YOLO model from ultralytics
HAVE_YOLO = False
IMAGE_FIELD_NAME_YOLO = 'cam1/image_array'
YOLO_MULTIPROCESS = False
YOLO_SWITCH = False
YOLO_MODEL_PATH = "models/yolo11n_20250913_202325/weights/best_openvino_model_224/best.xml"  # Updated to use OpenVINO model with 224x224 input
# YOLO_MODEL_PATH = "models/yolo11n_20250913_202325/weights/best.pt"  
YOLO_CONF_THRESHOLD = 0.3 #0.25
YOLO_IOU_THRESHOLD = 0.45
YOLO_DEVICE = 'AUTO'  # 'CPU', 'GPU', 'AUTO' for OpenVINO; 'cpu' or 'cuda' for PyTorch
YOLO_VISUALIZE = False  # Show detection results on camera feed

### IMX500 AI Camera Settings
USE_IMX500_CAMERA = False  # Enable IMX500 AI Camera with hardware inference
IMX500_MODEL_PATH = "models/yolo11n_20250913_202325/weights/best_imx_model/network.rpk"  # Path to IMX500 model
IMX500_CONF_THRESHOLD = YOLO_CONF_THRESHOLD  # Use same threshold as YOLO
IMX500_CAMERA_INDEX = 1  # Camera index for IMX500 (default: 1)

### YOLO LED Indicator Settings
YOLO_LED_ENABLE = False  # Enable YOLO LED indicator
YOLO_LED_CLASS_COLORS = {
    # Class name to RGB color mapping (based on yolo11n_20250913_202325 model)
    'right': (255, 0, 255),         # Magenta
    'left': (0, 255, 255),          # Cyan  
    'center': (255, 255, 0),        # Yellow
    'pylon': (255, 165, 0),         # Orange
    'p1': (0, 255, 0),              # Green
    'p2': (255, 0, 0),              # Red
    'p3': (0, 0, 255),              # Blue
}
YOLO_LED_DEFAULT_COLOR = (64, 64, 64)     # Dim white when no objects detected
YOLO_LED_PRIORITY_CLASSES = ['pylon', 'right', 'left', 'center', 'p1', 'p2', 'p3']  # Priority order for multiple detections

### add location model
LOCATION_MODEL_TYPE = "donkey_location" #see model_catalog.py
#LOCATION_BRAKE = True

# 位置制御設定
LOCATION_THROTTLE_CONTROL = False  # 位置制御の有効/無効
LOCATION_CONFIDENCE_THRESHOLD = 0.7  # 制御を適用する最小確信度
LOCATION_HISTORY_SIZE = 3  # 平滑化のための履歴サイズ
LOCATION_DEBUG_OUTPUT = False  # デバッグ出力の有効/無効

# 位置クラスデフォルト８
# 位置クラス別スロットル倍率
# クラス番号: 倍率（1.0=通常速度、0.5=半分の速度）
LOCATION_THROTTLE_MULTIPLIERS = {
    0: 0.8,    # s1 スタート
    1: 0.3,    # c1 brake
    2: 0.6,    # s2
    3: 0.3,    # c2
    4: 0.6,    # s3
    5: 0.3,    # c3
    6: 1.0,    # s4
    7: 0.8,    # s5
}
LOCATION_NUMBER = len(LOCATION_THROTTLE_MULTIPLIERS)

# 位置変化ブレーキルール
# (開始位置, 終了位置): ブレーキ時間(秒)
LOCATION_BRAKE_RULES = {
    (0, 1): 0.1,  
    # 必要に応じて追加
}

# LED INDICATOR
LED_INDICATOR_ATOM = False
### M5 Atom LED Mode Selection
M5_LED_MODE = 'model_switch'  # 'location', 'yolo', 'model_switch', 'off' - Choose LED display mode
LED_INDICATOR_PORT = "/dev/ttyUSB0"

### M5 Atom LED Color Configuration
# 各モードでのID毎の色設定 (RGB値: 0-255)
M5_LED_COLOR_CONFIG = {
    # 位置情報モード用の色設定 (0-7の位置に対応)
    'location': {
        0: (255, 0, 0),      # 位置0: 赤
        1: (0, 150, 0),      # 位置1: 緑
        2: (0, 0, 255),      # 位置2: 青
        3: (255, 165, 0),    # 位置3: オレンジ
        4: (128, 0, 128),    # 位置4: 紫
        5: (0, 128, 128),    # 位置5: ティール
        6: (255, 0, 255),    # 位置6: マゼンタ
        7: (128, 128, 0)     # 位置7: オリーブ
    },
    
    # モデルスイッチモード用の色設定 (MODEL_SWITCHER_MODELSから自動生成されるため不要)
    # 'model_switch': {
    #     0: (0, 255, 0),         # normal_driving: 緑
    #     1: (0, 0, 255),         # left_route: 青  
    #     2: (255, 0, 0),         # right_route: 赤
    #     3: (255, 255, 0),       # pylon_turn: 黄色
    #     4: (255, 0, 255),       # parking: マゼンタ
    # },
    
    # デフォルト色（未定義のIDの場合）
    'default': (64, 64, 64)  # グレー
}

# オプション: モデル名から自動的に色を生成するヘルパー関数
def generate_model_colors(model_names, base_colors=None):
    """
    モデル名のリストから自動的に色を生成
    
    Args:
        model_names: モデル名のリスト（例: MODEL_SWITCHER_MODELS.keys()）
        base_colors: ベースとなる色のリスト（省略時はデフォルト色セットを使用）
    
    Returns:
        モデル名と色の辞書
    """
    if base_colors is None:
        # デフォルトの色セット（視認性の高い色）
        base_colors = [
            (255, 0, 0),     # 赤
            (0, 255, 0),     # 緑
            (0, 0, 255),     # 青
            (255, 255, 0),   # 黄色
            (255, 0, 255),   # マゼンタ
            (0, 255, 255),   # シアン
            (255, 128, 0),   # オレンジ
            (128, 0, 255),   # 紫
            (255, 255, 255), # 白
            (128, 255, 0),   # ライム
            (255, 0, 128),   # ピンク
            (0, 128, 255),   # スカイブルー
        ]
    
    colors = {}
    for i, model_name in enumerate(model_names):
        colors[model_name] = base_colors[i % len(base_colors)]
    return colors

# ============================================================================
# Wall Following Controller Settings
# ============================================================================
# Wall following controller maximum steering angle (degrees)
# Positive angle = right turn, Negative angle = left turn
WALL_FOLLOWING_MAX_STEERING_ANGLE = 30.0  # Maximum steering angle in degrees

# Pure Pursuit Controller Settings
WALL_FOLLOWING_LOOKAHEAD_DISTANCE = 600  # Lookahead distance in mm
WALL_FOLLOWING_MIN_LOOKAHEAD = 300       # Minimum lookahead distance in mm  
WALL_FOLLOWING_MAX_LOOKAHEAD = 1000      # Maximum lookahead distance in mm
WALL_FOLLOWING_DYNAMIC_LOOKAHEAD = True  # Enable dynamic lookahead adjustment

# Wall Following Distance Settings
WALL_FOLLOWING_TARGET_DISTANCE = 100     # Target distance from wall in mm
WALL_FOLLOWING_VEHICLE_HALF_WIDTH = 100  # Vehicle half width in mm
WALL_FOLLOWING_PATH_LENGTH = 2000        # Path length for waypoint generation in mm
WALL_FOLLOWING_WAYPOINT_SPACING = 100    # Spacing between waypoints in mm

# ============================================================================
# Model Switcher Settings - 複数モデル切り替え設定
# ============================================================================

# ModelSwitcher有効化フラグ
ENABLE_MODEL_SWITCHER = True # 2025ミニバト用

# ジャイロルールベース制御有効化フラグ
ENABLE_GYRO_RULE = False # 2025ミニバト用

# 基本的な切り替えパラメータ
MODEL_SWITCH_SPEED_THRESHOLD = 0.5      # 速度による切り替え閾値
MODEL_SWITCH_STEERING_THRESHOLD = 0.3   # ステアリング角度による切り替え閾値

# モデル定義 - 用途別にモデルを設定
MODEL_SWITCHER_MODELS = {
    # 通常走行用モデル（デフォルト）
    'normal_driving': {        
        'path': 'models/donkeycar_20251110_173421_best_openvino.xml', 
        # 'path': 'models/donkeycar_20251101_120111_best_openvino.xml', #new OK
        # 'path': 'models/donkeycar_20251023_101020_new_cslow_best_openvino.xml', #new OK
        # 'path': 'models/donkeycar_20251023_101020_new_cslow.pth', #new OK
        ###'path': 'models/donkeycar_20251018_230547_cfast_openvino.xml', #new OK
        # 'path': 'models/donkeycar_20251018_230547_cfast.pth', #w/ AI_throtle 0.9
        ### 'path': 'models/donkeycar_20251019_023221_clr_lidar.pth', #all routes
        'model_type': 'donkeycar',
        'priority': 10,
        'is_default': True,
        'pre_delay': 0.0,   # モデル切り替え前のdelay（秒）
        'post_delay': 0.0,  # モデル切り替え後のdelay（秒）
        'led_color': (255, 255, 0),  # LED色: 黄色
        'conditions': {
            'route_type': 'normal',
            'speed': {'min': -1.0, 'max': 1.0}
        },
        'description': '通常走行用モデル、中央ルート'
    },
    
    # 左ルート用モデル
    'left_route': {
        'path': 'models/donkeycar_20251019_000746_l_best_openvino.xml', #best
        # 'path': 'models/donkeycar_20251019_220750_l.pth', #best
        ### 'path': 'models/donkeycar_20251022_175807_l.pth', #
        'model_type': 'donkeycar',
        'priority': 15,
        'pre_delay': 0.0,   # モデル切り替え前のdelay（秒）
        'post_delay': 2.0,  # モデル切り替え後のdelay（秒）
        'led_color': (0, 255, 255),  # LED色: シアン
        'conditions': {
            'route_type': 'left'
        },
        'description': '左ルート分岐路走行'
    },
    
    # 右ルート用モデル
    'right_route': {        
        'path': 'models/donkeycar_20251022_175153_r_openvino.xml', #best
        # 'path': 'models/donkeycar_20251022_175153_r.pth', #best
        #'path': 'models/donkeycar_20251019_005633_r.pth', #ok
        'model_type': 'donkeycar',
        'priority': 15,
        'pre_delay': 0.0,   # モデル切り替え前のdelay（秒）
        'post_delay': 2.0,  # モデル切り替え後のdelay（秒）
        'led_color': (255, 0, 255),  # LED色: マゼンタ
        'conditions': {
            'route_type': 'right'
        },
        'description': '右ルート分岐路走行'
    },
    
    # パイロン旋回用モデル
    'pylon_turn': {
        'path': 'models/donkeycar_20251110_173421_best_openvino.xml', 
        # 'path': 'models/donkeycar_20251023_101020_new_cslow_best_openvino.xml', #new OK
        # 'path': 'models/donkeycar_20251012_154526_c.pth',
        ### 'path': 'models/donkeycar_20251018_230547_cfast.pth',
        'model_type': 'donkeycar',
        'priority': 20,
        'pre_delay': 0.0,  # モデル切り替え前のdelay（秒）
        'post_delay': 0.0,  # モデル切り替え後のdelay（秒）
        'led_color': (199, 177, 131),  # LED色: 茶色
        'conditions': {
            'route_type': 'pylon',
            'obstacle_detected': True
        },
        'description': 'パイロン旋回走行'
    },
    
    # 駐車用モデル
    'parking': {        
        'path': 'models/donkeycar_20251027_153226_citypark_best_openvino.xml', # best
        # 'path': 'models/donkeycar_20251023_220934_p1_best_openvino.xml', # new p1
        # 'path': 'models/donkeycar_20251023_220934_p1.pth', # new p1
        # 'path': 'models/donkeycar_20251019_040333_p.pth', # best
        ### 'path': 'models/donkeycar_20251016_195122_p.pth', # 
        'model_type': 'donkeycar',
        'priority': 25,
        'pre_delay': 0.3,   # モデル切り替え前のdelay（秒）
        'post_delay': 0.0,  # モデル切り替え後のdelay（秒）
        'led_color': (255, 0, 0),  # LED色: 赤
        'conditions': {
            'route_type': 'parking',
            'parking_time_s': 120 #50
        },
        'description': '駐車走行'
    }
}

# モデル切り替えのキーマッピング（リモコンやキーボード入力用）
MODEL_SWITCHER_KEY_MAPPING = {
    0: 'normal_driving',    # デフォルト
    1: 'left_route',        # 左ルート
    2: 'right_route',       # 右ルート
    3: 'pylon_turn',        # パイロン旋回
    4: 'parking',           # 駐車
}

# カラーフィルターインデックスからモデルIDへのマッピング
# FILTER_NAMES = ['Magenta', 'Cyan', 'Yellow'] に対応
COLOR_FILTER_TO_MODEL_ID = {
    0: 2,  # Magenta (マゼンタ, 右) → right_route (MODEL_SWITCHER_KEY_MAPPING[2])
    1: 1#,  # Cyan (シアン, 左) → left_route (MODEL_SWITCHER_KEY_MAPPING[1])
    #2: 3   # Pink #Yellow (イエロー, パイロン) → pylon_turn (MODEL_SWITCHER_KEY_MAPPING[3])
}

# 自動切り替え設定
MODEL_SWITCHER_AUTO_SWITCH = True       # 条件による自動切り替えを有効化
MODEL_SWITCH_COOLDOWN_DURATION = 3.0    # モデル切り替え後の切り替え不可冷却期間（秒）

# ルートタイプ検出設定（将来的な拡張用）
ROUTE_DETECTION_ENABLED = False
ROUTE_DETECTION_CONFIDENCE_THRESHOLD = 0.8

# モデル切り替えログ設定
MODEL_SWITCHER_LOG_SWITCHES = True      # 切り替えをログに記録
MODEL_SWITCHER_LOG_LEVEL = 'INFO'       # ログレベル

# Web UI表示設定
MODEL_SWITCHER_WEB_DISPLAY = False       # Web UIでの表示を有効化
MODEL_SWITCHER_SHOW_PERFORMANCE = False  # 推論時間等の表示

# ============================================================================
# End of Model Switcher Settings
# ============================================================================
