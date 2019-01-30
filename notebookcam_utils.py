import IPython
import PIL.Image
import cv2
from io import BytesIO
from IPython.display import clear_output
from time import sleep
from traceback import print_exc


def construct_stream():
    WEBCAM_DEVICE_INDEX = 0
    return cv2.VideoCapture(WEBCAM_DEVICE_INDEX)


def process_on_webcam(process_function=lambda _: None, final_message="Stream stopped", finalize_function=lambda: None):
    source = construct_stream()
    try:
        while True:
                ret, frame = source.read()
                if frame is None:
                    wait_for_valid_frame()
                    continue
                output = process_function(frame)
                if output is not None:
                    frame = output
                showarray(frame)
    except Exception as e:
        print_exc()
    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        print(final_message)
        finalize_function()
        
        
def wait_for_valid_frame():
    clear_output(wait=True)
    print("No valid camera frames")
    sleep(0.1)


def showarray(array, displayables=[], fmt='jpeg', is_rgb=False):
    if not is_rgb:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    f = BytesIO()
    something = PIL.Image.fromarray(array)
    something.save(f, fmt)
    clear_output(wait=True)
    for d in displayables:
        display(d)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))

