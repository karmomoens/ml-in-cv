import IPython
import PIL.Image
import cv2
import sys
from os.path import split
from io import BytesIO
from IPython.display import clear_output
from time import sleep


def construct_stream():
    WEBCAM_DEVICE_INDEX = 0
    return cv2.VideoCapture(WEBCAM_DEVICE_INDEX)


def process_on_webcam(process_function=lambda _: None, final_message="Stream stopped", finalize_function=lambda: None):
    source = construct_stream()
    try:
        while True:
                ret, frame = source.read()
                if frame is None:
                    clear_output(wait=True)
                    print("No valid camera frames")
                    sleep(0.1)
                    continue
                output = process_function(frame)
                if output is not None:
                    frame = output
                showarray(frame)
    except Exception as e:
        print(e)
        info = sys.exc_info()
        exception_type = info[0]
        trace_back = info[2]
        filename = split(trace_back.tb_frame.f_code.co_filename)[1]
        line_number = 1 + trace_back.tb_lineno
        print(exception_type)
        print(filename)
        print(line_number)
        print("")
    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        print(final_message)
        finalize_function()


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

