# -*- coding: utf-8 -*-
"""
@author: Valeri
"""
import winsound
import os, sys, inspect, thread, time, random
src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
lib_dir = os.path.abspath(os.path.join(src_dir, '../LeapDeveloperKit_2.3.1+31549_win/LeapSDK/lib'))
sys.path.insert(0, lib_dir)
arch_dir = '../LeapDeveloperKit_2.3.1+31549_win/LeapSDK/lib/x64' if sys.maxsize > 2**32 else '../LeapDeveloperKit_2.3.1+31549_win/LeapSDK/lib/x86'
sys.path.insert(0, os.path.abspath(os.path.join(src_dir, arch_dir)))

import Leap
from Leap import * 


signs = ["rock", "paper", "scissors"]
period_length = 2

class SampleListener(Leap.Listener):
    sign = time.clock() + period_length
    blank = time.clock() + period_length*2
    stage = ""
    f = open("data.csv",'w')

    def on_connect(self, controller):
        print("Connected")
        controller.enable_gesture(Leap.Gesture.TYPE_SWIPE)


    def on_frame(self, controller):
        
        frame = controller.frame()
        #print "Frame id: %d, timestamp: %d, hands: %d, fingers: %d, tools: %d, gestures: %d" % (
        #  frame.id, frame.timestamp, len(frame.hands), len(frame.fingers), len(frame.tools), len(frame.gestures()))
        tm = time.clock()
        if self.sign > tm:
            if self.stage == " ":
                winsound.Beep(300, 200)
                self.stage = signs[random.randint(0, len(signs)-1)]
            print(self.stage, " ", tm, " ", self.sign)
        else:
            if not self.stage == "":
                if len(frame.hands) == 1:
                    hand = frame.hands[0]
                    line = self.stage
                    thumb = hand.fingers.finger_type(Finger.TYPE_THUMB)[0]
                    line = line + ", " + str(thumb.direction.x) + ", " + str(thumb.direction.y) + ", " + str(thumb.direction.z)
                    index = hand.fingers.finger_type(Finger.TYPE_INDEX)[0]
                    line = line + ", " + str(index.direction.x) + ", " + str(index.direction.y) + ", " + str(index.direction.z)
                    middle = hand.fingers.finger_type(Finger.TYPE_MIDDLE)[0]
                    line = line + ", " + str(middle.direction.x) + ", " + str(middle.direction.y) + ", " + str(middle.direction.z)
                    ring = hand.fingers.finger_type(Finger.TYPE_RING)[0]
                    line = line + ", " + str(ring.direction.x) + ", " + str(ring.direction.y) + ", " + str(ring.direction.z)
                    pinky = hand.fingers.finger_type(Finger.TYPE_PINKY)[0]
                    line = line + ", " + str(pinky.direction.x) + ", " + str(pinky.direction.y) + ", " + str(pinky.direction.z) + "\n"
                    self.f.write(line)
                    print(line)
                self.stage = ""
            print("BLANK!", " ", tm, " ", self.blank)
            if self.blank < tm:
                self.blank = tm + 2*period_length
                self.sign = tm + period_length
            

def main():
    # Create a sample listener and controller
    listener = SampleListener()
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)

    # Keep this process running until Enter is pressed
    print("Press Enter to quit...")
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        listener.f.close()
        controller.remove_listener(listener)


if __name__ == "__main__":
    main()
