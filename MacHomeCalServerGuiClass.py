from Tkinter import *
from PIL import ImageTk, Image
import socket
import base64
from subprocess import *
import time
import pexpect
import threading
import Pmw
import os
import sys


class BackgroundW(Tk):
    def __init__(self, parent):
        Tk.__init__(self, parent)
        self.parent = parent
        self.title("Calibration Home Screen")
        self.geometry("400x135")
        self.resizable(False, False)

        self.Home_Frame = HomeF(self)
        self.Home_Frame.grid(row=0)
        #self.Home_Frame.grid(row=0, column=0)


class HomeF(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.InitFrames()

    def InitFrames(self):
        self.instructions = Label(text="Please select a calibration setting")

        #self.psi_image = Image.open("psi_icon.png")
        self.psi_image = Image.open(os.path.join(sys._MEIPASS, "psi_icon.png"))
        self.psi_photo = ImageTk.PhotoImage(self.psi_image)

        #self.offset_image = Image.open("offset_icon.png")
        self.offset_image = Image.open(os.path.join(sys._MEIPASS, "offset_icon.png"))
        self.offset_photo = ImageTk.PhotoImage(self.offset_image)

        self.button_frame = Frame(height=10, width=40)
        self.psi_button = Button(self.button_frame, image=self.psi_photo, command=lambda: self.PsiClicked())
        self.offset_button = Button(self.button_frame, image=self.offset_photo, command=lambda: self.OffsetClicked())

        self.instructions.grid(row=0, column=0, padx=90)
        self.button_frame.grid(row=1, column=0)
        self.psi_button.grid(row=1, column=0)
        self.offset_button.grid(row=1, column=1)

    def PsiClicked(self):
        self.parent.title("PSI Calibration")
        self.update()
        self.parent.Home_Frame.destroy()

        self.parent.Psi_Frame = PsiF(self.parent)
        self.parent.Psi_Frame.grid(row=0, column=0)
        #self.parent.Psi_Frame.configure(background='red')

    def OffsetClicked(self):
        self.parent.title("Offset Calibration")
        self.update()
        self.parent.Home_Frame.destroy()

        self.parent.Offset_Frame = OffsetF(self.parent)
        self.parent.Offset_Frame.grid(row=0, column=0)
        #self.parent.Offset_Frame.configure(background='blue')


class PsiF(Frame):
    def __init__(self,parent):
        Frame.__init__(self,parent)
        self.parent = parent
        self.TCP_PORT = 5050
        self.BUFFER_SIZE = 1024

        # Initialize messages to display to user
        self.message = StringVar()
        #self.message.set("Program is ready to start")

        # Initialize ip address variable
        self.display_mac_eth_ip = StringVar()

        # Setup layout of GUI
        self.InitFrames()
        self.PlaceFrames()

        # Get ip address and update GUI with connection status
        self.get_ip()

    def InitFrames(self):
        # Initialize popup balloons
        self.balloon = Pmw.Balloon(self)

        # Help popop box
        self.help = Label(self, text="Help")
        help_title = "Setup Instructions\n"
        help_1 = "1. Connect ethernet cable to the computer and raspberry pi and click connect.\n"
        help_2 = "2. If this is your first time setting up or if you are switching to another computer, set a custom ip address.\n"
        help_3 = "3. On the raspberry pi, check the box to process on computer.\n"
        help_4 = "4. On the raspberry pi, go to the settings page and enter the ip address displayed on the computer.\n"
        help_5 = "5. On the computer, click start before running PSI Calibration on the raspberry pi.\n"
        help_6 = "6. After calibration is complete, click stop to stop the program."
        help_messages = help_title + help_1 + help_2 + help_3 + help_4 + help_5 + help_6
        self.balloon.bind(self.help, help_messages)

        # Display ip address
        self.ip = Label(self, textvariable=self.display_mac_eth_ip)

        # Set ip address
        self.set_ip_frame = Frame(self, height=50, width=200)
        self.set_ip_label = Label(self.set_ip_frame, text="Set Custom IP:")
        self.enter_ip = Entry(self.set_ip_frame, width=12, validate="focusin")
        self.set_ip_button = Button(self.set_ip_frame, text="Set", command=self.set_ip)

        # Buttons
        self.back_button = Button(self, text="Back", command=self.go_home)
        self.button_frame = Frame(self, height=50, width=200)
        self.connect_button = Button(self, text="Connect", width=7)
        self.connect_button.config(command=lambda: self.get_ip())
        self.start_button = Button(self.button_frame, text="Start")
        self.start_button.config(command=self.run_script, width=5)
        self.stop_button = Button(self.button_frame, text="Stop", state=DISABLED)
        self.stop_button.config(command=self.stop_script, width=5)

        self.feedback = Label(self, textvariable=self.message)

    def PlaceFrames(self):
        self.help.grid(row=0, column=0, sticky=E, padx=10)
        self.ip.grid(row=0, column=0)

        self.set_ip_frame.grid(row=1, column=0, padx=67)
        self.set_ip_label.grid(row=1, column=0)
        self.enter_ip.grid(row=1, column=1)
        self.set_ip_button.grid(row=1, column=2)

        self.back_button.grid(row=0, column=0, sticky=W, padx=5)
        self.connect_button.grid(row=2, column=0)
        self.button_frame.grid(row=3, column=0)
        self.start_button.grid(row=3, column=0)
        self.stop_button.grid(row=3, column=1)

        self.feedback.grid(row=4, column=0, padx=20)

    def go_home(self):
        self.parent.Psi_Frame.destroy()
        #self.parent.Home_Frame = HomeF(self.parent)
        #self.parent.Home_Frame.grid(row=0, column=0)

    # Becomes server and gets the parameters and ip from pi
    def read_from_pi(self):
        print("Starting socket connection")
        # Create an INET, STREAMing socket
        serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Allow the socket to use same PORT address
        serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Set timeout for 3 seconds
        serverSocket.settimeout(3)
        # Bind the socket to host and a port.
        # '' means socket is reachable by any address the machine happens to have
        serverSocket.bind(('', self.TCP_PORT))
        print("Socket hostname:", socket.gethostname())

        # Stay in this loop while the thread attribute is true
        while self.thread.do_run:
            # Want to catch timeout error
            try:
                # Become a server socket
                serverSocket.listen(1)
                print("Waiting for a connection...")
                # Accept connections from outside
                (clientsocket, addr) = serverSocket.accept()
                print('Connection address:', addr)

                # When connection has been established and data is transferring, disable stop button
                self.stop_button.config(state=DISABLED)

                self.message.set("Receiving data from pi")
                self.update()
                data = ""
                while 1:
                    # Let the socket sleep during exeption to free up resource
                    try:
                        data += clientsocket.recv(self.BUFFER_SIZE)
                        if not data: break
                        if data[-3:] == 'END': break
                    except socket.error, e:
                        # Handle resource temporarily unavilable exception
                        if "[Errno 35]" in str(e):
                            time.sleep(0.1)
                        # Handle address already in use error
                        elif "[Errno 10035]" in str(e):
                            time.sleep(0.1)
                        else:
                            raise e

                clientsocket.close()
                # print "Data:", data

                # Organize data into variables
                self.parameters = data.split(',')
                self.image = self.parameters[0]
                self.width = self.parameters[1]
                self.desiredWidth = self.parameters[2]
                self.spsi = self.parameters[3]
                self.ppmm = self.parameters[4]
                self.margin = self.parameters[5]
                self.pi_eth_ip = self.parameters[6]
                print("Parameters:", (self.width, self.desiredWidth, self.spsi, self.ppmm, self.margin))

                # Decode the image data
                decoded_data = base64.b64decode(self.image)
                # Create writable image and write the decoding result
                image_result = open(os.path.join(sys._MEIPASS, 'image_decode.png'), 'wb')
                #image_result = open(os.path.join(sys._MEIPASS, 'image_decode.png'), 'wb')
                image_result.write(decoded_data)
                break

                #return (self.width, self.desiredWidth, self.spsi, self.ppmm, self.margin, self.pi_eth_ip)
            except socket.error, e:
                print "Error:", e


    # Run psi cal and get the values
    def psi_cal(self):
        print("Running PSI Calibration Code...")
        self.message.set("Running PSI calibration algorithm")
        self.update()

        start_time = time.time()
        # Want to catch excpetion when there is a bad picture
        try:
            #self.offset_list = check_output(['./saveme1', '--image', 'image_decode.png', '--width', self.width, '--desiredwidth', self.desiredWidth, '--spsi', self.spsi, '--ppmm', self.ppmm, '--margin', self.margin])
            self.offset_list = check_output([os.path.join(sys._MEIPASS, 'saveme1'), '--image', os.path.join(sys._MEIPASS, 'image_decode.png'), '--width', self.width, '--desiredwidth', self.desiredWidth, '--spsi', self.spsi, '--ppmm', self.ppmm, '--margin', self.margin])

            print("PSI calibration code took", time.time() - start_time, "seconds")
            print("Output:", self.offset_list)
            #return offset_list
        except Exception as e:
            print "Algorithm failed. Please restart PSI calibration."
            print e
            self.message.set("Algorithm failed. Please restart PSI calibration.")
            self.update()
            self.stop_button.config(state=NORMAL)


    # Becomes client and sends the values back to the pi
    def send_to_pi(self):
        # Format data to send to pi
        self.offset_list = self.offset_list[1:-2]
        self.offset_list += "END"
        print("Data to send:", self.offset_list)

        # Creating socket connection to send data
        print("Sending data back to PI...")
        clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print "Connecting to", self.pi_eth_ip, "at port", self.TCP_PORT
        clientSocket.connect((self.pi_eth_ip, self.TCP_PORT))
        start_time = time.time()
        clientSocket.send(self.offset_list)
        clientSocket.close()
        print("Successfully sent data in", time.time() - start_time, "seconds")

    # Get mac's ethernet ip address
    def get_ip(self):
        # Run command to list connected hardware ports
        self.output = check_output(['networksetup', '-listallhardwareports'])
        self.output = self.output[1:-1]
        self.output = self.output.split('\n\n')
        self.device = ""
        self.interfaceName = ""
        self.mac_eth_ip = ""

        # Grab the ethernet port number
        for port in self.output:
            port = port.split('\n')
            # Let it check for thunderbolt or usb connection
            if "Thunderbolt Ethernet" in port[0] or " LAN" in port[0]:
                self.interfaceName = port[0].split(': ')[1]
                self.device = port[1].split(' ')[1]
                break

        # Successful Connection
        try:
            # Get the ip address associated with the ethernet port number
            self.mac_eth_ip = check_output(['ipconfig', 'getifaddr', self.device])[:-1]
            print("Ethernet cable is connected to Pi")

            # Update message and button, and display the ip
            self.message.set("Program is ready to start")
            self.start_button.config(state=NORMAL)
            self.set_ip_button.config(state=NORMAL)
            self.enter_ip.config(state=NORMAL)
            self.display_mac_eth_ip.set("My IP: " + self.mac_eth_ip)
            self.update()

            return self.mac_eth_ip

        # Unsuccessful Connection
        except Exception:
            print("Ethernet cable is not connected to Pi")
            self.mac_eth_ip = "N/A"

            # Update message and buttons, and display N/A ip
            self.message.set("Ethernet cable is not connected to Pi")
            self.start_button.config(state=DISABLED)
            self.set_ip_button.config(state=DISABLED)
            self.enter_ip.config(state=DISABLED)
            self.connect_button.config(state=NORMAL)
            self.display_mac_eth_ip.set("My IP: " + self.mac_eth_ip)
            self.update()

            return self.mac_eth_ip


    # Manually set the ip address
    def set_ip(self):
        self.interfaceName = self.interfaceName.replace(' ', '\ ')
        cmd = 'networksetup -setmanual ' + self.interfaceName + ' ' + self.enter_ip.get() + ' 255.255.255.0 8.8.8.8'
        child = pexpect.spawn(cmd)
        child.expect(pexpect.EOF)
        print "Set ip address to", self.enter_ip.get()
        self.interfaceName = self.interfaceName.replace('\ ', '')

        # Display new ip address
        self.set_ip_button.config(state=DISABLED)
        start_time = time.time()
        while self.get_ip() == "N/A":
            if time.time() - start_time > 3:
                self.message.set("Unable to set Ip, make sure ethernet cable is connected")
                self.update()
                break
            self.get_ip()
        self.set_ip_button.config(state=NORMAL)

    # Runs the entire script when start is clicked
    def run_script(self):
        # Configure GUI
        self.start_button.config(state=DISABLED)
        self.enter_ip.config(state=DISABLED)
        self.set_ip_button.config(state=DISABLED)
        self.connect_button.config(state=DISABLED)
        self.stop_button.config(state=NORMAL)
        self.message.set("Program running... continue PSI calibration on the Pi")
        self.update()

        # Use a thread to run process in background to prevent Tkinter gui from freezing
        self.thread = threading.Thread(name="Read", target=self.run)
        # Set thread as daemon so thread is terminated when main thread ends
        self.thread.daemon = True
        self.thread.start()
        self.thread.do_run = True

    # Thread that runs server
    def run(self):
        self.currThread = threading.currentThread()
        self.back_button.config(state=DISABLED)
        # Run script while the thread attribute is true
        while getattr(self.currThread, "do_run", True):
            # Want to catch return value error when thread ends and break loop
            try:
                self.read_from_pi()
                if self.currThread.do_run == False: break
                self.psi_cal()
                self.send_to_pi()

                self.message.set("PSI calibration is finished")
                self.update()
            except TypeError, e:
                print "Error:", e
                break

            self.stop_button.config(state=NORMAL)

    # Stops the thread process when stop is clicked
    def stop_script(self):
        self.message.set("Stopping server...")
        self.update()
        # Toggle the buttons
        self.back_button.config(state=NORMAL)
        self.stop_button.config(state=DISABLED)
        self.connect_button.config(state=NORMAL)
        # Set the thread attribute to False to signal stop
        self.thread.do_run = False
        # Join thread with main thread to end thread
        threading.Thread.join(self.thread, 1)
        self.message.set("Stopped server")
        self.update()


class OffsetF(Frame):
    def __init__(self,parent):
        Frame.__init__(self,parent)
        self.parent = parent
        self.TCP_PORT = 5050
        self.BUFFER_SIZE = 1024

        # Initialize messages to display to user
        self.message = StringVar()
        #self.message.set("Program is ready to start")

        # Initialize ip address variable
        self.display_mac_eth_ip = StringVar()

        # Setup layout of GUI
        self.InitFrames()
        self.PlaceFrames()

        # Get ip address and update GUI with connection status
        self.get_ip()

    def InitFrames(self):
        # Initialize popup balloons
        self.balloon = Pmw.Balloon(self)

        # Help popop box
        self.help = Label(self, text="Help")
        help_title = "Setup Instructions\n"
        help_1 = "1. Connect ethernet cable to the computer and raspberry pi and click connect.\n"
        help_2 = "2. If this is your first time setting up or if you are switching to another computer, set a custom ip address.\n"
        help_3 = "3. On the raspberry pi, check the box to process on computer.\n"
        help_4 = "4. On the raspberry pi, go to the settings page and enter the ip address displayed on the computer.\n"
        help_5 = "5. On the computer, click start before running Offset Calibration on the raspberry pi.\n"
        help_6 = "6. After calibration is complete, click stop to stop the program."
        help_messages = help_title + help_1 + help_2 + help_3 + help_4 + help_5 + help_6
        self.balloon.bind(self.help, help_messages)

        # Display ip address
        self.ip = Label(self, textvariable=self.display_mac_eth_ip)

        # Set ip address
        self.set_ip_frame = Frame(self, height=50, width=200)
        self.set_ip_label = Label(self.set_ip_frame, text="Set Custom IP:")
        self.enter_ip = Entry(self.set_ip_frame, width=12, validate="focusin")
        self.set_ip_button = Button(self.set_ip_frame, text="Set", command=self.set_ip)

        # Buttons
        self.back_button = Button(self, text="Back", command=self.go_home)
        self.button_frame = Frame(self, height=50, width=200)
        self.connect_button = Button(self, text="Connect", width=7)
        self.connect_button.config(command=lambda: self.get_ip())
        self.start_button = Button(self.button_frame, text="Start")
        self.start_button.config(command=self.run_script, width=5)
        self.stop_button = Button(self.button_frame, text="Stop", state=DISABLED)
        self.stop_button.config(command=self.stop_script, width=5)

        self.feedback = Label(self, textvariable=self.message)

    def PlaceFrames(self):
        self.help.grid(row=0, column=0, sticky=E, padx=10)
        self.ip.grid(row=0, column=0)

        self.set_ip_frame.grid(row=1, column=0, padx=67)
        self.set_ip_label.grid(row=1, column=0)
        self.enter_ip.grid(row=1, column=1)
        self.set_ip_button.grid(row=1, column=2)

        self.back_button.grid(row=0, column=0, sticky=W, padx=5)
        self.connect_button.grid(row=2, column=0)
        self.button_frame.grid(row=3, column=0)
        self.start_button.grid(row=3, column=0)
        self.stop_button.grid(row=3, column=1)

        self.feedback.grid(row=4, column=0, padx=20)

    def go_home(self):
        self.parent.Offset_Frame.destroy()
        #self.parent.Home_Frame = HomeF(self.parent)
        #self.parent.Home_Frame.grid(row=0, column=0)

    # Becomes server and gets the parameters and ip from pi
    def read_from_pi(self):
        print("Starting socket connection")
        # Create an INET, STREAMing socket
        serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Allow the socket to use same PORT address
        serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Set timeout for 3 seconds
        serverSocket.settimeout(3)
        # Bind the socket to host and a port.
        # '' means socket is reachable by any address the machine happens to have
        serverSocket.bind(('', self.TCP_PORT))
        print("Socket hostname:", socket.gethostname())

        # Stay in this loop while the thread attribute is true
        while self.thread.do_run:
            # Want to catch timeout error
            try:
                # Become a server socket
                serverSocket.listen(1)
                print("Waiting for a connection...")
                # Accept connections from outside
                (clientsocket, addr) = serverSocket.accept()
                print('Connection address:', addr)

                # When connection has been established and data is transferring, disable stop button
                self.stop_button.config(state=DISABLED)

                self.message.set("Receiving data from pi")
                self.update()
                data = ""
                while 1:
                    # Let the socket sleep during exeption to free up resource
                    try:
                        data += clientsocket.recv(self.BUFFER_SIZE)
                        if not data: break
                        if data[-3:] == 'END': break
                    except socket.error, e:
                        # Handle resource temporarily unavilable exception
                        if "[Errno 35]" in str(e):
                            time.sleep(0.1)
                        # Handle address already in use error
                        elif "[Errno 10035]" in str(e):
                            time.sleep(0.1)
                        else:
                            raise e

                clientsocket.close()
                # print "Data:", data

                # Organize data into variables
                self.parameters = data.split(',')
                self.image1 = self.parameters[0]
                self.image2 = self.parameters[1]
                self.refwidth = self.parameters[2]
                self.sdist = self.parameters[3]
                self.jump = self.parameters[4]
                self.pi_eth_ip = self.parameters[5]
                print "Parameters:", (self.refwidth, self.sdist, self.jump)
                # Decode the image data
                decoded_data1 = base64.b64decode(self.image1)
                decoded_data2 = base64.b64decode(self.image2)
                # Create writable image and write the decoding result
                #image_result = open('image_decode_ref.png', 'wb')
                image_result = open(os.path.join(sys._MEIPASS, 'image_decode_ref.png'), 'wb')
                image_result.write(decoded_data1)
                #image_result = open('image_decode_cal.png', 'wb')
                image_result = open(os.path.join(sys._MEIPASS, 'image_decode_cal.png'), 'wb')
                image_result.write(decoded_data2)
                break

                #return (self.width, self.desiredWidth, self.spsi, self.ppmm, self.margin, self.pi_eth_ip)
            except socket.error, e:
                print "Error:", e


    # Run Offset cal and get the values
    def offset_cal(self):
        print("Running Offset Calibration Code...")
        self.message.set("Running offset calibration algorithm")
        self.update()

        start_time = time.time()
        # Want to catch excpetion when there is a bad picture
        try:
            #self.offset_list = check_output(['python2', 'mul_proc_offsetv4.py', '--image1', 'image_decode_ref.png', '--image2', 'image_decode_cal.png', '--refwidth', self.refwidth, '--sdist', self.sdist, '--jump', self.jump])
            self.offset_list = check_output([os.path.join(sys._MEIPASS, 'mul_proc_offsetv4'), '--image1', os.path.join(sys._MEIPASS, 'image_decode_ref.png'), '--image2', os.path.join(sys._MEIPASS, 'image_decode_cal.png'), '--refwidth', self.refwidth, '--sdist', self.sdist, '--jump', self.jump])
            print("Offset calibration code took", time.time() - start_time, "seconds")
            print("Output:", self.offset_list)
            #return offset_list
        except Exception as e:
            print "Algorithm failed. Please restart offset calibration."
            print e
            self.message.set("Algorithm failed. Please restart offset calibration.")
            self.update()
            self.stop_button.config(state=NORMAL)


    # Becomes client and sends the values back to the pi
    def send_to_pi(self):
        # Format data to send to pi
        self.offset_list = self.offset_list[1:-2]
        self.offset_list += "END"
        print("Data to send:", self.offset_list)

        # Creating socket connection to send data
        print("Sending data back to PI...")
        clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print "Connecting to", self.pi_eth_ip, "at port", self.TCP_PORT
        clientSocket.connect((self.pi_eth_ip, self.TCP_PORT))
        start_time = time.time()
        clientSocket.send(self.offset_list)
        clientSocket.close()
        print("Successfully sent data in", time.time() - start_time, "seconds")

    # Get mac's ethernet ip address
    def get_ip(self):
        # Run command to list connected hardware ports
        self.output = check_output(['networksetup', '-listallhardwareports'])
        self.output = self.output[1:-1]
        self.output = self.output.split('\n\n')
        self.device = ""
        self.interfaceName = ""
        self.mac_eth_ip = ""

        # Grab the ethernet port number
        for port in self.output:
            port = port.split('\n')
            # Let it check for thunderbolt or usb connection
            if "Thunderbolt Ethernet" in port[0] or " LAN" in port[0]:
                self.interfaceName = port[0].split(': ')[1]
                self.device = port[1].split(' ')[1]
                break

        # Successful Connection
        try:
            # Get the ip address associated with the ethernet port number
            self.mac_eth_ip = check_output(['ipconfig', 'getifaddr', self.device])[:-1]
            print("Ethernet cable is connected to Pi")

            # Update message and button, and display the ip
            self.message.set("Program is ready to start")
            self.start_button.config(state=NORMAL)
            self.set_ip_button.config(state=NORMAL)
            self.enter_ip.config(state=NORMAL)
            self.display_mac_eth_ip.set("My IP: " + self.mac_eth_ip)
            self.update()

            return self.mac_eth_ip

        # Unsuccessful Connection
        except Exception:
            print("Ethernet cable is not connected to Pi")
            self.mac_eth_ip = "N/A"

            # Update message and buttons, and display N/A ip
            self.message.set("Ethernet cable is not connected to Pi")
            self.start_button.config(state=DISABLED)
            self.set_ip_button.config(state=DISABLED)
            self.enter_ip.config(state=DISABLED)
            self.connect_button.config(state=NORMAL)
            self.display_mac_eth_ip.set("My IP: " + self.mac_eth_ip)
            self.update()

            return self.mac_eth_ip


    # Manually set the ip address
    def set_ip(self):
        self.interfaceName = self.interfaceName.replace(' ', '\ ')
        cmd = 'networksetup -setmanual ' + self.interfaceName + ' ' + self.enter_ip.get() + ' 255.255.255.0 8.8.8.8'
        child = pexpect.spawn(cmd)
        child.expect(pexpect.EOF)
        print "Set ip address to", self.enter_ip.get()
        self.interfaceName = self.interfaceName.replace('\ ', '')

        # Display new ip address
        self.set_ip_button.config(state=DISABLED)
        start_time = time.time()
        while self.get_ip() == "N/A":
            if time.time() - start_time > 3:
                self.message.set("Unable to set Ip, make sure ethernet cable is connected")
                self.update()
                break
            self.get_ip()
        self.set_ip_button.config(state=NORMAL)

    # Runs the entire script when start is clicked
    def run_script(self):
        # Configure GUI
        self.start_button.config(state=DISABLED)
        self.enter_ip.config(state=DISABLED)
        self.set_ip_button.config(state=DISABLED)
        self.connect_button.config(state=DISABLED)
        self.stop_button.config(state=NORMAL)
        self.message.set("Program running... continue offset calibration on the Pi")
        self.update()

        # Use a thread to run process in background to prevent Tkinter gui from freezing
        self.thread = threading.Thread(name="Read", target=self.run)
        # Set thread as daemon so thread is terminated when main thread ends
        self.thread.daemon = True
        self.thread.start()
        self.thread.do_run = True

    # Thread that runs server
    def run(self):
        self.currThread = threading.currentThread()
        self.back_button.config(state=DISABLED)
        # Run script while the thread attribute is true
        while getattr(self.currThread, "do_run", True):
            # Want to catch return value error when thread ends and break loop
            try:
                self.read_from_pi()
                if self.currThread.do_run == False: break
                self.offset_cal()
                self.send_to_pi()

                self.message.set("Offset calibration is finished")
                self.update()
            except TypeError, e:
                print "Error:", e
                break

            self.stop_button.config(state=NORMAL)

    # Stops the thread process when stop is clicked
    def stop_script(self):
        self.message.set("Stopping server...")
        self.update()
        # Toggle the buttons
        self.back_button.config(state=NORMAL)
        self.stop_button.config(state=DISABLED)
        self.connect_button.config(state=NORMAL)
        # Set the thread attribute to False to signal stop
        self.thread.do_run = False
        # Join thread with main thread to end thread
        threading.Thread.join(self.thread, 1)
        self.message.set("Stopped server")
        self.update()



if __name__=="__main__":
    app = BackgroundW(None)
    #app.configure(background='black')
    app.mainloop()
