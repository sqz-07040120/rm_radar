import threading

import crc
import queue

import serial

import converts


seq = 0
sleep_time = 0.1
SOF = b'\xA5'


def Add_SOF(length: int) -> bytes:
    buff = SOF + converts.int22bytes(length)+converts.int21bytes(seq)
    crc_buff = crc.GetCRC8CheckSum(buff)
    crc_buff = converts.int21bytes(crc_buff)
    buff = buff + crc_buff
    return buff

def Add_TOF(buff: bytes) -> bytes:
    return converts.int22bytes(crc.GetCRC16CheckSum(buff))

class serial_port:
    def __init__(self, port: str,rx_able:bool = True,tx_able:bool = True,transmit_callback_func=None,receive_callback_func = None) -> None:
        '''init a serial port'''
        self.sp = serial.Serial(port, baudrate=115200)
        self.receive_callback_func = receive_callback_func
        self.transmit_callback_func = transmit_callback_func
        if rx_able == True:
            self.rx_buff = queue.Queue()
            self.rx_thread = threading.Thread(target=self.__receive_thread)
            self.rx_thread.start()
        if tx_able == True:
            self.tx_buff = queue.Queue()
            self.tx_thread = threading.Thread(target=self.__write_thread)
            self.tx_thread.start()

    def read_rx(self,timeout = 500) -> bytes:
        buff = self.rx_buff.get(timeout)
        return buff

    def write_tx(self, message: bytes) -> None:
        if self.sp.is_open == False:
            self.sp.open()
        self.tx_buff.put(message)

    def __write_thread(self):
        while self.sp.is_open == False:
            pass
        print('transmit thread is running')
        while True:
            buff = self.tx_buff.get()
            buff = Add_SOF(len(buff)-2) + buff
            buff = buff + Add_TOF(buff)
            self.sp.write(buff)
            self.sp.flushOutput()
            if self.transmit_callback_func != None:
                self(self.transmit_callback_func(buff))

    def __private_get_a_frame(self):
        sof = bytes()
        buff = bytes()
        sof = self.sp.read(4)
        if sof[3] == crc.GetCRC8CheckSum(SOF+sof[:3]):
            length = sof[1] << 8 | sof[0]
            buff = self.sp.read(length+4)
            if self.decode_func != None:
                self.receive_callback_func(buff)

    def __receive_thread(self):
        print('receive thread is running')
        while True:
            buff = self.sp.read(1)
            if buff == SOF:
                self.__private_get_a_frame()