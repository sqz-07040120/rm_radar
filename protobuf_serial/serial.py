import serial
import protobuf_serial.message_pb2 as mp

def serial_open(serialPort,baudRate):
    # 打开串口
    ser = serial.Serial(serialPort, baudRate, parity=serial.PARITY_ODD, stopbits=serial.STOPBITS_TWO,
                        bytesize=serial.EIGHTBITS)
    print(serialPort,baudRate)
    print("参数设置：串口={{}} ，波特率={{}}".format(serialPort, baudRate))
    return ser

def proto_FLyslope_serial(contours,ser):
    if (len(contours) != 0):
        Flyslope = mp.Flyslope()
        Flyslope.FSalarm = "1"
        mess = Flyslope.SerializeToString()
        # f = open("飞坡警告", "wb")
        # f.write(mess)
        # f.close()
        # serial_trans(mess,ser)
        ser.write(mess)
        # print(ser.readline())#可以接收中文
        # ser.close()

def proto_Energy_serial(contours,ser):
    if (len(contours) != 0):
        Energyorgan = mp.Energyorgan()
        Energyorgan.EOalarm = "2"
        mess = Energyorgan.SerializeToString()
        # f = open("能量机关警告", "wb")
        # f.write(mess)
        # f.close()
        ser.write(mess)


# 生成pb2文件
# So, for example, let's say you invoke the compiler as follows:
#
# protoc --proto_path=src --python_out=build/gen src/foo.proto src/bar/baz.proto

# The compiler will read the files src/foo.proto and src/bar/baz.proto and produce two output files: build/gen/foo_pb2.py and build/gen/bar/baz_pb2.py.
# The compiler will automatically create the directory build/gen/bar if necessary, but it will not create build or build/gen; they must already exist.
# Note that if the .proto file or its path contains any characters which cannot be used in Python module names (for example, hyphens),
# they will be replaced with underscores. So, the file foo-bar.proto becomes the Python file foo_bar_pb2.py.

# eg:    protoc --proto_path=project_protobuf_serial --python_out=project_protobuf_serial project_protobuf_serial/project.proto